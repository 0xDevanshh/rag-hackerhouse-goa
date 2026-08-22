"""
Vector store module.

Embeds text chunks with a sentence-transformers model and indexes them in a
FAISS index for fast approximate nearest-neighbor similarity search.
"""

import hashlib
import logging
import os
import pickle
from collections import OrderedDict
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


def _int_env(name: str, default: int) -> int:
    """
    Read an int from the environment, falling back to `default` for an unset,
    empty, or unparseable value. An empty string is the case that matters:
    `EMBEDDING_TORCH_THREADS=` in a shell or compose file is a very easy thing
    to write, and a bare int() on it raises at import time — turning a
    harmless config typo into a process that won't start.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r; using %d", name, raw, default)
        return default


def _configure_cpu_threads() -> None:
    """
    Cap torch's intra-op thread count before the model is loaded — see
    EMBEDDING_TORCH_THREADS for why this is required rather than a tuning
    preference.

    Called at model-load time (not import time) so importing this module
    doesn't silently reconfigure torch for an unrelated caller, and only for
    the CPU path, since the MPS path doesn't enter torch's OpenMP regions.
    """
    if torch.get_num_threads() != EMBEDDING_TORCH_THREADS:
        torch.set_num_threads(EMBEDDING_TORCH_THREADS)


DEFAULT_EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

# Device the embedding model runs on.
#
# Pinned to CPU by default, deliberately. sentence-transformers >= 3 picks the
# "best" available device automatically, which on Apple silicon means MPS —
# and for the single-short-query encodes this pipeline actually does per
# request, MPS is both slower and far less predictable than CPU. Measured on
# an M4 with this model and faiss loaded in the same process, batch size 1,
# varying query lengths, 60 encodes:
#
#   MPS:            p50 12.9ms  p95 23.8ms  p99 44.3ms  max   68.7ms
#   CPU (1 thread): p50  5.1ms  p95  6.9ms  p99  9.7ms  max   10.5ms
#
# and in an earlier run without faiss present, MPS reached max 1092ms. The MPS
# tail is Metal kernel compilation: each distinct padded input shape is a new
# kernel, compiled on first use. Query length varies per request, so novel
# shapes keep appearing in production forever and each one lands as a spike
# inside a live request. That is the mechanism behind the ~320ms embedding_ms
# in the original production trace.
#
# Bulk indexing is the opposite workload (thousands of texts, large batches),
# where MPS is ~1.6x faster — 9.7s vs 15.6s for this 9990-chunk corpus. That
# is a one-time startup cost, so it is the right thing to trade away for a
# tighter per-request tail, but EMBEDDING_DEVICE stays configurable for a
# deployment that would rather start faster. CPU and MPS vectors agree to
# cosine 1.000000 (max elementwise difference 1.7e-07), so switching devices
# does *not* invalidate a persisted index.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")

# Intra-op thread count for CPU inference. This is load-bearing for two
# independent reasons, which is why it is 1 and not the torch default of 4:
#
# 1. Correctness/stability. faiss-cpu bundles its own libomp.dylib and torch
#    brings another OpenMP runtime; on macOS/arm64, running torch's
#    multi-threaded CPU op alongside faiss in one process reliably
#    SIGSEGVs (exit 139) during a bulk encode. Import order makes no
#    difference and KMP_DUPLICATE_LIB_OK=TRUE does not help; limiting torch to
#    one intra-op thread does, because it stops torch entering a parallel
#    region at all.
# 2. Latency. At batch size 1 with a short query there is not enough work to
#    parallelize, so extra threads only add synchronization overhead. A
#    thread-count sweep measured p50 within noise across 1/2/4/6/8 threads
#    (5.02 / 4.88 / 5.11 / 4.90 / 5.50 ms) — single-threaded is as fast as any
#    of them for the shape this pipeline actually serves.
#
# The cost is the bulk index build, which is genuinely serial at 1 thread
# (~15.6s for 9990 chunks vs ~9.7s on MPS). Raising this trades the segfault
# risk back on; if a deployment needs a faster build, prefer
# EMBEDDING_DEVICE=mps over raising the CPU thread count.
EMBEDDING_TORCH_THREADS = _int_env("EMBEDDING_TORCH_THREADS", 1)

# Query embeddings worth keeping hot. Sized for the repeated/FAQ-style query
# traffic this cache exists to serve; chunk-indexing calls bypass it entirely
# (see Embedder.encode's use_cache) so a 10k-chunk index build can't evict
# every query entry on startup.
EMBEDDING_CACHE_MAX_SIZE = _int_env("EMBEDDING_CACHE_MAX_SIZE", 2048)

# Texts used to warm the model at startup. The point is not correctness but
# shape coverage: the first encode of a process pays ~690ms of lazy
# initialization (kernel/graph setup, allocator warmup), and on MPS each new
# padded sequence length pays again. Warming across a spread of lengths and
# scripts moves that cost out of the first real request's budget.
_WARMUP_TEXTS = (
    "warmup",
    "what is retrieval augmented generation",
    "how does a vector database answer nearest neighbour queries over dense embeddings",
    "पुनर्प्राप्ति संवर्धित पीढ़ी क्या है",
    "वेक्टर डेटाबेस कैसे काम करता है और यह जानकारी पुनर्प्राप्ति के लिए क्यों उपयोगी है",
    "இது தமிழில் ஒரு கேள்வி",
    "এটি বাংলায় একটি প্রশ্ন",
    " ".join(["a padded sentence to cover a longer sequence bucket"] * 6),
)


def normalize_text(text: str) -> str:
    """
    Canonical form used for every cache key in the pipeline (embeddings here,
    answers in the harness), so the two layers agree on what counts as "the
    same query". Collapses whitespace runs and case only — trivial phrasing
    differences share an entry, and the vectors for such near-identical inputs
    are effectively identical anyway, so this costs no retrieval quality.
    """
    return " ".join(text.split()).lower()


def text_fingerprint(text: str) -> str:
    """
    Short stable digest of a normalized text, used as a cache key instead of
    the text itself. Keeps key size constant regardless of query length and
    avoids holding a second copy of every cached query in memory.
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:32]


class Embedder:
    """
    Wraps a sentence-transformers embedding model for batch encoding into
    L2-normalized vectors. Defaults to "paraphrase-multilingual-MiniLM-L12-v2"
    for multilingual (e.g. Hindi/Tamil/Bengali) support.

    The underlying SentenceTransformer model is cached as a singleton per
    (model name, device) at the class level, so it is loaded once per process
    no matter how many Embedder instances are constructed — loading costs
    ~0.8-1.6s and ~470MB, so a per-request or per-component load would dwarf
    every other stage in the pipeline.
    """

    _model_cache: dict[tuple[str, str], SentenceTransformer] = {}

    # Process-local LRU embedding cache, keyed by
    # (model_name, device, sha256(normalized_text)).
    #
    # LRU rather than the previous FIFO: under FIFO a hot query that keeps
    # getting hits is still evicted on schedule by a stream of one-off
    # queries, which is precisely backwards for a cache whose job is to keep
    # repeated queries hot.
    #
    # Process-local because this project has no Redis. That is a real
    # limitation, not a design choice: with more than one backend instance,
    # each pays its own miss for the same query. The key is already a
    # content-addressed digest, so moving this to a shared store is a matter
    # of swapping the dict for a client, not redesigning the keys.
    _embedding_cache: "OrderedDict[tuple[str, str, str], np.ndarray]" = OrderedDict()
    _cache_hits = 0
    _cache_misses = 0

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or EMBEDDING_DEVICE
        key = (model_name, self.device)
        if key not in self._model_cache:
            if self.device == "cpu":
                _configure_cpu_threads()
            logger.info(
                "loading embedding model %s on device=%s (torch intra-op threads=%d)",
                model_name,
                self.device,
                torch.get_num_threads(),
            )
            self._model_cache[key] = SentenceTransformer(model_name, device=self.device)
            self._model_cache[key].max_seq_length = min(self._model_cache[key].max_seq_length, 64)
            self._model_cache[key].eval()
        self._model = self._model_cache[key]

    def _cache_key(self, text: str) -> tuple[str, str, str]:
        return (self.model_name, self.device, text_fingerprint(text))

    @classmethod
    def cache_stats(cls) -> dict[str, int | float]:
        """Hit/miss counters for the process-local embedding cache."""
        total = cls._cache_hits + cls._cache_misses
        return {
            "hits": cls._cache_hits,
            "misses": cls._cache_misses,
            "size": len(cls._embedding_cache),
            "hit_rate": cls._cache_hits / total if total else 0.0,
        }

    def warmup(self) -> float:
        """
        Run the model once per warmup shape so no live request pays lazy
        initialization. Returns the wall-clock cost in ms.

        Called at process startup (see src/api.py's lifespan). Deliberately
        bypasses the cache — the point is to warm the *model*, and caching
        these synthetic texts would only occupy cache slots that real queries
        should have.
        """
        import time

        started = time.perf_counter()
        for text in _WARMUP_TEXTS:
            self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("embedding model warmup: %.1f ms over %d shapes", elapsed, len(_WARMUP_TEXTS))
        return elapsed

    def encode(
        self,
        texts: list[str],
        use_cache: bool = True,
        timing: dict[str, float] | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Batch-encode texts into L2-normalized embedding vectors, reusing a
        cached vector for any text seen before (see _embedding_cache).

        Args:
            texts: the texts to embed.
            use_cache: consult and populate the process-local cache. Pass
                False for bulk indexing: every chunk text is unique, so the
                lookups all miss, and writing ~10k one-shot entries would
                evict the query embeddings the cache exists to serve.
            timing: optional sink for "embedding_cache_ms" (lookup cost) and
                "embedding_compute_ms" (actual model inference), so a caller's
                latency trace can separate a cache hit from real work instead
                of reporting one opaque embedding_ms.
            batch_size: forwarded to the model for the uncached remainder.

        Returns:
            np.ndarray: array of shape (len(texts), embedding_dim), where
            each row is L2-normalized.
        """
        import time

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        to_compute_idx: list[int] = []
        to_compute_texts: list[str] = []

        lookup_started = time.perf_counter()
        if use_cache:
            for i, text in enumerate(texts):
                key = self._cache_key(text)
                cached = self._embedding_cache.get(key)
                if cached is not None:
                    self._embedding_cache.move_to_end(key)  # LRU touch
                    results[i] = cached
                    type(self)._cache_hits += 1
                else:
                    type(self)._cache_misses += 1
                    to_compute_idx.append(i)
                    to_compute_texts.append(text)
        else:
            to_compute_idx = list(range(len(texts)))
            to_compute_texts = list(texts)
        lookup_ms = (time.perf_counter() - lookup_started) * 1000

        compute_ms = 0.0
        if to_compute_texts:
            compute_started = time.perf_counter()
            computed = self._model.encode(
                to_compute_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            compute_ms = (time.perf_counter() - compute_started) * 1000
            for idx, text, vector in zip(to_compute_idx, to_compute_texts, computed):
                results[idx] = vector
                if use_cache:
                    self._store(self._cache_key(text), vector)

        if timing is not None:
            timing["embedding_cache_ms"] = timing.get("embedding_cache_ms", 0.0) + lookup_ms
            timing["embedding_compute_ms"] = timing.get("embedding_compute_ms", 0.0) + compute_ms

        return np.asarray(results, dtype=np.float32)

    def _store(self, key: tuple[str, str, str], vector: np.ndarray) -> None:
        cache = self._embedding_cache
        while len(cache) >= EMBEDDING_CACHE_MAX_SIZE:
            cache.popitem(last=False)  # evict least recently used
        cache[key] = vector

    @property
    def dimension(self) -> int:
        """Embedding dimensionality of the underlying model."""
        return int(self._model.get_sentence_embedding_dimension())


class VectorStore:
    """
    Cosine-similarity vector store: embeds Chunk objects with an Embedder and
    indexes them in a faiss.IndexFlatIP. Since Embedder outputs are
    L2-normalized, inner product over those vectors is equivalent to cosine
    similarity.
    """

    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME):
        """
        Args:
            embedding_model_name: name/path of the sentence-transformers model
                the internal Embedder uses to embed chunks and queries.
        """
        self.embedder = Embedder(embedding_model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """
        Embed the given chunks and build the FAISS index from scratch.

        Args:
            chunks: chunk records to index.
        """
        embeddings = np.asarray(
            # use_cache=False: see Embedder.encode — bulk indexing must not
            # evict the query cache. batch_size well above the per-query
            # default, since this path is throughput-bound, not latency-bound.
            self.embedder.encode([chunk.text for chunk in chunks], use_cache=False, batch_size=256),
            dtype=np.float32,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = list(chunks)

    @staticmethod
    def _matches_filter(chunk: Chunk, metadata_filter: dict[str, Any]) -> bool:
        return all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float, int]]:
        """
        Search for the top_k chunks most similar to query_embedding.

        When metadata_filter is given (e.g. {"language": "hi"}), it's
        applied as an exact-match filter on each candidate's chunk.metadata,
        post-search: top_k*4 candidates are over-fetched from FAISS first,
        then filtered, then truncated back down to top_k. Without a filter,
        exactly top_k candidates are fetched.

        Args:
            query_embedding: a single embedding vector, shape (dim,) or (1, dim).
            top_k: number of results to return.
            metadata_filter: optional exact-match filters on chunk metadata.

        Returns:
            list[tuple[Chunk, float, int]]: (chunk, similarity score, index
            position) triples, ordered by descending similarity. The index
            position is returned so callers can recover the chunk's stored
            embedding via embeddings_for() instead of re-encoding its text —
            see GroundingGuardrail, which used to spend ~17-23ms per request
            re-embedding text whose vector was already sitting in this index.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        fetch_k = min(top_k * 4 if metadata_filter else top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                continue
            results.append((chunk, float(score), int(idx)))

        return results[:top_k]

    def embeddings_for(self, positions: list[int]) -> np.ndarray:
        """
        Recover the stored (already L2-normalized) embeddings for index
        positions returned by search().

        This is the cheap way to get a retrieved chunk's vector: the vector
        was computed once at index-build time and lives in the FAISS index, so
        reading it back is a memory copy rather than a model forward pass.

        Args:
            positions: index positions, as returned by search().

        Returns:
            np.ndarray: shape (len(positions), dim), rows in the given order.
        """
        if self.index is None or not positions:
            return np.empty((0, self.embedder.dimension), dtype=np.float32)
        return np.vstack([self.index.reconstruct(int(pos)) for pos in positions]).astype(np.float32)

    def save(self, index_path: str) -> None:
        """
        Persist the FAISS index to "{index_path}.faiss" and the chunk
        metadata to a pickled sidecar at "{index_path}.meta.pkl".

        Args:
            index_path: filesystem path/prefix to save the files under.
        """
        if self.index is None:
            raise ValueError("cannot save an empty VectorStore; call build() first")
        faiss.write_index(self.index, f"{index_path}.faiss")
        with open(f"{index_path}.meta.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, index_path: str) -> None:
        """
        Load a previously persisted FAISS index and its chunk metadata sidecar.

        Args:
            index_path: filesystem path/prefix the files were saved under
                (the same prefix passed to save()).
        """
        self.index = faiss.read_index(f"{index_path}.faiss")
        with open(f"{index_path}.meta.pkl", "rb") as f:
            self.chunks = pickle.load(f)
