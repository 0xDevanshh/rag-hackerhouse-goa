"""Load versioned FAISS/BM25 artifacts from S3-compatible object storage."""

import json
import logging
import os
import pickle
import tempfile
from pathlib import Path

import boto3
import faiss

from src.chunking import Chunk
from src.retrieval import BM25Index, Retriever
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """Raised when the configured RAG artifact set is missing or invalid."""


def artifact_storage_configured() -> bool:
    """Return whether startup should load a prebuilt artifact set."""
    return bool(os.getenv("RAG_ARTIFACT_BUCKET") and os.getenv("RAG_ARTIFACT_PREFIX"))


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        region_name=os.getenv("S3_REGION") or os.getenv("AWS_REGION") or None,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or None,
    )


def _download(client, bucket: str, prefix: str, name: str, directory: Path) -> Path:
    target = directory / name
    key = f"{prefix.strip('/')}/{name}"
    try:
        client.download_file(bucket, key, str(target))
    except Exception as exc:
        raise ArtifactError(f"could not download artifact {key!r}: {exc}") from exc
    return target


def load_artifacts(store: VectorStore, retriever: Retriever) -> dict:
    """Download and validate one complete artifact set into process memory."""
    bucket = os.environ["RAG_ARTIFACT_BUCKET"]
    prefix = os.environ["RAG_ARTIFACT_PREFIX"]
    expected_version = os.getenv("RAG_ARTIFACT_VERSION")
    expected_model = os.getenv("EMBEDDING_MODEL") or store.embedder.model_name

    with tempfile.TemporaryDirectory(prefix="rag-artifacts-") as temp_dir:
        directory = Path(temp_dir)
        client = _s3_client()
        manifest_path = _download(client, bucket, prefix, "manifest.json", directory)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid manifest.json: {exc}") from exc

        version = manifest.get("version")
        if not version:
            raise ArtifactError("manifest.json is missing required field 'version'")
        if expected_version and version != expected_version:
            raise ArtifactError(f"artifact version {version!r} does not match RAG_ARTIFACT_VERSION {expected_version!r}")
        model = manifest.get("embedding_model")
        if model and model != expected_model:
            raise ArtifactError(f"artifact model {model!r} does not match configured model {expected_model!r}")

        paths = {
            name: _download(client, bucket, prefix, name, directory)
            for name in ("faiss.index", "bm25.pkl", "chunks.json", "metadata.json")
        }
        try:
            chunks_data = json.loads(paths["chunks.json"].read_text(encoding="utf-8"))
            metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
            chunks = [Chunk.model_validate(item) for item in chunks_data]
            index = faiss.read_index(str(paths["faiss.index"]))
            with paths["bm25.pkl"].open("rb") as handle:
                bm25 = pickle.load(handle)
        except Exception as exc:
            raise ArtifactError(f"invalid RAG artifact payload: {exc}") from exc

        if not isinstance(chunks_data, list) or not isinstance(metadata, (dict, list)):
            raise ArtifactError("chunks.json must be a list and metadata.json must be an object or list")
        if not isinstance(bm25, BM25Index):
            raise ArtifactError("bm25.pkl does not contain a compatible BM25Index")
        if index.ntotal != len(chunks) or len(bm25.chunks) != len(chunks):
            raise ArtifactError("FAISS, BM25, and chunks artifact counts do not match")
        if manifest.get("chunks") is not None and int(manifest["chunks"]) != len(chunks):
            raise ArtifactError("manifest chunk count does not match downloaded artifacts")

        store.index = index
        store.chunks = chunks
        retriever._bm25 = bm25
        retriever._indexed_chunk_count = len(chunks)
        logger.info("loaded RAG artifact version=%s chunks=%d", version, len(chunks))
        return manifest