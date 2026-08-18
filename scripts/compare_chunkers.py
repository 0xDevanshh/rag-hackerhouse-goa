"""
Compares the four chunking strategies on data/sample_corpus.json.

Runs FixedSizeChunker, SentenceSemanticChunker, MetadataAwareChunker, and
RecursiveChunker over every document in the sample corpus and prints, per
strategy: chunk count, average chunk length, and one sample chunk.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import ChunkerRegistry  # noqa: E402

CORPUS_PATH = ROOT / "data" / "sample_corpus.json"

STRATEGIES = [
    ("fixed_size", {"chunk_size": 200, "overlap": 20}),
    ("sentence_semantic", {"similarity_threshold": 0.75}),
    ("metadata_aware", {}),
    ("recursive", {"max_chunk_size": 200}),
]


def main() -> None:
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    for name, params in STRATEGIES:
        chunker = ChunkerRegistry.build(name, **params)
        chunks = [c for doc in documents for c in chunker.chunk(doc)]

        print(f"=== {name} ===")
        print(f"chunk count: {len(chunks)}")
        if chunks:
            avg_len = sum(len(c.text) for c in chunks) / len(chunks)
            print(f"avg chunk length: {avg_len:.1f} chars")
            print(f"sample chunk: {chunks[0].model_dump()}")
        else:
            print("avg chunk length: n/a (no chunks produced)")
        print()


if __name__ == "__main__":
    main()
