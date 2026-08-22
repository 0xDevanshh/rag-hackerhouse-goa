"""Build the versioned FAISS/BM25 artifact set consumed at API startup."""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss

# Support direct execution from the repository root: Python otherwise puts
# scripts/ first on sys.path and cannot resolve the src package.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import data_loader
from src.chunking import ChunkerRegistry
from src.retrieval import BM25Index
from src.vectorstore import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--no-english", action="store_true")
    args = parser.parse_args()

    documents = data_loader.load_chunker_docs(
        language=args.language,
        split=args.split,
        limit=args.limit,
        include_english=not args.no_english,
    )
    chunker = ChunkerRegistry.build("metadata_aware")
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]

    store = VectorStore()
    store.build(chunks)
    retriever_index = BM25Index(chunks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(args.output_dir / "faiss.index"))
    with (args.output_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(retriever_index, handle)
    (args.output_dir / "chunks.json").write_text(
        json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {"language": args.language, "split": args.split, "include_english": not args.no_english},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": args.version,
                "chunks": len(chunks),
                "embedding_model": store.embedder.model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(chunks)} chunks to {args.output_dir}")


if __name__ == "__main__":
    main()