"""
Inspects raw ai4bharat/MSMARCO-XI examples before any pydantic mapping.

Streams a few examples directly from the dataset (no full download) and
prints their raw dict structure, so the shape of fields like `passages`
(parallel arrays vs. a list of dicts) can be visually confirmed before
trusting src/data_loader.py's mapping into QueryDoc.

Language selection reuses src.data_loader._data_file, so this script and the
loader address exactly the same parquet — the dataset has no per-language
config, and omitting the config silently yields Assamese (see the
src/data_loader.py module docstring).
"""

import itertools
import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.data_loader import DATASET_ID, _data_file  # noqa: E402

LANGUAGE = "hi"
SPLIT = "validation"
NUM_SAMPLES = 3


def main() -> None:
    data_file = _data_file(LANGUAGE, SPLIT)
    print(f"streaming {DATASET_ID}:{data_file}\n")
    dataset = load_dataset(
        DATASET_ID,
        data_files={SPLIT: data_file},
        split=SPLIT,
        streaming=True,
    )
    for i, example in enumerate(itertools.islice(dataset, NUM_SAMPLES)):
        print(f"--- example {i} ---")
        pprint.pprint(example)
        print()


if __name__ == "__main__":
    main()
