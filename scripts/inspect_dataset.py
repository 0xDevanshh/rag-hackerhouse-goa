"""
Inspects raw ai4bharat/MSMARCO-XI examples before any pydantic mapping.

Streams a few examples directly from the dataset (no full download) and
prints their raw dict structure, so the shape of fields like `passages`
(parallel arrays vs. a list of dicts) can be visually confirmed before
trusting src/data_loader.py's mapping into QueryDoc.
"""

import itertools
import pprint

from datasets import load_dataset

LANGUAGE = "hi"
SPLIT = "validation"
NUM_SAMPLES = 3


def main() -> None:
    dataset = load_dataset("ai4bharat/MSMARCO-XI", LANGUAGE, split=SPLIT, streaming=True)
    for i, example in enumerate(itertools.islice(dataset, NUM_SAMPLES)):
        print(f"--- example {i} ---")
        pprint.pprint(example)
        print()


if __name__ == "__main__":
    main()
