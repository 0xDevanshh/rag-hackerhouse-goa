"""
Shared text utilities.

Holds the one definition of a sentence boundary used across the pipeline, so
chunking, grounding checks, and streaming all agree on where a sentence ends.
"""

import re

# Sentence-final punctuation, Latin plus the Indic and Urdu marks the corpus
# actually contains. Devanagari and most Indic scripts end sentences with the
# danda "।" (U+0964), not a period, and Urdu uses the Arabic full stop "۔"
# (U+06D4) and question mark "؟" (U+061F). Matching only [.!?] would treat a
# whole multi-sentence Hindi passage or answer as a single sentence, which
# silently degrades every consumer below — see split_sentences.
SENTENCE_TERMINATORS = ".!?।॥۔؟"

# A boundary is a terminator followed by whitespace. Kept as a lookbehind so
# re.split() leaves the terminator attached to the sentence it ends.
SENTENCE_BOUNDARY_RE = re.compile(rf"(?<=[{re.escape(SENTENCE_TERMINATORS)}])\s+")


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on any terminator in SENTENCE_TERMINATORS
    followed by whitespace, dropping empty fragments.

    Args:
        text: the text to split.

    Returns:
        list[str]: the sentences, terminators retained.
    """
    return [s.strip() for s in SENTENCE_BOUNDARY_RE.split(text.strip()) if s.strip()]
