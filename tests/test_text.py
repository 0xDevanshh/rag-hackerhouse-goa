"""
Unit tests for src/text.py.

These pin the Indic sentence terminators. Chunking, grounding checks, and
streaming all split on this one definition, and matching only [.!?] silently
treats a whole multi-sentence Hindi passage as one sentence — which
collapses GroundingGuardrail's unsupported-sentence ratio to all-or-nothing
and stops streaming from emitting until the answer is complete.
"""

import pytest

from src.text import split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three?", 3),
        ("निगम एक कानूनी संस्था है। यह अपने मालिकों से अलग है। इसके शेयरधारक होते हैं।", 3),
        ("यह एक जملہ ہے۔ یہ دوسرا ہے۔", 2),
        ("श्लोक एक॥ श्लोक दो॥", 2),
        ("Mixed script. हिंदी वाक्य। Back to English.", 3),
    ],
)
def test_splits_on_latin_and_indic_terminators(text, expected):
    assert len(split_sentences(text)) == expected


def test_retains_terminator_on_the_sentence_it_ends():
    assert split_sentences("पहला वाक्य। दूसरा वाक्य।") == ["पहला वाक्य।", "दूसरा वाक्य।"]


def test_single_sentence_without_trailing_whitespace_is_one_sentence():
    assert split_sentences("कॉर्पोरेशन क्या है?") == ["कॉर्पोरेशन क्या है?"]


def test_drops_empty_fragments_and_surrounding_whitespace():
    assert split_sentences("  A.   B.  ") == ["A.", "B."]


def test_empty_input_yields_no_sentences():
    assert split_sentences("   ") == []


def test_danda_without_following_space_does_not_split():
    # The boundary requires whitespace after the terminator, so an abbreviation
    # or a mid-token mark doesn't fragment the sentence.
    assert len(split_sentences("क।ख")) == 1
