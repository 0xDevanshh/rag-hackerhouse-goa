"""
Speech-to-text (STT) module.

Wraps the Sarvam AI speech-to-text API to transcribe spoken audio into text,
so the rest of the pipeline can operate on a text query.
"""


def transcribe_audio(audio_path: str, language: str = "en-IN") -> str:
    """
    Transcribe an audio file to text using the Sarvam STT API.

    Args:
        audio_path: filesystem path to the input audio file (e.g. wav/mp3).
        language: language/locale code to hint the STT model.

    Returns:
        str: the transcribed text.
    """
    raise NotImplementedError


def transcribe_bytes(audio_bytes: bytes, language: str = "en-IN") -> str:
    """
    Transcribe raw audio bytes (e.g. from a microphone stream or upload) to text.

    Args:
        audio_bytes: raw audio content.
        language: language/locale code to hint the STT model.

    Returns:
        str: the transcribed text.
    """
    raise NotImplementedError
