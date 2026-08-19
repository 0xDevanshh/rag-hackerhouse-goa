"""
Speech-to-text (STT) module.

SarvamSTT wraps Sarvam AI's speech-to-text REST API (the Saaras model —
Sarvam's current transcription model; Saarika has been retired/renamed to
Saaras in their current API). MockSTT implements the same interface with
canned transcripts, for testing without hitting the real API or a mic.
"""

import os
from pathlib import Path

import requests
from pydantic import BaseModel

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "saaras:v3"

# Sarvam validates the multipart part's Content-Type against an explicit
# allow-list and rejects the request outright if it's missing (`None`) —
# relying on `requests`/`mimetypes` to guess it from the filename is not
# reliable across systems (e.g. ".webm" isn't registered everywhere), so we
# resolve it ourselves and always send an explicit, real Content-Type.
_EXTENSION_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/x-m4a",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
    ".amr": "audio/amr",
    ".wma": "audio/x-ms-wma",
    ".aiff": "audio/aiff",
}


def _resolve_content_type(filename: str, content_type: str | None) -> str:
    """
    Pick the Content-Type to send with the audio part: prefer an explicit
    one (e.g. the browser's real Blob.type), normalized to just the base
    MIME type (drop any ";codecs=..." parameter, since Sarvam's allow-list
    matches exact values); otherwise derive it from the filename extension;
    otherwise fall back to "application/octet-stream" (itself allow-listed).
    """
    if content_type:
        return content_type.split(";")[0].strip()
    return _EXTENSION_CONTENT_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")


class Timestamps(BaseModel):
    """Chunk-level timestamps for a transcript, when requested."""

    words: list[str] = []
    start_time_seconds: list[float] = []
    end_time_seconds: list[float] = []


class TranscriptResult(BaseModel):
    """The result of a speech-to-text transcription."""

    transcript: str
    language_code: str | None = None
    language_probability: float | None = None
    request_id: str | None = None
    timestamps: Timestamps | None = None


class SarvamSTT:
    """Transcribes audio via Sarvam AI's speech-to-text API (Saaras model)."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 30.0):
        """
        Args:
            model: Sarvam STT model ID ("saaras:v3" or "saaras:v4").
            timeout: request timeout in seconds.
        """
        api_key = os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY environment variable is not set")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        filename: str = "audio.wav",
        content_type: str | None = None,
    ) -> TranscriptResult:
        """
        Transcribe audio bytes via Sarvam's speech-to-text API.

        Args:
            audio_bytes: raw audio file content (WAV, MP3, OGG, FLAC, M4A, etc.).
            language_code: BCP-47 language code (e.g. "hi-IN"), or "unknown"
                to let Sarvam auto-detect the spoken language.
            filename: filename (with a real audio extension, e.g. "recording.webm")
                sent as part of the multipart upload.
            content_type: the audio's real MIME type (e.g. "audio/webm"), if
                known — Sarvam validates this against an allow-list and
                rejects the request if it's missing, so this must always
                resolve to something. Falls back to a guess from `filename`'s
                extension, then to "application/octet-stream".

        Returns:
            TranscriptResult: the transcript plus detected language info.

        Raises:
            requests.HTTPError: if the Sarvam API returns an error response.
        """
        resolved_content_type = _resolve_content_type(filename, content_type)
        response = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": self.api_key},
            data={
                "model": self.model,
                "language_code": language_code,
                "mode": "transcribe",
            },
            files={"file": (filename, audio_bytes, resolved_content_type)},
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(f"{exc} — response body: {response.text}", response=response) from exc
        payload = response.json()

        timestamps = payload.get("timestamps")
        return TranscriptResult(
            transcript=payload["transcript"],
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            request_id=payload.get("request_id"),
            timestamps=Timestamps(**timestamps) if timestamps else None,
        )


class MockSTT:
    """
    Drop-in stand-in for SarvamSTT with the same interface, returning canned
    transcripts instead of calling the real API. For tests and local
    development without a Sarvam API key or a microphone.
    """

    DEFAULT_TRANSCRIPTS = {
        "unknown": "what is retrieval augmented generation",
        "en-IN": "what is retrieval augmented generation",
        "hi-IN": "यह एक हिंदी में प्रश्न है",
        "ta-IN": "இது தமிழில் ஒரு கேள்வி",
        "bn-IN": "এটি বাংলায় একটি প্রশ্ন",
    }

    def __init__(
        self,
        transcripts: dict[str, str] | None = None,
        default_transcript: str = "This is a mock transcript.",
    ):
        """
        Args:
            transcripts: overrides/additions to the canned per-language-code
                transcript map.
            default_transcript: returned when language_code isn't in the map.
        """
        self.transcripts = {**self.DEFAULT_TRANSCRIPTS, **(transcripts or {})}
        self.default_transcript = default_transcript

    def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        filename: str = "audio.wav",
        content_type: str | None = None,
    ) -> TranscriptResult:
        """
        Return a canned TranscriptResult without making any network call.

        Args:
            audio_bytes: ignored; accepted only to match SarvamSTT's interface.
            language_code: selects which canned transcript to return.
            filename: ignored; accepted only to match SarvamSTT's interface.
            content_type: ignored; accepted only to match SarvamSTT's interface.

        Returns:
            TranscriptResult: a canned transcript for language_code (or
            default_transcript if language_code is unrecognized).
        """
        transcript = self.transcripts.get(language_code, self.default_transcript)
        resolved_language = "en-IN" if language_code == "unknown" else language_code
        return TranscriptResult(
            transcript=transcript,
            language_code=resolved_language,
            language_probability=1.0,
            request_id="mock-request-id",
        )
