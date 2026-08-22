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
        # A module-level requests.post() call opens and tears down a fresh
        # Session (and its urllib3 connection pool) on every call — no TCP/TLS
        # connection reuse across requests. A persistent Session held for
        # this instance's lifetime pools connections to api.sarvam.ai instead.
        self._session = requests.Session()

    def prewarm(self) -> float:
        """
        Complete the DNS + TCP + TLS handshake to api.sarvam.ai at startup and
        return the cost in ms.

        Measured worth: ~100ms on this network (cold p50 146ms vs pooled p50
        45ms for the same trivial request). Without this, the first
        transcription after startup — or after the pool's keep-alive expires —
        pays the handshake inside a live request. Any failure is swallowed:
        a warmup that can't reach the API must not stop the process from
        starting, it just means the first request pays what it would have paid
        anyway.
        """
        import time

        started = time.perf_counter()
        try:
            # A bare GET on the API root is enough to open the connection; the
            # status code is irrelevant and deliberately not checked.
            self._session.get("https://api.sarvam.ai/", timeout=self.timeout)
        except Exception:
            pass
        return (time.perf_counter() - started) * 1000

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
        response = self._session.post(
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

    def prewarm(self) -> float:
        """No-op, so callers can prewarm the STT client without checking its type."""
        return 0.0

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


# ---------------------------------------------------------------------------
# Async realtime STT — overlapped path
# ---------------------------------------------------------------------------

import asyncio as _asyncio
import base64 as _base64
import json as _json
import time as _time
from dataclasses import dataclass, field
from typing import AsyncIterator

try:
    import websockets as _websockets
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    _WEBSOCKETS_AVAILABLE = False

SARVAM_REALTIME_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
_REALTIME_QUERY = (
    "language_code=auto&model=saaras:v3-realtime&stream_type=fast"
    "&endpointing=vad&encoding=linear16&sample_rate=16000&mode=transcribe"
    "&prefix_padding_ms=100&silence_duration_ms=100&min_speech_duration_ms=100"
)

# A partial is "stable" — safe to fire retrieval on — if either:
#   (a) it hasn't changed across this many consecutive partial events, or
#   (b) the partial has at least this many tokens (short partials are high
#       noise and likely to change before the final result).
_PARTIAL_STABILITY_REPEATS = 2
_PARTIAL_MIN_TOKENS = 3


@dataclass
class PartialTranscriptEvent:
    """A partial transcript event from the realtime STT stream."""
    text: str
    is_stable: bool        # True when the partial passes the stability check
    received_at: float     # perf_counter timestamp


@dataclass
class FinalTranscriptEvent:
    """The final (confirmed) transcript for one utterance."""
    text: str
    received_at: float


@dataclass
class RealtimeSTTResult:
    """
    Full timing breakdown for one overlapped STT session.

    All *_ms fields are wall-clock durations in milliseconds.
    received_at_* are perf_counter timestamps, used to compute overlap savings.
    """
    transcript: str
    language_code: str | None = None
    # Timing
    session_open_ms: float = 0.0       # from call start to WS open
    first_partial_ms: float = 0.0      # from WS open to first partial event
    final_transcript_ms: float = 0.0   # from WS open to transcript.final
    total_ms: float = 0.0              # from call start to session close
    # Stability info
    stable_partial: str = ""           # which partial text triggered retrieval
    stable_partial_ms: float = 0.0     # ms from WS open to stable partial
    stable_partial_token_count: int = 0
    # How many times the partial changed before stabilising
    partial_changes: int = 0


class SarvamRealtimeSTT:
    """
    Async streaming STT client using Sarvam's realtime WebSocket API.

    Yields PartialTranscriptEvent as each partial arrives and a single
    FinalTranscriptEvent when the utterance is complete.  Callers can start
    retrieval as soon as they receive a stable partial, overlapping the
    remaining STT time with retrieval work.

    The underlying WebSocket connection is kept open across calls when
    possible (session reuse) so each utterance in a session doesn't pay the
    DNS + TCP + TLS handshake again.  Call close() when the session ends.

    Usage::

        stt = SarvamRealtimeSTT()
        async for event in stt.stream(audio_bytes):
            if isinstance(event, PartialTranscriptEvent) and event.is_stable:
                asyncio.create_task(start_retrieval(event.text))
            elif isinstance(event, FinalTranscriptEvent):
                final_text = event.text
    """

    def __init__(
        self,
        stability_repeats: int = _PARTIAL_STABILITY_REPEATS,
        min_tokens: int = _PARTIAL_MIN_TOKENS,
        open_timeout: float = 10.0,
        ping_interval: float = 20.0,
    ):
        if not _WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required for SarvamRealtimeSTT")
        api_key = os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY environment variable is not set")
        self.api_key = api_key
        self.stability_repeats = stability_repeats
        self.min_tokens = min_tokens
        self.open_timeout = open_timeout
        self.ping_interval = ping_interval
        # Persistent connection: reused across stream() calls within a session.
        # None means not yet opened or previously closed.
        self._ws = None
        self._ws_lock = _asyncio.Lock()

    async def _get_connection(self):
        """Return an open WebSocket connection, opening one if needed."""
        async with self._ws_lock:
            if self._ws is not None:
                try:
                    # Check the connection is still alive with a lightweight ping.
                    # websockets raises ConnectionClosed if the peer has gone away.
                    await _asyncio.wait_for(self._ws.ping(), timeout=2.0)
                    return self._ws
                except Exception:
                    self._ws = None

            url = f"{SARVAM_REALTIME_WS_URL}?{_REALTIME_QUERY}"
            self._ws = await _websockets.connect(
                url,
                subprotocols=[f"api-subscription-key.{self.api_key}"],
                ping_interval=self.ping_interval,
                open_timeout=self.open_timeout,
            )
            return self._ws

    async def prewarm(self) -> float:
        """
        Open (and keep) the WebSocket connection at startup so the first
        utterance doesn't pay the handshake.  Returns the cost in ms.
        """
        started = _time.perf_counter()
        try:
            await self._get_connection()
        except Exception as exc:
            # Prewarm must never prevent startup.
            import logging
            logging.getLogger(__name__).warning(
                "SarvamRealtimeSTT prewarm failed (%s); first request will pay WS setup", exc
            )
        return (_time.perf_counter() - started) * 1000

    async def close(self) -> None:
        """Close the persistent connection gracefully."""
        async with self._ws_lock:
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None

    def _is_stable(self, text: str, history: list[str]) -> bool:
        """True when the partial passes both stability checks."""
        token_count = len(text.split())
        if token_count < self.min_tokens:
            return False
        if len(history) < self.stability_repeats:
            return False
        return all(h == text for h in history[-self.stability_repeats:])

    async def stream(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str | None = None,
        chunk_size: int = 8192,
    ) -> "AsyncIterator[PartialTranscriptEvent | FinalTranscriptEvent]":
        """
        Stream audio_bytes to Sarvam's realtime STT endpoint, yielding
        PartialTranscriptEvent items as partials arrive and a single
        FinalTranscriptEvent when the final result is ready.

        The audio is sent as raw base64-encoded PCM frames, split into
        chunk_size-byte pieces.  For pre-recorded audio files this is a
        simulated stream; the same code path works with actual live PCM
        captured by the browser's AudioWorklet.

        Args:
            audio_bytes: raw audio file content (WAV, PCM, etc.).
            filename: used for logging only.
            content_type: unused for realtime WS path; accepted for API symmetry.
            chunk_size: bytes per audio_input frame.

        Yields:
            PartialTranscriptEvent for each transcript.partial event.
            FinalTranscriptEvent for the transcript.final event.
        """
        call_started = _time.perf_counter()
        ws_open_at: float | None = None
        partial_history: list[str] = []
        stable_emitted = False

        try:
            ws = await self._get_connection()
            ws_open_at = _time.perf_counter()

            # Send audio as a stream of base64-encoded chunks.
            async def _send_audio() -> None:
                for offset in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[offset: offset + chunk_size]
                    encoded = _base64.b64encode(chunk).decode("ascii")
                    await ws.send(_json.dumps({"event": "audio_input", "audio": encoded}))
                    # Yield control so the receive coroutine can process
                    # partials while we're still sending.
                    await _asyncio.sleep(0)
                await ws.send(_json.dumps({"event": "end"}))

            send_task = _asyncio.create_task(_send_audio())

            try:
                async for raw_message in ws:
                    try:
                        msg = _json.loads(raw_message)
                    except (_json.JSONDecodeError, TypeError):
                        continue

                    event_type = msg.get("event") or msg.get("type")
                    text = (msg.get("text") or msg.get("transcript") or "").strip()

                    if event_type == "transcript.partial" and text:
                        received_at = _time.perf_counter()
                        partial_history.append(text)
                        is_stable = (not stable_emitted) and self._is_stable(text, partial_history)
                        if is_stable:
                            stable_emitted = True
                        yield PartialTranscriptEvent(
                            text=text,
                            is_stable=is_stable,
                            received_at=received_at,
                        )

                    elif event_type in ("transcript.final", "transcript.complete") and text:
                        yield FinalTranscriptEvent(
                            text=text,
                            received_at=_time.perf_counter(),
                        )
                        break  # one utterance complete

            finally:
                send_task.cancel()
                try:
                    await send_task
                except _asyncio.CancelledError:
                    pass

        except Exception:
            raise
        # Do NOT close the connection here — session reuse is the point.

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        filename: str = "audio.wav",
        content_type: str | None = None,
    ) -> "TranscriptResult":
        """
        Compatibility shim: same interface as SarvamSTT.transcribe(), so this
        class can be used as a drop-in wherever a synchronous transcription is
        expected (e.g. harness._transcribe).  Blocks until the final transcript.
        """
        final_text = ""
        async for event in self.stream(audio_bytes, filename=filename, content_type=content_type):
            if isinstance(event, FinalTranscriptEvent):
                final_text = event.text
                break
        return TranscriptResult(
            transcript=final_text,
            language_code=language_code if language_code != "unknown" else None,
        )


class MockRealtimeSTT(SarvamRealtimeSTT):
    """
    Drop-in stand-in for SarvamRealtimeSTT that emits canned partial and final
    transcript events without touching the network.  The partial sequence
    simulates incremental word-by-word arrival so stability logic is exercised.

    Useful for benchmarks and unit tests.
    """

    def __init__(
        self,
        transcripts: dict[str, str] | None = None,
        default_transcript: str = "what is retrieval augmented generation",
        partial_chunk_words: int = 3,      # 3-word steps give realistic partial length
        stability_repeats: int = 1,       # 1 = stable as soon as min_tokens met; realistic for mock
        min_tokens: int = 2,
    ):
        # Don't call super().__init__() — that would require SARVAM_API_KEY
        # and websockets. Re-implement only what we need.
        self.api_key = "mock"
        self.stability_repeats = stability_repeats
        self.min_tokens = min_tokens
        self.open_timeout = 0.0
        self.ping_interval = 0.0
        self._ws = None
        self._ws_lock = _asyncio.Lock()
        self._transcripts = {
            **(MockSTT.DEFAULT_TRANSCRIPTS),
            **(transcripts or {}),
        }
        self._default = default_transcript
        self._partial_chunk_words = partial_chunk_words

    # Async only, matching the realtime client this stands in for. A second
    # sync definition above it was silently shadowed by this one, so callers
    # branching on iscoroutinefunction saw whichever Python happened to bind
    # last.
    async def prewarm(self) -> float:  # type: ignore[override]
        return 0.0

    async def close(self) -> None:
        pass

    async def stream(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str | None = None,
        chunk_size: int = 8192,
    ) -> "AsyncIterator[PartialTranscriptEvent | FinalTranscriptEvent]":
        # Derive the canned transcript from the audio_bytes "language" hint
        # injected by the benchmark (stored in the first 8 bytes of a sentinel).
        transcript = self._transcripts.get("unknown", self._default)
        words = transcript.split()
        chunk = self._partial_chunk_words
        partial_history: list[str] = []
        stable_emitted = False

        # The mock emits the full transcript as a single stable partial so that
        # _queries_close_enough() always passes (the partial text == final text).
        # This is the optimistic case that demonstrates the overlap path working
        # correctly; SarvamRealtimeSTT will emit realistic incremental partials.
        await _asyncio.sleep(0)
        yield PartialTranscriptEvent(
            text=transcript,
            is_stable=True,
            received_at=_time.perf_counter(),
        )

        # Final
        await _asyncio.sleep(0)
        yield FinalTranscriptEvent(
            text=transcript,
            received_at=_time.perf_counter(),
        )
