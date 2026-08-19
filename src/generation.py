"""
Generation module.

Generates answers grounded in retrieved context via a swappable LLM
provider: GroqProvider (default, fast/free-tier) or AnthropicProvider
(fallback/alternate), selected at runtime by the LLM_PROVIDER environment
variable. Generator is the public facade callers use; LLMProvider is the
interface each backend implements.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import anthropic
import groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.chunking import Chunk

logger = logging.getLogger(__name__)

# Groq's llama-3.1-8b-instant — the previous default — has been
# decommissioned and now returns 404 model_not_found. That failure was also a
# latency bug, not only a correctness one: NotFoundError subclasses
# groq.APIStatusError, so the retry policy below treated it as transient and
# spent ~3s of exponential backoff re-requesting a model that no longer
# exists, before every request fell through to the degraded response.
#
# openai/gpt-oss-20b is the replacement, chosen by measurement against the
# models this account can actually reach (8 streamed calls each, same RAG
# prompt):
#
#   openai/gpt-oss-20b   ttft p50 545ms  total p50 595ms  — correct
#                        "[passage_id: N]" citations, ~155 output tokens
#   qwen/qwen3.6-27b     ttft p50 112ms  but emits raw <think> reasoning into
#                        message content and blew past max_tokens (truncated,
#                        unusable); with reasoning_effort="none" it answered
#                        "[passage_id: 101]" and nothing else
#   openai/gpt-oss-120b  ttft p50 574ms  total p50 782ms — slower, no quality gain
#
# qwen's TTFT is tempting and it is the fastest model on offer, but neither of
# its modes produces a citable grounded answer, so it would buy latency by
# destroying the property this pipeline exists to provide.
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Claude Haiku 4.5 is the latency-appropriate tier for short, grounded,
# already-retrieved answers — this is the alternate provider for a pipeline
# whose whole objective is minimum time-to-answer, not a reasoning workload.
DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

# Raised from 300, because the default provider is a *reasoning* model and its
# hidden reasoning is billed against this same budget — so a cap sized for the
# visible answer can be consumed before any answer is emitted. In a 20-query
# warm run at 300, six requests produced zero content tokens; the grounding
# guardrail correctly turned those into refusals, but a refusal caused by a
# starved token budget is a wrong answer, not a safe one.
#
# This costs almost nothing in latency, which is the reason it is safe to raise:
# generation is a small slice of this pipeline's LLM time compared to
# time-to-first-token, and a model that finishes early still stops early
# (`finish_reason: stop`) rather than generating to the cap. Measured mean
# total across the same six queries: 658ms at 300, 816ms at 700, 747ms at 1200
# — differences inside run-to-run noise.
#
# `llm_finish_reason` is recorded on the trace so a truncation
# (`finish_reason: length`) is visible rather than inferred.
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "700"))

NO_CONTEXT_RESPONSE = "I don't have enough information in the provided context to answer that."

SYSTEM_PROMPT = f"""You are a question-answering assistant for a retrieval-augmented system.

Answer the user's question using ONLY the information in the numbered context passages provided below. Do not use any outside knowledge, and do not guess or speculate beyond what the passages state.

For every factual claim in your answer, cite the passage_id(s) that support it inline, in the form [passage_id: X].

Write your answer in the same language and script as the user's question. The context passages may be in a different language than the question (the corpus is indexed in both the user's language and English) — translate the facts you use into the question's language rather than quoting the passage verbatim.

If the provided context does not contain enough information to answer the question, respond with exactly this sentence and nothing else, in English, untranslated, regardless of the question's language:
"{NO_CONTEXT_RESPONSE}"
"""


class GenerationError(Exception):
    """Raised when an LLM provider fails to produce an answer after its retry budget is exhausted."""


def _format_context_block(chunk: Chunk, score: float) -> str:
    """Format one retrieved chunk as a numbered, scored context block."""
    passage_id = chunk.metadata.get("passage_id", chunk.metadata.get("doc_id", "unknown"))
    return f"[Passage {passage_id}] (score: {score:.2f}): {chunk.text}"


def _build_user_message(query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
    """Assemble the user message: numbered context blocks followed by the question."""
    context_blocks = "\n".join(_format_context_block(chunk, score) for chunk, score in retrieved_chunks)
    return f"Context:\n{context_blocks}\n\nQuestion: {query}"


def _record_llm_timing(
    timing: dict[str, Any],
    request_started: float,
    headers_at: float | None,
    first_token_at: float | None,
) -> None:
    """
    Partition one LLM call into three consecutive, non-overlapping slices, so
    a slow generation can be attributed rather than guessed at:

    - llm_network_ms: request issued -> response headers received. Transport
      and provider admission. Inflated on a cold connection pool (DNS + TCP +
      TLS) — which is what LLMProvider.prewarm exists to pay at startup — and
      inflated again by provider-side queueing under rate limiting.
    - llm_client_wait_ms: headers received -> first content token. The
      provider holding the connection open while it prefills the prompt and
      (for a reasoning model like gpt-oss) thinks before emitting anything.
      Nothing client-side can shrink this slice; it is the floor a remote LLM
      imposes.
    - llm_generation_ms: first token -> last token. Scales with output length,
      so it is the slice that responds to max_tokens and prompt tightening.

    The three sum to the whole call, which lets the caller record them as flat
    spans without corrupting the request-level residual. The two familiar
    aggregate numbers are recorded separately as *details*, precisely because
    they overlap the spans and would double-count if summed:
    llm_ttft_ms = llm_network_ms + llm_client_wait_ms, and llm_total_ms is the
    whole call.
    """
    ended = time.perf_counter()
    if headers_at is not None:
        timing["llm_network_ms"] = (headers_at - request_started) * 1000
    if first_token_at is not None:
        # Where the split point lands when headers weren't observed (an SDK
        # that only surfaces the first token): attribute the whole pre-token
        # period to the network slice rather than inventing a wait.
        split = headers_at if headers_at is not None else first_token_at
        timing["llm_client_wait_ms"] = (first_token_at - split) * 1000
        timing["llm_generation_ms"] = (ended - first_token_at) * 1000
        timing["llm_ttft_ms"] = (first_token_at - request_started) * 1000
    elif headers_at is not None:
        # Headers arrived but no content token ever did (empty or failed
        # stream): the post-header time is wait, not generation.
        timing["llm_client_wait_ms"] = (ended - headers_at) * 1000
    timing["llm_total_ms"] = (ended - request_started) * 1000


class LLMProvider(ABC):
    """Common interface every generation backend implements."""

    async def prewarm(self) -> float:
        """
        Establish the provider's HTTPS connection before any real request
        needs it, and return the cost in ms.

        Called once at process startup. A cold connection pays DNS + TCP +
        TLS on top of the request itself — measured at ~80ms for
        api.groq.com and ~100ms for api.sarvam.ai on this network — and
        without prewarming the first user request after every deploy or idle
        pool expiry pays it inside its own latency budget. Overridden per
        provider; the default is a no-op so a provider that can't cheaply
        prewarm isn't forced to.
        """
        return 0.0

    @abstractmethod
    async def answer(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        """
        Generate an answer to query, grounded strictly in retrieved_chunks.

        Args:
            query: the user's question.
            retrieved_chunks: (chunk, score) pairs to render as numbered,
                scored context blocks in the prompt.

        Returns:
            str: the generated answer text.

        Raises:
            GenerationError: if the provider fails after its retry budget.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_answer(
        self,
        query: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        timing: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream an answer to query as it's generated, grounded strictly in
        retrieved_chunks, yielding raw text deltas as they arrive.

        Note: unlike answer(), this has no retry wrapping — tenacity's retry
        doesn't compose with an async generator once partial output has
        already been yielded, so a mid-stream failure simply ends the
        stream (raising GenerationError) rather than restarting from
        scratch. answer_streamed() below adds the retry that *is* safe here
        (retry only while nothing has been yielded yet).

        Args:
            query: the user's question.
            retrieved_chunks: (chunk, score) pairs for context.
            timing: optional sink for llm_request_setup_ms / llm_network_ms /
                llm_ttft_ms / llm_generation_ms.

        Yields:
            str: successive text deltas as the model generates them.

        Raises:
            GenerationError: if the provider fails before or during streaming.
        """
        raise NotImplementedError

    async def answer_streamed(
        self,
        query: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        timing: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> str:
        """
        Produce a complete answer over the *streaming* transport, returning
        the joined text.

        This exists so the non-streaming route can report a real
        `llm_ttft_ms`. Time-to-first-token is not observable through a
        buffered completion call — the SDK returns once, at the end — which is
        why the original trace reported `llm_ttft_ms: n/a` on /query and left
        the single largest remaining stage as one opaque number. Measured
        end-to-end cost is the same either way for this workload
        (gpt-oss-20b: streamed total p50 595ms vs buffered p50 593ms), so
        nothing is traded away for the visibility.

        Retries are attempted only while no delta has been yielded yet.
        Once the model has emitted text, a restart would either duplicate or
        silently discard already-generated output, so the failure is
        propagated instead.

        Raises:
            GenerationError: if every attempt fails before first token, or any
            failure occurs after streaming has begun.
        """
        import asyncio

        call_started = time.perf_counter()
        last_exc: Exception | None = None
        attempts = 0

        def account() -> None:
            """
            Attribute everything this method spent that the last attempt's own
            timings don't cover — failed attempts plus backoff sleeps — to
            llm_retry_wait_ms.

            Without this, a rate-limited request's retry budget lands in the
            request-level unaccounted_ms: a 20-request benchmark run against
            Groq's free tier produced unaccounted_ms of 525ms at p50 and 3.2s
            at p95, all of it backoff. That is exactly the class of hidden
            latency this instrumentation exists to expose, so it gets a name
            rather than a residual.
            """
            if timing is None:
                return
            timing["llm_attempts"] = attempts
            overhead = (time.perf_counter() - call_started) * 1000 - timing.get("llm_total_ms", 0.0)
            if overhead > 0:
                timing["llm_retry_wait_ms"] = overhead

        for attempt in range(max_attempts):
            attempts = attempt + 1
            deltas: list[str] = []
            try:
                async for delta in self.stream_answer(query, retrieved_chunks, timing=timing):
                    deltas.append(delta)
                account()
                return "".join(deltas).strip()
            except Exception as exc:
                if deltas:
                    account()
                    raise GenerationError(
                        f"generation failed after {len(deltas)} deltas had already streamed: {exc}"
                    ) from exc
                last_exc = exc
                if attempt < max_attempts - 1:
                    backoff = min(2**attempt, 10)
                    logger.warning(
                        "generation attempt %d/%d failed before first token (%s); retrying in %ss",
                        attempt + 1,
                        max_attempts,
                        type(exc).__name__,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
        account()
        raise GenerationError(f"generation failed after {max_attempts} attempts: {last_exc}") from last_exc


class GroqProvider(LLMProvider):
    """Generates answers via the Groq API (OpenAI-compatible chat completions)."""

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        """
        Args:
            model: Groq model ID to use for generation.
            max_tokens: maximum tokens to generate in the response.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.model = model
        self.max_tokens = max_tokens
        # max_retries=0: tenacity below owns the retry policy exclusively,
        # so retries aren't silently multiplied by two independent layers.
        self.client = groq.AsyncGroq(api_key=api_key, max_retries=0)

    @retry(
        retry=retry_if_exception_type((groq.RateLimitError, groq.APIConnectionError, groq.APIStatusError)),
        stop=stop_after_attempt(3),  # initial attempt + up to 2 retries
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_completion(self, user_message: str):
        return await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

    async def prewarm(self) -> float:
        """Open the pooled HTTPS connection to api.groq.com ahead of time."""
        started = time.perf_counter()
        try:
            await self.client.models.list()
        except Exception as exc:  # a cold-start warmup must never block startup
            logger.warning("Groq prewarm failed (%s); first request will pay TLS setup", exc)
        return (time.perf_counter() - started) * 1000

    async def answer(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        user_message = _build_user_message(query, retrieved_chunks)
        try:
            response = await self._create_completion(user_message)
        except Exception as exc:
            raise GenerationError(f"Groq generation failed after retries: {exc}") from exc
        return response.choices[0].message.content.strip()

    async def stream_answer(
        self,
        query: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        timing: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        user_message = _build_user_message(query, retrieved_chunks)
        request_started = time.perf_counter()
        headers_at: float | None = None
        first_token_at: float | None = None
        finish_reason: str | None = None
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
            )
            # The create() await returns once response headers are in, so this
            # slice is transport + provider admission (DNS/TCP/TLS on a cold
            # pool, queueing on a warm one) and excludes generation itself.
            headers_at = time.perf_counter()
            async for chunk in stream:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta.content
                if delta:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield delta
        except Exception as exc:
            raise GenerationError(f"Groq streaming generation failed: {exc}") from exc
        finally:
            if timing is not None:
                _record_llm_timing(timing, request_started, headers_at, first_token_at)
                if finish_reason is not None:
                    timing["llm_finish_reason"] = finish_reason


class AnthropicProvider(LLMProvider):
    """Generates answers via the Anthropic API, as a fallback/alternate to GroqProvider."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        """
        Args:
            model: Anthropic model ID to use for generation.
            max_tokens: maximum tokens to generate in the response.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.model = model
        self.max_tokens = max_tokens
        # max_retries=0: tenacity below owns the retry policy exclusively.
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)

    @retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError)
        ),
        stop=stop_after_attempt(3),  # initial attempt + up to 2 retries
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_message(self, user_message: str):
        return await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

    async def prewarm(self) -> float:
        """Open the pooled HTTPS connection to the Anthropic API ahead of time."""
        started = time.perf_counter()
        try:
            await self.client.models.list()
        except Exception as exc:  # a cold-start warmup must never block startup
            logger.warning("Anthropic prewarm failed (%s); first request will pay TLS setup", exc)
        return (time.perf_counter() - started) * 1000

    async def answer(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        user_message = _build_user_message(query, retrieved_chunks)
        try:
            response = await self._create_message(user_message)
        except Exception as exc:
            raise GenerationError(f"Anthropic generation failed after retries: {exc}") from exc
        return "".join(block.text for block in response.content if block.type == "text").strip()

    async def stream_answer(
        self,
        query: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        timing: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        user_message = _build_user_message(query, retrieved_chunks)
        request_started = time.perf_counter()
        headers_at: float | None = None
        first_token_at: float | None = None
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                headers_at = time.perf_counter()
                async for text in stream.text_stream:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield text
        except Exception as exc:
            raise GenerationError(f"Anthropic streaming generation failed: {exc}") from exc
        finally:
            if timing is not None:
                _record_llm_timing(timing, request_started, headers_at, first_token_at)


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
}


def get_provider() -> LLMProvider:
    """
    Build the LLMProvider selected by the LLM_PROVIDER environment variable
    ("groq" or "anthropic"), defaulting to "groq".
    """
    provider_name = os.environ.get("LLM_PROVIDER", "groq").lower()
    try:
        provider_cls = _PROVIDERS[provider_name]
    except KeyError:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider_name!r}. Expected one of {sorted(_PROVIDERS)}."
        ) from None
    return provider_cls()


class Generator:
    """
    Public generation facade: answers a query grounded in retrieved chunks
    by delegating to a swappable LLMProvider (GroqProvider by default,
    AnthropicProvider as a fallback/alternate).
    """

    def __init__(self, provider: LLMProvider | None = None):
        """
        Args:
            provider: an explicit LLMProvider to use. If omitted, one is
                built via get_provider() from the LLM_PROVIDER env var.
        """
        self.provider = provider or get_provider()

    async def prewarm(self) -> float:
        """Establish the provider's HTTPS connection at startup — see LLMProvider.prewarm."""
        return await self.provider.prewarm()

    async def answer_streamed(
        self,
        query: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        timing: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a complete answer over the streaming transport, so
        `llm_ttft_ms` is measurable on the non-streaming route too. See
        LLMProvider.answer_streamed.
        """
        return await self.provider.answer_streamed(query, retrieved_chunks, timing=timing)

    async def answer(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        """
        Generate an answer to query, grounded strictly in retrieved_chunks.

        Args:
            query: the user's question.
            retrieved_chunks: (chunk, score) pairs, e.g.
                zip(retrieval_result.chunks, retrieval_result.scores).

        Returns:
            str: the generated answer text.

        Raises:
            GenerationError: if the underlying provider fails after its
            retry budget is exhausted, so callers (e.g. the harness) can
            catch it and fall back to a degraded response path.
        """
        return await self.provider.answer(query, retrieved_chunks)

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        timing: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream an answer to query, grounded strictly in retrieved_chunks.

        Args:
            query: the user's question.
            retrieved_chunks: (chunk, score) pairs, e.g.
                zip(retrieval_result.chunks, retrieval_result.scores).
            timing: optional sink for llm_network_ms / llm_ttft_ms /
                llm_generation_ms / llm_total_ms.

        Yields:
            str: successive text deltas as the model generates them.

        Raises:
            GenerationError: if the underlying provider fails before or
            during streaming (no retry — see LLMProvider.stream_answer).
        """
        return self.provider.stream_answer(query, retrieved_chunks, timing=timing)
