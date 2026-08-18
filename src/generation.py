"""
Generation module.

Generates answers grounded in retrieved context via a swappable LLM
provider: GroqProvider (default, fast/free-tier) or AnthropicProvider
(fallback/alternate), selected at runtime by the LLM_PROVIDER environment
variable. Generator is the public facade callers use; LLMProvider is the
interface each backend implements.
"""

import os
from abc import ABC, abstractmethod

import anthropic
import groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.chunking import Chunk

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 1024

NO_CONTEXT_RESPONSE = "I don't have enough information in the provided context to answer that."

SYSTEM_PROMPT = f"""You are a question-answering assistant for a retrieval-augmented system.

Answer the user's question using ONLY the information in the numbered context passages provided below. Do not use any outside knowledge, and do not guess or speculate beyond what the passages state.

For every factual claim in your answer, cite the passage_id(s) that support it inline, in the form [passage_id: X].

If the provided context does not contain enough information to answer the question, respond with exactly this sentence and nothing else:
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


class LLMProvider(ABC):
    """Common interface every generation backend implements."""

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

    async def answer(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        user_message = _build_user_message(query, retrieved_chunks)
        try:
            response = await self._create_completion(user_message)
        except Exception as exc:
            raise GenerationError(f"Groq generation failed after retries: {exc}") from exc
        return response.choices[0].message.content.strip()


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

    async def answer(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        user_message = _build_user_message(query, retrieved_chunks)
        try:
            response = await self._create_message(user_message)
        except Exception as exc:
            raise GenerationError(f"Anthropic generation failed after retries: {exc}") from exc
        return "".join(block.text for block in response.content if block.type == "text").strip()


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
