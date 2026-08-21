"""
Request-lifecycle latency tracing.

The problem this module exists to solve: the previous latency breakdown
computed `total_ms` as the *sum of the stages it knew about*, which makes it
structurally impossible for the breakdown to ever reveal missing time. Any
work that happened outside a named stage — network-bound STT, guardrail
embedding, ASGI middleware, body parsing, response serialization — simply
vanished, and the reported total silently shrank to match whatever was
instrumented.

RequestTrace inverts that: `total_ms` is measured independently, as wall
clock across the whole request, and `unaccounted_ms` is the *residual*
`total_ms - sum(spans)`. A large residual is then a signal to go add a span,
rather than something the arithmetic hides. `unaccounted_ms` near zero is the
invariant that says the trace is honest.

Spans are flat and non-overlapping by convention: `span()` calls must not
nest, or the residual double-counts and can go negative. For genuinely nested
work, record the inner piece as a `detail` (which is reported but excluded
from the residual arithmetic) rather than as a second span.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from pydantic import BaseModel, Field, computed_field

# Ordering used when rendering a trace, so every report reads in request order
# rather than in whatever order stages happened to complete. Names not listed
# here still render, after these, in insertion order.
SPAN_ORDER: tuple[str, ...] = (
    "middleware",
    "body_parse",
    "auth",
    "db",
    # Serial STT (batch upload path)
    "stt_network",
    # Overlapped STT path: partial arrives, retrieval fires concurrently,
    # final transcript confirms.  stt_overlap_savings records the wall-clock
    # that retrieval "got for free" by running during STT rather than after.
    # retrieval_on_partial is a detail (not a span) because it overlaps the
    # flat retrieval sub-spans and must not be summed into total_ms.
    "stt_to_first_partial",
    "stt_final",
    "stt_overlap_savings",
    "cache_lookup",
    "query_preprocessing",
    "embedding_cache",
    "embedding_compute",
    "vector_search",
    "bm25",
    "fusion",
    "reranking",
    "retrieval_overhead",
    "relevance_guard",
    "context_build",
    "llm_network",
    "llm_client_wait",
    "llm_generation",
    "llm_retry_wait",
    "grounding_guard",
    "serialization",
    "response_write",
)


class Span(BaseModel):
    """One measured, non-overlapping slice of a request's wall clock."""

    name: str
    started_at: float
    ended_at: float
    duration_ms: float


class RequestTrace(BaseModel):
    """
    Wall-clock trace for one request.

    `total_ms` comes from start()/finish() around the entire request (ideally
    from ASGI middleware, so it includes framework overhead), never from
    summing spans. `unaccounted_ms` is the residual — the whole point of the
    type.
    """

    spans: list[Span] = Field(default_factory=list)
    # Sub-measurements that live *inside* a span (e.g. llm_ttft_ms inside
    # llm_generation). Reported for diagnosis, deliberately excluded from the
    # residual so they can't double-count.
    details: dict[str, float] = Field(default_factory=dict)
    # Non-timing facts worth carrying alongside the timings: cache hit/miss,
    # retry counts, which path served the request.
    labels: dict[str, str | int | bool] = Field(default_factory=dict)
    # Named instants on the request timeline, for slices whose start and end
    # live in different callbacks (see mark()). Excluded from serialization:
    # these are raw perf_counter values with an arbitrary origin, meaningful
    # only for subtracting from each other inside this process, so shipping
    # them to an API consumer would be noise it could only misread.
    marks: dict[str, float] = Field(default_factory=dict, exclude=True)

    started_at: float | None = None
    ended_at: float | None = None

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Mark the beginning of the request's wall clock."""
        self.started_at = time.perf_counter()

    def finish(self) -> None:
        """Mark the end of the request's wall clock."""
        self.ended_at = time.perf_counter()

    # total_ms and unaccounted_ms are computed_field, not plain property, so
    # they are *serialized* with the model.
    #
    # That matters as much for consumers as for this process. A client handed
    # only `spans` has no way to recover the real total except by summing them
    # — which is precisely the mistake this class exists to prevent, since that
    # sum silently excludes whatever wasn't instrumented. Shipping the measured
    # total means no consumer has to re-derive it, and none can accidentally
    # re-derive it wrongly.
    @computed_field
    @property
    def total_ms(self) -> float:
        """
        Wall clock for the whole request, in ms.

        Falls back to the span sum only if start()/finish() were never called
        (e.g. a trace built by a unit test), which is the one case where no
        independent measurement exists.
        """
        if self.started_at is None:
            return self.measured_ms
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return (end - self.started_at) * 1000

    @property
    def measured_ms(self) -> float:
        """Sum of all recorded spans, in ms."""
        return sum(span.duration_ms for span in self.spans)

    @computed_field
    @property
    def unaccounted_ms(self) -> float:
        """
        Wall clock not attributed to any span.

        Can go slightly negative (sub-millisecond) from timer granularity, or
        substantially negative if spans overlap — treat the latter as a bug in
        the instrumentation, not as noise.
        """
        return self.total_ms - self.measured_ms

    # ---- recording -------------------------------------------------------

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """
        Time a block and record it as a flat span. Must not be nested inside
        another span() for the same trace — see the module docstring.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - started) * 1000, started_at=started)

    def add(self, name: str, duration_ms: float, started_at: float | None = None) -> None:
        """
        Record a span whose duration was measured elsewhere (e.g. handed up
        from a library that timed itself). Repeated names accumulate into one
        span rather than appearing twice, so a stage that runs more than once
        per request (retrieval's query-rewrite retry) reports one total.
        """
        for span in self.spans:
            if span.name == name:
                span.duration_ms += duration_ms
                span.ended_at = time.perf_counter()
                return
        now = time.perf_counter()
        start = started_at if started_at is not None else now - duration_ms / 1000
        self.spans.append(
            Span(name=name, started_at=start, ended_at=start + duration_ms / 1000, duration_ms=duration_ms)
        )

    def detail(self, name: str, value: float) -> None:
        """Record a sub-measurement inside a span (excluded from the residual)."""
        self.details[name] = value

    def mark(self, name: str) -> float:
        """
        Stamp a named instant on the request timeline and return it.

        Used by the ASGI middleware, which cannot wrap the framework's own
        work in a `span()`: the boundaries it needs (route handler entered,
        route handler returned, response headers sent, body flushed) are
        separate callbacks, not a block. Recording instants and subtracting
        them afterwards is how those slices get measured — see
        span_between().
        """
        self.marks[name] = time.perf_counter()
        return self.marks[name]

    def span_between(self, name: str, start_mark: str, end_mark: str) -> None:
        """
        Record a span from the interval between two marks, if both were
        stamped and the interval is positive.
        """
        start, end = self.marks.get(start_mark), self.marks.get(end_mark)
        if start is None or end is None or end <= start:
            return
        self.add(name, (end - start) * 1000, started_at=start)

    def label(self, name: str, value: str | int | bool) -> None:
        """Record a non-timing fact about this request."""
        self.labels[name] = value

    def get(self, name: str) -> float | None:
        """Duration of a named span if it ran, else None (distinct from 0.0)."""
        for span in self.spans:
            if span.name == name:
                return span.duration_ms
        return None

    # ---- rendering -------------------------------------------------------

    def ordered_spans(self) -> list[Span]:
        """Spans in canonical request order (see SPAN_ORDER)."""
        rank = {name: i for i, name in enumerate(SPAN_ORDER)}
        return sorted(self.spans, key=lambda s: (rank.get(s.name, len(SPAN_ORDER)), s.started_at))

    def as_dict(self) -> dict[str, float]:
        """
        Flat `{name_ms: duration}` mapping, ending with unaccounted_ms and
        total_ms so the arithmetic is checkable by eye:
        sum(stages) + unaccounted_ms == total_ms.
        """
        out = {f"{span.name}_ms": span.duration_ms for span in self.ordered_spans()}
        out.update({f"{k}_ms": v for k, v in self.details.items()})
        out["unaccounted_ms"] = self.unaccounted_ms
        out["total_ms"] = self.total_ms
        return out

    def render(self) -> str:
        """Human-readable multi-line breakdown, for logging one record per request."""
        lines = [f"{span.name}_ms: {span.duration_ms:.1f}" for span in self.ordered_spans()]
        lines += [f"  ({k}_ms: {v:.1f})" for k, v in sorted(self.details.items())]
        lines.append(f"unaccounted_ms: {self.unaccounted_ms:.1f}")
        lines.append(f"total_ms: {self.total_ms:.1f}")
        if self.labels:
            lines.append("labels: " + " ".join(f"{k}={v}" for k, v in sorted(self.labels.items())))
        return "\n".join(lines)


def percentiles(values: list[float], quantiles: tuple[int, ...] = (50, 95, 99)) -> dict[str, float] | None:
    """
    Linear-interpolated percentiles over `values`, plus n/min/max/mean.

    Returns None for an empty input so callers can distinguish "stage never
    ran" from "stage ran and took 0ms" — the same distinction RequestTrace.get
    preserves.
    """
    if not values:
        return None
    ordered = sorted(values)

    def at(q: int) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q / 100
        low, high = int(pos), min(int(pos) + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

    out = {f"p{q}": at(q) for q in quantiles}
    out.update(
        n=len(ordered),
        min=ordered[0],
        max=ordered[-1],
        mean=sum(ordered) / len(ordered),
    )
    return out
