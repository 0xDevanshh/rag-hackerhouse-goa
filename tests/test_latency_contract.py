"""Tests for the non-overlapping request timing contract."""

from src.latency import RequestTrace


def test_trace_residual_is_zero_for_sequential_stages():
    trace = RequestTrace()
    trace.start()
    with trace.span("bm25"):
        pass
    with trace.span("fusion"):
        pass
    trace.finish()
    assert trace.total_ms >= trace.measured_ms
    assert abs(trace.unaccounted_ms) < 1.0
