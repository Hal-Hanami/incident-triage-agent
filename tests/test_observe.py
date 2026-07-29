"""Offline tests for the observability + cost-ceiling seed (design §7, §8).

Per-model cost, latency percentiles, the request Trace, and the per-incident
budget trip — all pure / no key / no network.
"""

from __future__ import annotations

import pytest

from triage import observe


def test_cost_usd_splits_by_model_and_totals():
    usage = {
        "claude-opus-4-8": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        "claude-haiku-4-5": {"input_tokens": 1_000_000, "output_tokens": 0},
        "voyage-4-lite": {"total_tokens": 1_000_000},
    }
    cost = observe.cost_usd(usage)
    assert cost["claude-opus-4-8"] == 5.0 + 25.0   # $5/1M in + $25/1M out
    assert cost["claude-haiku-4-5"] == 1.0         # $1/1M in, no output
    assert cost["voyage-4-lite"] == 0.02           # $0.02/1M total
    assert cost["total"] == 30.0 + 1.0 + 0.02


def test_cost_usd_prices_cache_tokens_at_read_and_write_rates():
    # §7: prompt-cache reads bill ~0.1x input, 5-minute writes 1.25x — the
    # orchestrator's spend is mostly cache reads, so the cross-check needs these.
    usage = {"claude-opus-4-8": {"cache_read_input_tokens": 1_000_000,
                                 "cache_creation_input_tokens": 1_000_000}}
    cost = observe.cost_usd(usage)
    assert cost["claude-opus-4-8"] == 0.50 + 6.25
    haiku = observe.cost_usd({"claude-haiku-4-5": {"cache_read_input_tokens": 1_000_000}})
    assert haiku["claude-haiku-4-5"] == 0.10


def test_cost_usd_skips_unpriced_models():
    # an unknown/fake model contributes 0 and is absent — token counts stay
    # authoritative even when a price isn't known (fakes in tests don't break costing).
    cost = observe.cost_usd({"fake-classifier": {"total_tokens": 9_999}})
    assert "fake-classifier" not in cost
    assert cost["total"] == 0.0


def test_percentile_interpolates_and_handles_edges():
    assert observe.percentile([], 50) is None
    assert observe.percentile([4.2], 95) == 4.2
    data = [1, 2, 3, 4]
    assert observe.percentile(data, 0) == 1
    assert observe.percentile(data, 100) == 4
    assert observe.percentile(data, 50) == 2.5


def test_trace_records_spans_and_usage():
    trace = observe.Trace()
    with trace.span("classify"):
        pass
    with trace.span("retrieve"):
        pass
    trace.add_usage("claude-opus-4-8", {"input_tokens": 10, "output_tokens": 2})
    trace.add_usage("claude-opus-4-8", {"input_tokens": 5})
    trace.add_usage("claude-opus-4-8", {})  # empty usage is a no-op
    assert [name for name, _ in trace.spans] == ["classify", "retrieve"]
    assert trace.total_seconds >= 0
    assert trace.usage_by_model["claude-opus-4-8"] == {"input_tokens": 15, "output_tokens": 2}


def test_budget_trips_when_running_cost_exceeds_cap():
    trace = observe.Trace()
    trace.add_usage("claude-opus-4-8", {"input_tokens": 1_000, "output_tokens": 1_000})
    # ~ $0.005 + $0.025 = $0.03 < default $0.05 budget -> no trip
    trace.check_budget()
    # push past the cap
    trace.add_usage("claude-opus-4-8", {"output_tokens": 2_000})  # +$0.05
    with pytest.raises(observe.BudgetExceeded):
        trace.check_budget()


def test_span_helper_is_noop_without_trace():
    with observe.span(None, "x"):
        pass  # must not raise


def test_format_trace_renders_the_budget_verdict_when_given_a_cap():
    trace = observe.Trace()
    trace.add_usage("claude-haiku-4-5", {"input_tokens": 1_000})  # $0.001
    within = "\n".join(observe.format_trace(trace, budget=0.05))
    assert "within" in within and "$0.0500" in within
    exceeded = "\n".join(observe.format_trace(trace, budget=0.0005))
    assert "EXCEEDED" in exceeded
    assert "budget" not in "\n".join(observe.format_trace(trace))  # opt-in only
