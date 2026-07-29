"""Cross-cutting: observability + the cost ceiling.

The same "measure and operate" layer as tech-docs-rag's `rag/observe.py`, reshaped for
the agent: make one triage explainable after the fact (which stages ran, how long
each took, how many tokens per model), put a dollar figure on it, and enforce a
hard per-incident budget that trips an ABSTAIN+escalate rather than overspending
(design §7, §8).

Cost is split *by model* on purpose — that split is both the story (cheap
`claude-haiku-4-5` classifier vs `claude-opus-4-8` drafter vs near-free Voyage
retrieval) and the optimization lever, exactly as in tech-docs-rag. Token counts are
authoritative (returned by each API); USD is an estimate from the rate table.
Everything here is pure / offline-testable — no key, no network.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Iterator, Sequence

# USD per 1M tokens. NOT memorized — confirmed from the source each release:
#   Claude:  `claude-api` skill pricing table (re-checked 2026-07-04).
#   Voyage:  https://docs.voyageai.com/docs/pricing (inherited from tech-docs-rag).
# Re-check on model/price changes. Claude bills input and output separately;
# prompt-cache reads bill at ~0.1x the input rate and 5-minute-TTL cache writes
# at 1.25x — modeled because the agent shell's orchestrator reuses
# a cached prefix every turn, so the §7 cross-check would otherwise undercount.
# Voyage bills one "total" token stream per call.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00,    # drafter + orchestrator (design §2)
                        "cache_read": 0.50, "cache_write": 6.25},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00,    # classifier / judge (design §2)
                         "cache_read": 0.10, "cache_write": 1.25},
    "voyage-4-lite": {"total": 0.02},                      # runbook embeddings (tech-docs-rag)
    "rerank-2.5-lite": {"total": 0.02},                    # runbook rerank (tech-docs-rag)
}

# Maps a PRICING rate key to the usage dict key the APIs report.
_USAGE_KEY = {"input": "input_tokens", "output": "output_tokens",
              "cache_read": "cache_read_input_tokens",
              "cache_write": "cache_creation_input_tokens",
              "total": "total_tokens"}

# Default hard budget per incident (USD). Crossing it trips an ABSTAIN+escalate
# (design §8) rather than letting one incident run away. Tunable per deployment.
DEFAULT_INCIDENT_BUDGET_USD = 0.05


class BudgetExceeded(Exception):
    """Raised when a triage's running cost crosses its per-incident budget."""

    def __init__(self, spent: float, budget: float):
        self.spent = spent
        self.budget = budget
        super().__init__(f"cost ${spent:.4f} exceeded budget ${budget:.4f}")


def merge_usage(into: dict[str, dict[str, int]], model: str, usage: dict[str, int]) -> None:
    """Accumulate one model's `{*_tokens: n}` usage into a by-model ledger (in place)."""
    bucket = into.setdefault(model, {})
    for key, val in usage.items():
        bucket[key] = bucket.get(key, 0) + val


def cost_usd(usage_by_model: dict[str, dict[str, int]]) -> dict[str, float]:
    """Per-model USD from a by-model token ledger, plus a `"total"` key.

    Models absent from PRICING contribute 0 and are skipped (e.g. a fake model in
    tests) — token counts stay authoritative even when a price isn't known.
    """
    breakdown: dict[str, float] = {}
    total = 0.0
    for model, usage in usage_by_model.items():
        rates = PRICING.get(model)
        if rates is None:
            continue
        c = sum(usage.get(_USAGE_KEY[kind], 0) / 1_000_000 * rate
                for kind, rate in rates.items())
        if c:
            breakdown[model] = c
            total += c
    breakdown["total"] = total
    return breakdown


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated p-th percentile (p in [0,100]); None for empty input."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


@dataclass
class Trace:
    """One triage's observability record: ordered stage spans + per-model usage.

    `span(name)` times a stage (classify / retrieve / draft / decide);
    `add_usage(model, usage)` files token counts under the model that produced
    them. `cost()` is the running USD — `check_budget()` turns it into the cost
    ceiling. Stages are sequential, so `total_seconds` is the traced wall-clock.
    """

    spans: list[tuple[str, float]] = field(default_factory=list)
    usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append((name, time.perf_counter() - t0))

    def add_usage(self, model: str, usage: dict[str, int]) -> None:
        if usage:
            merge_usage(self.usage_by_model, model, usage)

    def cost(self) -> float:
        return cost_usd(self.usage_by_model)["total"]

    def check_budget(self, budget: float = DEFAULT_INCIDENT_BUDGET_USD) -> None:
        """Raise BudgetExceeded if the running cost has crossed `budget` (design §8)."""
        spent = self.cost()
        if spent > budget:
            raise BudgetExceeded(spent, budget)

    @property
    def total_seconds(self) -> float:
        return sum(s for _, s in self.spans)


def span(trace: Trace | None, name: str):
    """`with span(trace, name):` — times the block if a Trace is given, else no-op."""
    return trace.span(name) if trace is not None else nullcontext()


def format_cost_block(usage_by_model: dict[str, dict[str, int]], *, indent: str = "  ",
                      latencies: Sequence[float] | None = None,
                      n: int | None = None) -> list[str]:
    """Render the per-model token/cost table (+ optional latency percentiles)."""
    lines: list[str] = []
    costs = cost_usd(usage_by_model)
    for model in sorted(usage_by_model):
        u = usage_by_model[model]
        toks = ", ".join(f"{k.replace('_tokens', '')}={v}" for k, v in u.items())
        usd = costs.get(model)
        money = f"  ${usd:.4f}" if usd is not None else ""
        lines.append(f"{indent}{model:<18} {toks}{money}")
    total = costs["total"]
    if n:
        lines.append(f"{indent}{'TOTAL':<18} ${total:.4f}  (${total / n:.4f}/incident over {n})")
    else:
        lines.append(f"{indent}{'TOTAL':<18} ${total:.4f}")
    if latencies:
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        lines.append(f"{indent}{'latency':<18} p50={p50:.2f}s  p95={p95:.2f}s  "
                     f"(end-to-end per incident, n={len(latencies)})")
    return lines


def format_trace(trace: Trace, *, budget: float | None = None) -> list[str]:
    """Render one triage's spans + cost — the per-incident observability footer.

    With `budget`, also renders the §8 verdict: spend against the per-incident
    cap (the trace's own cost only — the agent CLI prints a combined line that
    additionally folds in the orchestrator's spend)."""
    lines = ["--- trace ---"]
    for name, secs in trace.spans:
        lines.append(f"  {name:<10} {secs * 1000:7.1f} ms")
    lines.append(f"  {'total':<10} {trace.total_seconds * 1000:7.1f} ms")
    if trace.usage_by_model:
        lines.append("  cost:")
        lines.extend(format_cost_block(trace.usage_by_model, indent="    "))
    if budget is not None:
        spent = trace.cost()
        state = "EXCEEDED" if spent > budget else "within"
        lines.append(f"  {'budget':<10} ${spent:.4f} spent of ${budget:.4f} cap — {state} (design §8)")
    return lines
