"""The published figures, checked against the code and the record they came from.

Four documents state measured numbers: `README.md` summarises, `docs/EVALUATION.md`
records, `demo/examples.json` carries a baked transcript of real runs, and the CI
workflow quotes the suite size. Every figure that appears twice is a place two
files can disagree, and nothing here re-runs a paid measurement — so the copies
were free to drift apart silently, which is the failure this file exists to make
loud.

Nothing is copied into this module. The numbers are read out of the documents and
re-derived from each other, because a literal expected value written here would
just be a fifth copy. The chain is:

    the per-run table  ->  the range table  ->  the README headline
    the token counts   ->  observe.PRICING  ->  the baked demo's dollars
    the JSONL fixture  ->  the published slice sizes

If one of these fails, the document is wrong or the run was never re-measured;
re-measuring is a paid operation and the reproduce command sits beside each
figure in `docs/EVALUATION.md`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from triage import agent, classify, demo, draft, eval as eval_mod, observe, schema

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
EVALUATION = (ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")


def _row(text: str, label: str) -> list[str]:
    """The cells of the markdown table row whose first column contains `label`."""
    for line in text.splitlines():
        if line.startswith("|") and label in line:
            return [c.strip().strip("*").strip() for c in line.strip("|").split("|")]
    pytest.fail(f"no table row for {label!r} — was it renamed?")


def _nums(cell: str) -> list[float]:
    """Every figure in a cell, with $ % and thousands separators removed."""
    return [float(m.replace("$", "").replace(",", "").replace("%", ""))
            for m in re.findall(r"\$?\d+(?:\.\d+)?%?", cell)]


def _range(cell: str) -> tuple[float, float]:
    """A published `a – b` range (en dash), or a single figure standing for both."""
    lo, *hi = _nums(cell)
    return lo, (hi[0] if hi else lo)


# The two tables share several row labels, so each lookup is scoped to the region
# it belongs to rather than to the first match in the file.
PER_RUN = "| | 2026-" + EVALUATION.split("| | 2026-", 1)[1].split("\n\n", 1)[0]
RANGES = EVALUATION.split("| | range across runs |", 1)[1].split("\n\n", 1)[0]

RUN_COLUMNS = slice(1, 5)  # the four dated result columns


def _first(cell: str) -> float:
    return _nums(cell)[0]


def _p95(cell: str) -> float:
    """`p50 6.81s / p95 13.86s` -> 13.86. Reading positionally would pick up the
    50 and 95 in the percentile names themselves."""
    match = re.search(r"p95\s+([\d.]+)s", cell)
    assert match, f"no p95 figure in {cell!r}"
    return float(match.group(1))


@pytest.mark.parametrize("per_run,summary,read", [
    ("false abstentions", "false abstentions (of 17)", _first),
    ("action correctness", "action correctness (key-match)", _first),
    ("cost per incident", "cost per incident", _first),
    ("end-to-end latency", "end-to-end p95", _p95),
])
def test_the_published_range_is_the_range_of_the_runs_above_it(per_run, summary, read):
    """EVALUATION.md prints each run, then a range over them. The range is a claim
    about the runs, so it has to follow from them rather than be typed alongside."""
    cells = [c for c in _row(PER_RUN, per_run)[RUN_COLUMNS] if _nums(c)]
    measured = [read(c) for c in cells]
    assert len(measured) >= 2, f"{per_run!r} has too few runs to summarise"

    published = _range(_row(RANGES, summary)[1])
    assert published == (min(measured), max(measured)), (
        f"the summary row for {summary!r} says {published}, but the runs above it "
        f"give {(min(measured), max(measured))}"
    )


def test_the_severity_spread_is_the_arithmetic_it_claims():
    """The document calls severity the least stable number and quotes the gap.
    That gap is a subtraction of two figures on the same page."""
    lo, hi = _range(_row(RANGES, "severity accuracy (in-scope)")[1])
    stated = re.search(r"a (\d+\.\d+)-point spread", EVALUATION)
    assert stated, "EVALUATION.md no longer states the severity spread"
    assert round(hi - lo, 1) == float(stated.group(1))


def test_the_invariant_metrics_really_are_invariant_across_every_run():
    """The two headline claims are the only ones asserted for *every* run, so
    they are the two that must never be quoted from a single column."""
    for label in ("abstention rate (15 must-abstain)", "missed escalations"):
        cells = _row(PER_RUN, label)[RUN_COLUMNS]
        values = [_nums(c)[0] for c in cells]
        assert len(set(values)) == 1, (
            f"{label!r} varies across runs ({values}) — the README calls it "
            f"unchanged in every run, which would no longer be true")


# --- the README quotes the record ------------------------------------------------------

def _headline_table() -> str:
    return README.split("| | measured (4 runs) | what it means |", 1)[1].split("\n\n", 1)[0]


def test_the_readme_headline_figures_come_from_the_evaluation_record():
    """The README rounds — 15.18s is published as 15.2s — so this checks that each
    headline figure *derives* from the record rather than appearing in it
    verbatim. A README that rounds is fine; one that rounds a number nobody
    measured is not.
    """
    record = set()
    for value in re.findall(r"\$?\d+(?:\.\d+)?%?", EVALUATION):
        raw = float(value.replace("$", "").replace("%", ""))
        record |= {raw, round(raw, 1), round(raw)}

    figures = [f for f in re.findall(r"\*\*([^*]+)\*\*", _headline_table())
               if any(c.isdigit() for c in f)]
    assert figures, "the README headline table no longer contains any figures"

    for figure in figures:
        for number in _nums(figure):
            assert number in record, (
                f"the README headlines {number} (in {figure!r}), which does not "
                f"appear in docs/EVALUATION.md at any published precision. The "
                f"record is the source; the README quotes it."
            )


def test_the_readme_counts_the_runs_the_record_actually_holds():
    """"over four full-pipeline runs" is itself a published number."""
    stated = re.search(r"over \*\*(\w+)\*\*\s*\n?\s*full-pipeline runs", README)
    assert stated, "the README no longer says how many runs it summarises"
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    header = _row(PER_RUN, "2026-07-02")
    assert words[stated.group(1)] == len([c for c in header[RUN_COLUMNS] if c])


# --- the baked demo prices itself from the current rate table ----------------------------

def test_the_baked_demo_dollars_reprice_from_the_rate_table():
    """`demo/examples.json` stores both token counts and dollars. The replay
    re-derives dollars from the tokens, but the stored figures are what
    `python -m triage demo` prints in its header — so a change to `observe.PRICING`
    would leave the demo quoting a price the same command no longer computes.
    """
    payload = demo.load_examples()
    for example in payload["examples"]:
        trace = example["trace"]
        ledger = {model: {k: v for k, v in usage.items() if k != "usd"}
                  for model, usage in trace["cost_by_model"].items()}
        derived = observe.cost_usd(ledger)

        for model, usage in trace["cost_by_model"].items():
            assert round(derived.get(model, 0.0), 6) == usage["usd"], (
                f"{example['incident']['id']}: the baked ${usage['usd']} for {model} "
                f"is not what its token counts cost today — re-bake the demo")
        assert round(derived["total"], 6) == trace["total_usd"]

    total = round(sum(e["trace"]["total_usd"] for e in payload["examples"]), 4)
    assert payload["bake_cost_usd"] == total, (
        "the demo header quotes a bake cost that is not the sum of the runs it replays")


def test_every_model_the_demo_names_is_one_the_pipeline_actually_calls():
    payload = demo.load_examples()
    assert payload["models"]["classify"] == classify.MODEL
    assert payload["models"]["draft"] == draft.MODEL


# --- the fixture is the size the documents say it is --------------------------------------

def test_the_published_slice_sizes_match_the_incident_set():
    """Every metric is a fraction of one of these slices, so a fixture that grew
    without the record following would silently restate what 100% means."""
    incidents = schema.load_incidents(ROOT / "fixtures" / "incidents" / "incidents.jsonl")
    in_scope = [i for i in incidents if i.in_scope]

    stated_total = re.search(r"\*\*(\d+) synthetic incidents\*\*", EVALUATION)
    assert stated_total and int(stated_total.group(1)) == len(incidents)

    assert _nums(_row(EVALUATION, "| in-scope |")[1])[0] == len(in_scope)
    assert _nums(_row(EVALUATION, "out-of-scope |")[1])[0] == len(incidents) - len(in_scope)
    assert _nums(_row(EVALUATION, "of which **must abstain**")[1])[0] == len(
        [i for i in in_scope if i.must_abstain])
    assert _nums(_row(EVALUATION, "(a PROPOSE is correct)")[1])[0] == len(
        [i for i in incidents if not i.must_abstain])
    assert _nums(_row(EVALUATION, "(a PROPOSE is a miss)")[1])[0] == len(
        [i for i in incidents if i.must_abstain])


def test_the_red_team_fixture_stays_out_of_the_measured_set():
    """The record says adding it never moves the numbers. That only holds while it
    is in its own file."""
    measured = schema.load_incidents(ROOT / "fixtures" / "incidents" / "incidents.jsonl")
    redteam = schema.load_incidents(ROOT / "fixtures" / "incidents" / "redteam.jsonl")
    assert redteam and not {i.id for i in redteam} & {i.id for i in measured}


# --- the documents name the models the code calls -------------------------------------------

@pytest.mark.parametrize("model", [
    classify.MODEL, draft.MODEL, eval_mod.JUDGE_MODEL, agent.ORCHESTRATOR_MODEL,
])
def test_every_model_the_code_calls_is_priced_and_documented(model):
    """An unpriced model contributes $0 to every cost figure this repository
    publishes, and it does so without raising."""
    assert model in observe.PRICING, f"{model} is called but not priced"
    assert model in README or model in EVALUATION
