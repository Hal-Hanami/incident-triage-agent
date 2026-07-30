"""The public page, driven through Streamlit's own test harness — design §11.

The page is the one surface most readers will ever see, and it is the surface
furthest from the suite: it renders numbers nobody re-derives by hand and states
an outcome nobody re-checks. `tests/test_published_numbers.py` pins the prose it
quotes, but prose is not the risk here — a page that raises on the third
transcript, or labels an abstention as a proposal, would be wrong in a way no
text check can see.

So this runs the real script for every baked incident and asserts what it put on
screen against what the transcript actually recorded. No key, no network; the
transcripts are the same committed ones the CLI replays.

Skipped when Streamlit is absent, so the stdlib-only suite still runs anywhere;
CI installs it from `requirements.txt` — the file the public deploy uses — so the
page is checked against the dependency it is actually served with.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from triage import demo as demo_mod
from triage import observe, schema

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="streamlit is not installed (pip install -r requirements.txt)",
)

APP = str(Path(__file__).resolve().parent.parent / "app.py")
EXAMPLES = demo_mod.load_examples()["examples"]
IDS = [f"{i}-{e['incident']['id']}" for i, e in enumerate(EXAMPLES)]


def run_app(index: int):
    """The real script, with the incident selector moved to `index`."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception, app.exception
    app.selectbox[0].set_value(index).run()
    assert not app.exception, app.exception
    return app


@pytest.mark.parametrize("index", range(len(EXAMPLES)), ids=IDS)
def test_every_baked_incident_renders(index):
    """Every entry in the showcase has to survive being displayed. The page ships
    with the repository, so a shape it cannot render is a broken public page."""
    app = run_app(index)
    assert app.title[0].value
    assert len(app.selectbox[0].options) == len(EXAMPLES)


@pytest.mark.parametrize("index", range(len(EXAMPLES)), ids=IDS)
def test_the_page_states_the_outcome_the_transcript_recorded(index):
    """§6: the decision is the whole point, so the page must not soften it. An
    abstention rendered as a proposal would invert what this project claims."""
    app = run_app(index)
    result, _, _ = demo_mod.restore(EXAMPLES[index])

    if result.outcome is schema.Outcome.PROPOSE:
        assert app.success, "a proposal must render as a proposal"
        assert result.proposed_action in app.success[0].value
        assert not app.info
        # §5: a recommendation is only auditable with the sections it rests on.
        rendered = " ".join(m.value for m in app.markdown)
        assert result.citations
        for citation in result.citations:
            assert citation.section in rendered
    else:
        assert app.info, "an abstention must render as an abstention"
        shown = app.info[0].value
        assert "ABSTAIN" in shown
        assert result.escalation_target in shown   # §6.2: escalation names a human
        assert result.escalation_reason in shown
        assert not app.success


@pytest.mark.parametrize("index", range(len(EXAMPLES)), ids=IDS)
def test_the_page_prices_the_incident_from_its_tokens(index):
    """§12: the dollars on screen are re-derived from the recorded token counts,
    not read out of the baked file — so a rate change re-prices the page instead
    of leaving it quoting a number the project no longer computes."""
    app = run_app(index)
    _, trace, _ = demo_mod.restore(EXAMPLES[index])
    shown = next(m for m in app.metric if m.label.startswith("Cost"))
    assert shown.value == f"${observe.cost_usd(trace.usage_by_model)['total']:.4f}"


@pytest.mark.parametrize("index", range(len(EXAMPLES)), ids=IDS)
def test_the_page_reports_the_budget_verdict_the_ceiling_gives(index):
    """§8: the cap is inclusive, and the page is one more place that comparison
    is written down. It has to agree with the seven in the package."""
    app = run_app(index)
    _, trace, budget = demo_mod.restore(EXAMPLES[index])
    verdict = next(m for m in app.metric if m.label == "Budget")
    assert verdict.value == f"${budget:.4f}"
    assert verdict.delta == ("EXCEEDED" if trace.cost() > budget else "within")


def test_the_budget_trip_is_visible_as_a_stage_that_never_ran():
    """The cost ceiling is only a real guarantee if skipping the expensive stage
    is observable. The transcript that trips it must say the draft never ran."""
    index = next(i for i, e in enumerate(EXAMPLES)
                 if e["result"]["escalation_reason"] == "cost_budget_exceeded")
    app = run_app(index)
    _, trace, _ = demo_mod.restore(EXAMPLES[index])
    assert "draft" not in dict(trace.spans)
    assert any("never ran" in c.value for c in app.caption)


def test_the_page_ships_no_runbook_text():
    """§11: citations are pointers. The demo file carries section paths and never
    runbook bodies, and the page must not become the place that leaks them."""
    bodies = [p.read_text(encoding="utf-8")
              for p in (Path(APP).parent / "fixtures" / "runbooks").glob("*.md")]
    rendered = " ".join(m.value for m in run_app(0).markdown)
    for body in bodies:
        for line in body.splitlines():
            line = line.strip()
            # section headings are legitimately quoted by citations; prose is not
            if len(line) > 60 and not line.startswith("#"):
                assert line not in rendered
