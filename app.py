"""Streamlit front-end: one incident, one triage transcript, one trace.

The thing worth showing here is not that the agent answers — it is that it
declines. So the page leads with the outcome, and four of the five transcripts
are refusals: a SEV1 that is human-only by policy, an incident no runbook
covers, a ticket demanding a destructive fix, and a run that hit its cost
ceiling before it could buy the expensive stage.

**Cached only, and deliberately so.** Every transcript is replayed from
`demo/examples.json` — real measured runs, baked by `python -m triage bake-demo`
— through `triage.demo.restore()`, the same function the CLI replays through.
Zero LLM calls, no API key, no runbook index, nothing to leak and nothing to
spend. There is no live mode: a live agent run needs the Claude Code CLI on
PATH, which a hosted app does not have, and a live pipeline run needs a built
index that this repository does not ship (design §11).

Dollars are re-derived from the recorded token counts at render time, so a
change to the rate table re-prices the page rather than leaving a stale figure.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from triage import demo as demo_mod
from triage import observe, schema

ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Hal-Hanami/incident-triage-agent"

st.set_page_config(page_title="incident-triage-agent", page_icon="🚨", layout="centered")


@st.cache_data
def load_examples() -> dict:
    return demo_mod.load_examples()


def _blob(source: str) -> str:
    """A citation's `path#anchor` as a link to that runbook section on GitHub."""
    path, _, anchor = source.partition("#")
    return f"{REPO_URL}/blob/main/{path}" + (f"#{anchor}" if anchor else "")


def render_decision(result: schema.TriageResult) -> None:
    """The outcome first: what the agent decided, and who now owns the incident."""
    if result.outcome is schema.Outcome.PROPOSE:
        st.success(f"**PROPOSE** — `{result.proposed_action}`  ·  for the on-call human to apply")
        st.caption("A recommendation, not an action. This agent has no tool that could apply it.")
        st.markdown("**Citations** &nbsp; :green[✓ every claim is grounded in a runbook section]")
        for c in result.citations:
            st.markdown(
                f"<span style='color:#888'>[{c.n}]</span> "
                f"[{c.section}]({_blob(c.source)})",
                unsafe_allow_html=True,
            )
    else:
        st.info(f"**ABSTAIN → escalate to {result.escalation_target}**  ·  "
                f"reason: `{result.escalation_reason}`")
        st.caption(
            "Abstaining is a first-class success here, not a failure. The agent "
            "proposed nothing and handed the incident to a named human."
        )


def render_classification(result: schema.TriageResult) -> None:
    c1, c2 = st.columns(2)
    c1.metric("Severity", result.severity.value,
              f"confidence {result.severity_confidence:.2f}", delta_color="off")
    c2.metric("Type", result.type.value,
              f"confidence {result.type_confidence:.2f}", delta_color="off")
    retrieved = ", ".join(f"`{r}`" for r in result.retrieved_runbooks) or "_(none)_"
    st.markdown(f"**Runbooks retrieved:** {retrieved}")


def render_trace(trace: observe.Trace, budget: float) -> None:
    """What the incident cost and where the time went (design §7, §8)."""
    spent = trace.cost()
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost (this incident)", f"${spent:.4f}")
    c2.metric("Latency", f"{trace.total_seconds:.1f}s")
    c3.metric("Budget", f"${budget:.4f}",
              "within" if spent <= budget else "EXCEEDED", delta_color="off")

    with st.expander("🔎 trace — per-stage latency + per-model cost"):
        stages = " · ".join(f"{name} {secs * 1000:.0f}ms" for name, secs in trace.spans)
        st.markdown(f"**Stages:** {stages}")
        costs = observe.cost_usd(trace.usage_by_model)
        rows = ["| model | tokens | USD |", "|---|---|---|"]
        for model, usage in sorted(trace.usage_by_model.items()):
            toks = ", ".join(f"{k.replace('_tokens', '')}={v}" for k, v in usage.items())
            rows.append(f"| `{model}` | {toks} | ${costs.get(model, 0.0):.6f} |")
        rows.append(f"| **total** | | **${spent:.4f}** |")
        st.markdown("\n".join(rows))
        if "draft" not in dict(trace.spans):
            st.caption("The draft stage never ran — the cost ceiling tripped first, "
                       "so the expensive model was never bought (design §8).")


# --- page ---------------------------------------------------------------------

st.title("🚨 incident-triage-agent")
st.markdown(
    "Read-only first-pass incident triage on the **Claude Agent SDK + MCP**. It "
    "classifies an alert, retrieves the matching runbook, and proposes a **cited "
    "first response** — or **abstains and escalates to a human** when confidence "
    "is low, the situation is out of scope, or policy says a person decides. "
    f"It never executes remediation. [Code & design notes →]({REPO_URL})"
)

payload = load_examples()
examples = payload["examples"]

choice = st.selectbox(
    "Pick an incident",
    range(len(examples)),
    format_func=lambda i: f"{examples[i]['incident']['id']} — {examples[i]['story']}",
)
example = examples[choice]
result, trace, budget = demo_mod.restore(example)

st.markdown(f"### {example['incident']['title']}")
render_decision(result)
st.divider()
render_classification(result)
render_trace(trace, budget)

st.caption(
    f"Cached transcript · baked {payload['generated_at']} from real runs for "
    f"${payload['bake_cost_usd']:.4f} · classify **{payload['models']['classify']}** · "
    f"draft **{payload['models']['draft']}** · retrieval = {payload['models']['retrieval']}. "
    "Real measured numbers — no live call, no key, no runbook text shipped."
)

st.divider()
# These figures are quoted from docs/EVALUATION.md, not maintained here. A test
# checks that each still derives from that file, so the page cannot go on
# claiming a number the record has moved past.
st.caption(
    "**What is actually measured** — over four full-pipeline runs on a frozen set "
    "of 32 synthetic incidents: **100%** abstention on the 15 that must abstain and "
    "**0** missed escalations, in every run. Everything else moved: 1–2 of 17 "
    "answerable incidents were over-escalated, action correctness ran 93.3–93.8%, "
    "and severity accuracy swung 68.2–81.8% — the least stable number here, and "
    "reported as a range for that reason. Cost $0.0135–$0.0142 per incident. "
    f"Every figure, its date, and its reproduce command: "
    f"[docs/EVALUATION.md]({REPO_URL}/blob/main/docs/EVALUATION.md)"
)
