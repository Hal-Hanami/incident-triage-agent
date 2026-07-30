"""Offline tests for the command line — the flag surface and the wiring it drives.

This module holds no triage logic, which is exactly why it was the last place a
defect could sit unseen: every interesting command needs a key, so the suite
never entered it and the translation from parsed flag to constructed stage went
unchecked. A green suite said nothing about whether `--no-draft` actually
withheld the Opus drafter, because no test had ever asked.

What is pinned here is that translation and nothing else — which stages a flag
combination builds (design §4), which caps reach the harness (§8), and the
guards that must fire *before* a paid client is ever constructed. The stages
themselves are fakes; there is no key, no network, and no index.
"""

from __future__ import annotations

import argparse

import pytest

from triage import __main__ as cli
from triage import classify as classify_mod
from triage import draft as draft_mod
from triage import eval as eval_mod
from triage import observe
from triage import retrieve as retrieve_mod
from triage import schema


def parse(argv: list[str]) -> argparse.Namespace:
    """One argv through the real parser — the same object `main()` dispatches on."""
    return cli.build_parser().parse_args(argv)


# --- the flag surface -------------------------------------------------------------

@pytest.mark.parametrize("argv,expected", [
    (["incidents"], cli.cmd_incidents),
    (["validate"], cli.cmd_validate),
    (["index-runbooks"], cli.cmd_index_runbooks),
    (["eval"], cli.cmd_eval),
    (["triage", "INC-0001"], cli.cmd_triage),
    (["agent", "INC-0001"], cli.cmd_agent),
    (["demo"], cli.cmd_demo),
    (["bake-demo"], cli.cmd_bake_demo),
])
def test_every_subcommand_dispatches_to_its_handler(argv, expected):
    assert parse(argv).func is expected


def test_a_bare_invocation_is_an_error_rather_than_a_default_command():
    """`required=True` on the subparser: no argv must ever mean "run the paid one"."""
    with pytest.raises(SystemExit):
        parse([])


@pytest.mark.parametrize("argv", [
    ["triage", "INC-0001"], ["agent", "INC-0001"], ["eval"],
])
def test_the_retrieval_and_budget_defaults_are_the_documented_ones(argv):
    args = parse(argv)
    assert args.k == retrieve_mod.DEFAULT_K == 5
    assert args.no_rerank is False           # rerank is on unless switched off
    assert args.budget == observe.DEFAULT_INCIDENT_BUDGET_USD


def test_eval_defaults_select_everything_and_cap_nothing():
    args = parse(["eval"])
    assert (args.scope, args.limit, args.max_cost) == ("all", 0, None)
    assert not (args.classify_only or args.retrieval_only or args.no_draft or args.judge)


@pytest.mark.parametrize("pair", [
    ("--classify-only", "--retrieval-only"),
    ("--classify-only", "--no-draft"),
    ("--retrieval-only", "--no-draft"),
])
def test_the_stage_flags_cannot_be_combined(pair):
    """They name three different pipelines; accepting two would run one of them
    while the banner claimed the other."""
    with pytest.raises(SystemExit):
        parse(["eval", *pair])


def test_scope_rejects_a_value_the_harness_does_not_understand():
    with pytest.raises(SystemExit):
        parse(["incidents", "--scope", "sideways"])


# --- flags in, stages out ----------------------------------------------------------

@pytest.fixture
def wiring(monkeypatch):
    """Replace every paid constructor and the harness itself with recorders.

    Nothing here reaches a network: the point is to observe *which* stages the
    CLI decided to build, and with what, for a given argv.
    """
    seen: dict = {"built": [], "retriever_kw": None, "evaluate_kw": None}

    class FakeRetriever:
        def __init__(self, *a, **kw):
            seen["built"].append("retrieve")
            seen["retriever_kw"] = kw

    def built(name):
        def make(*a, **kw):
            seen["built"].append(name)
            return object()
        return make

    monkeypatch.setattr(classify_mod, "ClaudeClassifier", built("classify"))
    monkeypatch.setattr(draft_mod, "ClaudeDrafter", built("draft"))
    monkeypatch.setattr(eval_mod, "ClaudeJudge", built("judge"))
    monkeypatch.setattr(retrieve_mod, "RagRetriever", FakeRetriever)

    def fake_evaluate(incidents, classifier=None, retriever=None, drafter=None, **kw):
        seen["evaluate_kw"] = kw
        seen["stages"] = {"classifier": classifier is not None,
                          "retriever": retriever is not None,
                          "drafter": drafter is not None,
                          "judge": kw.get("judge") is not None}
        return []

    monkeypatch.setattr(eval_mod, "evaluate", fake_evaluate)
    monkeypatch.setattr(eval_mod, "summarize", lambda rows: {})
    monkeypatch.setattr(eval_mod, "format_report", lambda rows, summary: "(report)")
    return seen


@pytest.mark.parametrize("flags,expected", [
    ([],                   {"classify", "retrieve", "draft"}),   # the full pipeline
    (["--classify-only"],  {"classify"}),                        # no index, no Opus
    (["--retrieval-only"], {"retrieve"}),                        # no Anthropic spend
    (["--no-draft"],       {"classify", "retrieve"}),            # no Opus spend
])
def test_each_stage_flag_builds_exactly_the_stages_it_names(wiring, flags, expected):
    """A stage that is not built cannot be billed — this is the money-shaped claim
    the README's `--classify-only` / `--no-draft` reproduce commands rest on."""
    args = parse(["eval", *flags])
    args.func(args)
    assert set(wiring["built"]) == expected


def test_the_retrieval_flags_reach_the_retriever(wiring):
    args = parse(["eval", "-k", "9", "--no-rerank"])
    args.func(args)
    assert wiring["retriever_kw"] == {"k": 9, "rerank": False}


def test_rerank_stays_on_when_the_flag_is_absent(wiring):
    args = parse(["eval"])
    args.func(args)
    assert wiring["retriever_kw"] == {"k": 5, "rerank": True}


def test_the_caps_and_the_selection_reach_the_harness(wiring):
    """§8: `--budget` is per incident, `--max-cost` is the aggregate fail-safe.
    They are different ceilings and must not be collapsed into one."""
    args = parse(["eval", "--budget", "0.02", "--max-cost", "0.30",
                  "--scope", "out", "--limit", "4"])
    args.func(args)
    assert wiring["evaluate_kw"]["budget"] == 0.02
    assert wiring["evaluate_kw"]["max_total_usd"] == 0.30
    assert wiring["evaluate_kw"]["scope"] == "out"
    assert wiring["evaluate_kw"]["limit"] == 4


def test_the_judge_is_built_only_when_asked_for(wiring):
    args = parse(["eval", "--judge"])
    args.func(args)
    assert "judge" in wiring["built"] and wiring["stages"]["judge"]


@pytest.mark.parametrize("flag", ["--classify-only", "--retrieval-only", "--no-draft"])
def test_the_judge_refuses_a_pipeline_that_produces_nothing_to_grade(wiring, flag):
    """§10: the judge grades PROPOSE outcomes, and only the full pipeline emits
    them. Accepting the combination would bill the judge to score an empty set."""
    args = parse(["eval", "--judge", flag])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert "--judge" in str(exc.value)
    assert "judge" not in wiring["built"]


def test_the_aggregate_cap_is_reported_when_it_stops_the_run(wiring, capsys):
    """§8 fail-safe: fewer rows than selected means the run was cut short, and a
    report that did not say so would read as a complete measurement."""
    args = parse(["eval", "--limit", "3", "--budget", "0.02"])
    args.func(args)                                     # fake_evaluate returns []
    out = capsys.readouterr().out
    assert "aggregate cost cap" in out
    assert "0 of 3 incidents" in out
    assert f"${0.02 * 3:.4f}" in out                    # default cap = budget x selected


def test_a_complete_run_reports_no_cap(wiring, monkeypatch, capsys):
    monkeypatch.setattr(eval_mod, "select", lambda incidents, **kw: [])
    args = parse(["eval"])
    args.func(args)
    assert "aggregate cost cap" not in capsys.readouterr().out


# --- the guards that run before anything is paid for --------------------------------

@pytest.mark.parametrize("command", ["triage", "agent"])
def test_an_unknown_incident_exits_before_any_client_is_constructed(wiring, command):
    """A typo must cost nothing. The lookup has to precede the constructors, which
    are what open a connection and demand a key."""
    args = parse([command, "INC-9999"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert wiring["built"] == []


@pytest.mark.parametrize("command", ["triage", "agent"])
def test_the_red_team_incident_is_addressable_but_not_in_the_measured_set(command):
    """INC-R001 lives in its own fixture so adding it never moves the published
    numbers (design §3), yet the single-incident commands must still reach it."""
    index = cli._incident_index()
    assert "INC-R001" in index
    assert "INC-R001" not in {i.id for i in schema.load_incidents(cli.INCIDENTS_FILE)}


# --- the offline commands ------------------------------------------------------------

@pytest.mark.parametrize("scope,expected", [
    ("in", lambda i: i.in_scope),
    ("out", lambda i: not i.in_scope),
    ("all", lambda i: True),
])
def test_incidents_lists_exactly_the_scope_it_was_asked_for(scope, expected, capsys):
    args = parse(["incidents", "--scope", scope])
    args.func(args)
    out = capsys.readouterr().out
    wanted = [i for i in schema.load_incidents(cli.INCIDENTS_FILE) if expected(i)]
    assert f"{len(wanted)} incidents (scope={scope})" in out
    for inc in wanted:
        assert inc.id in out


def test_incidents_marks_the_must_abstain_cases(capsys):
    """The abstention set is the point of the fixture; a listing that did not
    distinguish it would hide what the numbers are measured over."""
    args = parse(["incidents"])
    args.func(args)
    lines = {ln.split()[0]: ln for ln in capsys.readouterr().out.splitlines() if "INC-" in ln}
    for inc in schema.load_incidents(cli.INCIDENTS_FILE):
        assert ("abstain" in lines[inc.id]) is inc.must_abstain


def test_validate_passes_on_the_shipped_fixtures(capsys):
    args = parse(["validate"])
    args.func(args)
    assert "OK" in capsys.readouterr().out


def test_validate_exits_nonzero_when_the_fixtures_are_broken(monkeypatch, capsys):
    """CI runs this command; it has to fail the build rather than print and pass."""
    broken = schema.Incident(id="INC-X", title="t", body="b", source="ticket",
                             gold_severity=schema.Severity.SEV3,
                             gold_type=schema.IncidentType.UNKNOWN,
                             gold_action="not_a_real_action", in_scope=False)
    monkeypatch.setattr(schema, "load_incidents", lambda path: [broken])
    args = parse(["validate"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert "INVALID" in capsys.readouterr().out


def test_demo_replays_the_baked_transcripts_with_no_key(capsys):
    args = parse(["demo"])
    args.func(args)
    out = capsys.readouterr().out
    assert "key-free cached demo" in out
    for outcome in ("PROPOSE", "ABSTAIN"):
        assert outcome in out          # the showcase tells both halves of §6


def test_demo_can_replay_one_incident(capsys):
    args = parse(["demo", "INC-0003"])
    args.func(args)
    out = capsys.readouterr().out
    assert "INC-0003" in out and "INC-0008" not in out


def test_demo_names_the_baked_incidents_when_asked_for_one_it_lacks(capsys):
    args = parse(["demo", "INC-0002"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert "INC-0001" in capsys.readouterr().out      # tells the caller what exists


# --- credentials ---------------------------------------------------------------------

def test_dotenv_fills_only_what_the_environment_has_not_already_set(tmp_path, monkeypatch):
    """An exported key must win over a stale `.env` — the shell is the more
    deliberate of the two."""
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nANTHROPIC_API_KEY="from-file"\nVOYAGE_API_KEY=voyage\nnot a pair\n')
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    cli._load_dotenv(env)

    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"   # not overridden
    assert os.environ["VOYAGE_API_KEY"] == "voyage"          # quotes stripped, filled in


def test_a_missing_dotenv_is_not_an_error(tmp_path):
    cli._load_dotenv(tmp_path / "nope.env")                  # the documented normal case


# --- the shared renderer --------------------------------------------------------------

def _result(**kw) -> schema.TriageResult:
    base = dict(incident_id="INC-0001", severity=schema.Severity.SEV2,
                severity_confidence=0.9, type=schema.IncidentType.APP_ERROR,
                type_confidence=0.8, outcome=schema.Outcome.PROPOSE,
                retrieved_runbooks=["RB-app-5xx"])
    return schema.TriageResult(**{**base, **kw})


def test_a_proposal_renders_its_action_and_every_citation():
    """§5: a recommendation without its citations is not auditable, and the
    renderer is the only thing a human actually reads."""
    out = cli._format_triage("Checkout 5xx", _result(
        proposed_action="roll_back_last_deploy",
        citations=[schema.Citation(n=1, runbook_id="RB-app-5xx",
                                   section="Rollback", source="fixtures/runbooks/RB-app-5xx.md")]))
    assert "PROPOSE" in out and "roll_back_last_deploy" in out
    assert "[1] Rollback" in out and "RB-app-5xx.md" in out


def test_an_abstention_renders_the_human_it_hands_off_to():
    """§6.2: escalation names a target and a reason. "ABSTAIN" alone tells the
    on-call nothing about who now owns the incident."""
    out = cli._format_triage("Stuck ETL", _result(
        outcome=schema.Outcome.ABSTAIN, escalation_target="data on-call",
        escalation_reason="no_supporting_runbook"))
    assert "ABSTAIN" in out
    assert "data on-call" in out and "no_supporting_runbook" in out
    assert "citations" not in out          # nothing to cite when nothing is proposed


# --- the single-incident commands ------------------------------------------------------

def test_triage_hands_the_flags_to_the_pipeline_and_prints_the_trace(wiring, monkeypatch, capsys):
    """The one paid path a reader reproduces from the README. What it must not do
    is quietly run a different configuration than the flags named."""
    from triage import decide as decide_mod

    seen: dict = {}

    def fake_run_triage(inc, classifier, retriever, drafter, *, budget, trace):
        seen["incident"] = inc.id
        seen["budget"] = budget
        trace.add_usage("claude-haiku-4-5", {"input_tokens": 1000, "output_tokens": 0})
        return _result(incident_id=inc.id, proposed_action="roll_back_last_deploy")

    monkeypatch.setattr(decide_mod, "run_triage", fake_run_triage)
    args = parse(["triage", "INC-0001", "-k", "7", "--no-rerank", "--budget", "0.02"])
    args.func(args)

    assert seen == {"incident": "INC-0001", "budget": 0.02}
    assert wiring["retriever_kw"] == {"k": 7, "rerank": False}
    out = capsys.readouterr().out
    assert "PROPOSE" in out
    assert "claude-haiku-4-5" in out            # §7: the per-model footer rendered


class _FakeAgentRun:
    """Enough of an `AgentTriage` for the CLI footer, with no SDK anywhere near it."""

    def __init__(self, *, budget, stage_usd, sdk_usd, computed_usd,
                 tool_calls=(), denied=()):
        from triage import agent as agent_mod

        trace = observe.Trace()
        # Price a token count backwards into the stage spend the footer must report.
        trace.add_usage("claude-haiku-4-5", {"input_tokens": int(stage_usd * 1_000_000)})
        self.session = agent_mod.AgentSession(
            incident=schema.load_incidents(cli.INCIDENTS_FILE)[0],
            classifier=object(), retriever=object(), drafter=object(),
            budget=budget, trace=trace)
        self.result = _result(outcome=schema.Outcome.ABSTAIN,
                              escalation_target="on-call SRE",
                              escalation_reason="sev1_human_only")
        self.tool_calls = list(tool_calls)
        self.denied_calls = list(denied)
        self.total_cost_usd = sdk_usd
        self._computed = computed_usd
        self.num_turns = 3
        self.duration_ms = 4200

    @property
    def non_triage_tool_calls(self):
        from triage import agent as agent_mod
        return [t for t in self.tool_calls if not agent_mod.is_tool_allowed(t)]

    @property
    def orchestrator_cost(self):
        return self._computed

    @property
    def cost_crosscheck_usd(self):
        if self._computed is None or self.total_cost_usd is None:
            return None
        return self._computed - self.total_cost_usd


def _run_agent_cli(monkeypatch, capsys, run, argv=("agent", "INC-0001")):
    from triage import agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_agent_triage", lambda *a, **kw: run)
    args = parse(list(argv))
    args.func(args)
    return capsys.readouterr().out


def test_the_agent_footer_reports_the_guardrail_outcome(wiring, monkeypatch, capsys):
    """§9: the read-only claim is only credible if every run says what it called
    and what was denied. "NONE" has to be a measured statement, not a default."""
    run = _FakeAgentRun(budget=0.05, stage_usd=0.001, sdk_usd=0.002, computed_usd=0.002,
                        tool_calls=["mcp__triage__classify_incident", "mcp__triage__escalate"],
                        denied=["Bash"])
    out = _run_agent_cli(monkeypatch, capsys, run)
    assert "mcp__triage__classify_incident, mcp__triage__escalate" in out
    assert "denied      Bash" in out
    assert "NONE (read-only held)" in out


def test_the_agent_footer_names_a_call_that_escaped_the_allowlist(wiring, monkeypatch, capsys):
    """The red-team metric must be able to report a failure. A line that can only
    ever print NONE is not evidence of anything."""
    run = _FakeAgentRun(budget=0.05, stage_usd=0.001, sdk_usd=0.002, computed_usd=0.002,
                        tool_calls=["mcp__triage__classify_incident", "Bash"])
    out = _run_agent_cli(monkeypatch, capsys, run)
    assert "non-allowlisted calls: ['Bash']" in out
    assert "read-only held" not in out


@pytest.mark.parametrize("budget,expected", [
    (0.05,   "within"),      # comfortably under
    (0.0030, "within"),      # §8: the cap is inclusive — spending it exactly is not crossing it
    (0.0029, "EXCEEDED"),    # a hundredth of a cent over is over
    (0.002,  "EXCEEDED"),
])
def test_the_incident_budget_verdict_covers_stages_plus_orchestrator(
        wiring, monkeypatch, capsys, budget, expected):
    """§8 under the agent shell: the ceiling is the *incident's*, so the
    orchestrator's own spend counts against it too. Judging the stages alone
    would let a chatty orchestrator run past the cap and still read "within"."""
    run = _FakeAgentRun(budget=budget, stage_usd=0.001, sdk_usd=0.002, computed_usd=0.002)
    out = _run_agent_cli(monkeypatch, capsys, run)
    assert "0.0030 = stages 0.0010 + orchestrator 0.0020" in out
    assert expected in out


def test_the_budget_verdict_falls_back_to_our_own_price_when_the_sdk_reports_none(
        wiring, monkeypatch, capsys):
    """A missing `total_cost_usd` must not silently price the orchestrator at $0 —
    that would turn an unmeasured run into one that looks free."""
    run = _FakeAgentRun(budget=0.05, stage_usd=0.001, sdk_usd=None, computed_usd=0.002)
    out = _run_agent_cli(monkeypatch, capsys, run)
    assert "stages 0.0010 + orchestrator 0.0020" in out
    assert "SDK=n/a" in out and "computed=$0.0020" in out
    assert "delta n/a" in out               # §7 cross-check needs both accountings


def test_the_cross_check_delta_is_signed(wiring, monkeypatch, capsys):
    """§7: the sign says which accounting is high. An absolute value would hide
    whether our rate table over- or under-charges."""
    run = _FakeAgentRun(budget=0.05, stage_usd=0.001, sdk_usd=0.0020, computed_usd=0.0025)
    out = _run_agent_cli(monkeypatch, capsys, run)
    assert "delta +0.0005" in out


# --- the remaining commands -------------------------------------------------------------

def test_index_runbooks_reports_what_it_built(wiring, monkeypatch, capsys):
    monkeypatch.setattr(retrieve_mod, "index_runbooks", lambda: {
        "count": 42, "runbooks": 7, "dim": 1024, "model": "voyage-4-lite",
        "embed_secs": 1.5, "db_path": "data/runbooks.db"})
    args = parse(["index-runbooks"])
    args.func(args)
    out = capsys.readouterr().out
    assert "indexed 42 chunks from 7 runbooks" in out and "data/runbooks.db" in out


def test_bake_demo_writes_the_payload_it_baked(wiring, monkeypatch, capsys, tmp_path):
    from triage import demo as demo_mod

    payload = {"examples": [{"trace": {"total_usd": 0.01}}], "bake_cost_usd": 0.01}
    monkeypatch.setattr(demo_mod, "bake_examples", lambda *a, **kw: payload)
    monkeypatch.setattr(demo_mod, "write_examples",
                        lambda p: cli.ROOT / "demo" / "examples.json")
    args = parse(["bake-demo"])
    args.func(args)
    out = capsys.readouterr().out
    assert f"baking {len(demo_mod.SHOWCASE)} showcase runs" in out
    assert "demo/examples.json" in out and "1 entries" in out


def test_main_loads_credentials_before_it_parses_and_dispatches(monkeypatch):
    """`.env` has to be read before the handler runs, or the documented key path
    only works for people who also exported the variable."""
    order: list[str] = []
    monkeypatch.setattr(cli, "_load_dotenv", lambda *a, **kw: order.append("dotenv"))
    monkeypatch.setattr(cli, "cmd_validate", lambda args: order.append("dispatch"))
    monkeypatch.setattr("sys.argv", ["triage", "validate"])
    cli.main()
    assert order == ["dotenv", "dispatch"]
