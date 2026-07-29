"""Offline tests for the classify seam (design §4.1).

The real claude-haiku-4-5 path (`ClaudeClassifier`) is exercised offline; a fake
`Classifier` stands in so the orchestration, the prompt-view contract, and the
trace/cost wiring are asserted with no key and no network.
"""

from __future__ import annotations

import pytest

from triage import classify, observe
from triage.schema import IncidentType, Severity


class FakeClassifier:
    """Records the view it was handed and returns a canned Classification."""

    model = "fake-classifier"

    def __init__(self, severity=Severity.SEV2, type=IncidentType.APP_ERROR,
                 sev_conf=0.9, type_conf=0.8):
        self.result = classify.Classification(
            severity=severity, type=type,
            severity_confidence=sev_conf, type_confidence=type_conf,
            usage={"input_tokens": 5, "output_tokens": 2},
        )
        self.seen_view: dict | None = None

    def classify(self, view):
        self.seen_view = view
        return self.result


def test_classify_incident_returns_verdict_and_sees_only_the_prompt_view():
    fake = FakeClassifier()
    view = {"id": "INC-0001", "title": "t", "body": "b", "source": "pagerduty"}
    out = classify.classify_incident(view, fake)
    assert out.severity is Severity.SEV2 and out.type is IncidentType.APP_ERROR
    # the classifier is handed exactly the prompt view — no gold labels leak in
    assert fake.seen_view == view
    assert "gold_severity" not in fake.seen_view and "notes" not in fake.seen_view


def test_min_confidence_is_the_weaker_signal():
    c = classify.Classification(Severity.SEV1, IncidentType.DATABASE,
                                severity_confidence=0.95, type_confidence=0.4)
    assert c.min_confidence == 0.4


def test_classify_incident_times_and_costs_when_traced():
    fake = FakeClassifier()
    trace = observe.Trace()
    classify.classify_incident({"id": "x", "title": "", "body": "", "source": "s"},
                               fake, trace=trace)
    assert [name for name, _ in trace.spans] == ["classify"]
    # token usage is filed under the classifier's model for the per-model ledger (§7)
    assert trace.usage_by_model["fake-classifier"] == {"input_tokens": 5, "output_tokens": 2}


def test_classify_incident_without_trace_is_a_noop_timer():
    fake = FakeClassifier()
    out = classify.classify_incident({"id": "x", "title": "", "body": "", "source": "s"}, fake)
    # usage still rides on the returned Classification for the eval harness to record
    assert out.usage == {"input_tokens": 5, "output_tokens": 2}


# --- the real claude-haiku-4-5 path — prompt / schema / parse (offline) ---

def test_build_classify_user_carries_prompt_fields_only():
    view = {"id": "INC-0007", "title": "Spike in 401s", "body": "auth 401 rate 8x", "source": "pagerduty"}
    msg = classify.build_classify_user(view)
    assert "INC-0007" in msg and "Spike in 401s" in msg
    assert "auth 401 rate 8x" in msg and "pagerduty" in msg
    assert "gold" not in msg.lower()  # the view has no gold labels to leak


def test_classify_schema_pins_closed_enums_without_numeric_constraints():
    props = classify.CLASSIFY_SCHEMA["properties"]
    assert set(props["severity"]["enum"]) == {s.value for s in Severity}
    assert set(props["type"]["enum"]) == {t.value for t in IncidentType}
    assert classify.CLASSIFY_SCHEMA["additionalProperties"] is False
    # structured-output json_schema rejects numeric bounds -> confidence stays unbounded here
    assert "minimum" not in props["severity_confidence"]
    assert "maximum" not in props["severity_confidence"]


def test_parse_classification_reads_enums_and_clamps_confidence():
    out = classify.parse_classification(
        '{"severity": "SEV1", "type": "database", '
        '"severity_confidence": 1.4, "type_confidence": -0.2}')
    assert out.severity is Severity.SEV1 and out.type is IncidentType.DATABASE
    assert out.severity_confidence == 1.0 and out.type_confidence == 0.0  # clamped to [0,1]


def test_parse_classification_is_strict_on_a_bad_enum():
    with pytest.raises(ValueError):
        classify.parse_classification(
            '{"severity": "SEV9", "type": "database", '
            '"severity_confidence": 0.5, "type_confidence": 0.5}')


def test_system_prompt_lists_every_type_and_severity():
    for s in Severity:
        assert s.value in classify.SYSTEM
    for t in IncidentType:
        assert t.value in classify.SYSTEM
