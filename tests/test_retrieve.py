"""Offline tests for the retrieve stage + runbook corpus adapter (design §4.2).

No `rag`, no Voyage, no key, no network: the chunker is pure stdlib, and the live
`RagRetriever` sits behind the `Retriever` Protocol so a fake drives the orchestration
+ recall scoring. Asserts the corpus-adapter row shape (so the `rag` index can
consume it), the dedup-to-runbook ranking, the trace/cost wiring, and the recall@k
math end to end on the real fixtures.
"""

from __future__ import annotations

from pathlib import Path

from triage import observe, retrieve, runbooks, schema

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "incidents" / "incidents.jsonl"
_ROW_KEYS = {"id", "source_url", "url", "page_title", "section_path", "anchor", "text"}


# --- the runbook corpus adapter (pure stdlib) -------------------------------

def test_chunk_runbook_rows_carry_index_keys_and_the_runbook_id():
    rows = runbooks.chunk_runbook(runbooks.RUNBOOK_DIR / "RB-app-5xx.md")
    assert len(rows) >= 2  # preamble + at least one section
    for r in rows:
        assert _ROW_KEYS <= set(r)          # every key rag.store.insert reads
        assert r["source_url"] == "RB-app-5xx"  # the recall key
        assert r["text"].strip()
    assert rows[0]["anchor"] == ""          # preamble chunk has no section anchor
    assert any(r["anchor"] for r in rows[1:])  # later chunks anchor to their H2
    # the action key the runbook documents survives into a chunk (drafter signal)
    assert any("roll_back_last_deploy" in r["text"] for r in rows)


def test_build_runbook_chunks_covers_every_runbook_and_every_recall_target():
    rows = runbooks.build_runbook_chunks()
    indexed = {r["source_url"] for r in rows}
    assert indexed == set(runbooks.runbook_ids())          # all 7 runbooks chunked
    # every in-scope incident's recall target exists in the corpus (eval ground truth)
    targets = {i.expected_runbook for i in schema.load_incidents(FIXTURE)
               if i.in_scope and i.expected_runbook}
    assert targets <= indexed


def test_chunk_ids_are_unique_across_the_corpus():
    rows = runbooks.build_runbook_chunks()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


# --- stage helpers ----------------------------------------------------------

def test_build_query_uses_title_and_body_only():
    view = {"id": "INC-0001", "title": "5xx on checkout", "body": "error_rate > 5%",
            "source": "pagerduty"}
    q = retrieve.build_query(view)
    assert "5xx on checkout" in q and "error_rate > 5%" in q
    assert "INC-0001" not in q and "pagerduty" not in q  # query is the symptom text


def test_build_retrieval_dedups_to_runbooks_in_rank_order():
    chunks = [
        {"source_url": "RB-app-5xx", "score": 0.91, "chunk_id": "RB-app-5xx#1",
         "section_path": "…", "url": "u1", "text": "t1"},
        {"source_url": "RB-app-5xx", "score": 0.40, "chunk_id": "RB-app-5xx#0",
         "section_path": "…", "url": "u0", "text": "t0"},   # same runbook, lower — dropped
        {"source_url": "RB-latency", "score": 0.55, "chunk_id": "RB-latency#2",
         "section_path": "…", "url": "u2", "text": "t2"},
    ]
    ret = retrieve.build_retrieval(chunks)
    assert ret.runbook_ids == ["RB-app-5xx", "RB-latency"]  # first occurrence wins
    assert ret.top_score == 0.91
    assert len(ret.chunks) == 3                              # raw chunks kept for the drafter


# --- a fake retriever drives the orchestration + eval -----------------------

class GoldRetriever:
    """Test oracle: returns the incident's expected runbook as the top hit. `miss_ids`
    forces a miss (a non-existent runbook) and `distractor` ranks a wrong runbook above
    the expected one to exercise recall@1 < recall@3."""

    def __init__(self, incidents, *, miss_ids=(), distractor=None):
        self.by_id = {i.id: i for i in incidents}
        self.miss_ids = set(miss_ids)
        self.distractor = distractor

    def retrieve(self, view):
        inc = self.by_id[view["id"]]
        ids = []
        if self.distractor:
            ids.append(self.distractor)
        if inc.id not in self.miss_ids and inc.expected_runbook:
            ids.append(inc.expected_runbook)
        ids = ids or ["RB-__miss__"]  # always non-empty; sentinel is never a real target
        hits = [retrieve.RunbookHit(runbook_id=rb, score=1.0 - 0.1 * n)
                for n, rb in enumerate(ids)]
        return retrieve.Retrieval(runbooks=hits, usage={"voyage-4-lite": {"total_tokens": 4}})


def test_retrieve_runbooks_times_and_costs_when_traced():
    inc = schema.load_incidents(FIXTURE)[0]
    trace = observe.Trace()
    out = retrieve.retrieve_runbooks(inc.prompt_view(), GoldRetriever([inc]), trace=trace)
    assert out.runbook_ids == [inc.expected_runbook]
    assert [name for name, _ in trace.spans] == ["retrieve"]
    # Voyage tokens are filed under the embedding model for the per-model ledger (§7)
    assert trace.usage_by_model["voyage-4-lite"] == {"total_tokens": 4}

