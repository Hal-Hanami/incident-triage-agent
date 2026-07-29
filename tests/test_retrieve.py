"""Offline tests for the retrieve stage + runbook corpus adapter (design §4.2).

No `rag`, no Voyage, no key, no network: the chunker is pure stdlib, and the live
`RagRetriever` sits behind the `Retriever` Protocol so a fake drives the orchestration
+ recall scoring. Asserts the corpus-adapter row shape (so the `rag` index can
consume it), the dedup-to-runbook ranking, the trace/cost wiring, and the recall@k
math end to end on the real fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from triage import eval as eval_mod
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


def test_evaluate_with_perfect_retriever_scores_recall_100():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, retriever=GoldRetriever(incidents), scope="in")
    s = eval_mod.summarize(rows)
    assert s["n_recallable"] == sum(1 for i in incidents if i.in_scope and i.expected_runbook)
    assert s["recall_at_1"] == 1.0 and s["recall_at_3"] == 1.0 and s["recall_at_k"] == 1.0
    assert s["retrieval_mrr"] == 1.0
    # retrieval-only run: classification metrics stay n/a (no classifier ran)
    assert s["severity_accuracy"] is None and s["type_accuracy"] is None
    # Voyage tokens are tallied per model and priced (voyage-4-lite is in PRICING)
    assert s["usage_by_model"]["voyage-4-lite"]["total_tokens"] == 4 * s["n_recallable"]
    assert s["cost"]["total"] > 0.0


def test_evaluate_retriever_miss_lowers_recall():
    incidents = schema.load_incidents(FIXTURE)
    in_scope = [i for i in incidents if i.in_scope]
    miss = [i.id for i in in_scope][:2]
    rows = eval_mod.evaluate(incidents, retriever=GoldRetriever(incidents, miss_ids=miss),
                             scope="in")
    s = eval_mod.summarize(rows)
    n = s["n_recallable"]
    assert s["recall_at_k"] == (n - 2) / n


def test_evaluate_distractor_separates_recall_at_1_from_recall_at_3():
    incidents = schema.load_incidents(FIXTURE)
    # a wrong runbook ranked first pushes the expected one to rank 2 everywhere
    rows = eval_mod.evaluate(incidents, retriever=GoldRetriever(incidents, distractor="RB-thirdparty"),
                             scope="in")
    s = eval_mod.summarize(rows)
    assert s["recall_at_1"] < s["recall_at_3"] == 1.0  # rank-2 hits: missed@1, caught@3


class GoldClassifier:
    """Test oracle: echoes each incident's gold labels (the real classifier sees only
    the prompt view). Kept inline so this file needs no cross-test import."""

    model = "fake-classifier"

    def __init__(self, incidents):
        self.by_id = {i.id: i for i in incidents}

    def classify(self, view):
        from triage.classify import Classification
        inc = self.by_id[view["id"]]
        return Classification(inc.gold_severity, inc.gold_type, 0.9, 0.85,
                              usage={"input_tokens": 7, "output_tokens": 3})


def test_evaluate_runs_classify_and_retrieve_together():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents),
                             GoldRetriever(incidents), scope="in")
    s = eval_mod.summarize(rows)
    # both stages populate their metrics in one pass, per model
    assert s["severity_accuracy"] == 1.0 and s["type_accuracy"] == 1.0
    assert s["recall_at_k"] == 1.0
    assert set(s["usage_by_model"]) == {"fake-classifier", "voyage-4-lite"}


# --- the seam onto tech-docs-rag's `rag` package ---------------------------------
# `RagRetriever` is the only place that names modules inside another repository, so
# a rename there breaks this repo silently: the offline suite stays green because it
# never imports `rag`, and the failure only surfaces on a paid run. These tests build
# a stand-in `rag` package with the module layout this adapter requires, so the
# import contract is pinned without the sibling checkout, a key, or the network.

RAG_STUB = {
    "rag/__init__.py": "",
    "rag/search.py": """
CALLS = []
def search(query, db_path, embedder, k=5, *, hybrid=True, reranker=None, **kw):
    CALLS.append(dict(query=query, db_path=db_path, k=k, hybrid=hybrid,
                      reranker=reranker, embedder=embedder))
    embedder.usage["total_tokens"] += 11
    if reranker is not None:
        reranker.usage["total_tokens"] += 23
    return [{"source_url": "RB-app-5xx", "score": 0.9, "section_path": "First response",
             "url": "u1", "text": "t1", "chunk_id": "c1"},
            {"source_url": "RB-app-5xx", "score": 0.5, "section_path": "Other",
             "url": "u2", "text": "t2", "chunk_id": "c2"},
            {"source_url": "RB-latency", "score": 0.4, "section_path": "Sym",
             "url": "u3", "text": "t3", "chunk_id": "c3"}]
""",
    "rag/index.py": """
CALLS = []
def build(chunks_path, db_path, embedder, **kw):
    CALLS.append(dict(chunks_path=chunks_path, db_path=db_path, embedder=embedder))
    return {"chunks": 27, "dim": 1024}
""",
    "rag/clients/__init__.py": "",
    "rag/clients/voyage.py": """
class VoyageEmbedder:
    def __init__(self, *a, **kw):
        self.model = "voyage-4-lite"
        self.usage = {"total_tokens": 0}
class VoyageReranker:
    def __init__(self, *a, **kw):
        self.model = "rerank-2.5-lite"
        self.usage = {"total_tokens": 0}
""",
}


@pytest.fixture
def rag_stub(tmp_path, monkeypatch):
    """A minimal stand-in for the sibling checkout, wired in via TECH_DOCS_RAG_PATH."""
    for rel, src in RAG_STUB.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(src, encoding="utf-8")
    monkeypatch.setenv("TECH_DOCS_RAG_PATH", str(tmp_path))
    for name in [m for m in sys.modules if m == "rag" or m.startswith("rag.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_rag_root_prefers_the_env_override_over_the_sibling_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TECH_DOCS_RAG_PATH", str(tmp_path / "elsewhere"))
    assert retrieve._rag_root() == tmp_path / "elsewhere"
    monkeypatch.delenv("TECH_DOCS_RAG_PATH")
    assert retrieve._rag_root().name == "tech-docs-rag"


def test_load_rag_binds_search_index_and_the_voyage_clients(rag_stub):
    search, voyage, index = retrieve._load_rag()
    # the exact module layout the adapter depends on: clients live under rag.clients
    assert callable(search.search) and callable(index.build)
    assert voyage.VoyageEmbedder().model == "voyage-4-lite"
    assert voyage.VoyageReranker().model == "rerank-2.5-lite"


def test_load_rag_explains_itself_when_the_sibling_checkout_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("TECH_DOCS_RAG_PATH", str(tmp_path / "nope"))
    with pytest.raises(SystemExit) as e:
        retrieve._load_rag()
    assert "tech-docs-rag" in str(e.value) and "TECH_DOCS_RAG_PATH" in str(e.value)


def test_rag_retriever_attributes_the_per_query_token_delta_per_model(rag_stub, tmp_path):
    db = tmp_path / "runbooks.db"
    db.write_bytes(b"")
    r = retrieve.RagRetriever(db_path=db, k=4)
    out = r.retrieve({"id": "INC-0001", "title": "5xx spike", "body": "checkout-api"})

    # chunks fold to ranked runbooks, first occurrence winning
    assert out.runbook_ids == ["RB-app-5xx", "RB-latency"]
    # each model is billed its own delta, under its own name
    assert out.usage == {"voyage-4-lite": {"total_tokens": 11},
                         "rerank-2.5-lite": {"total_tokens": 23}}
    import rag.search as stub_search
    call = stub_search.CALLS[-1]
    assert call["k"] == 4 and call["hybrid"] is True and call["reranker"] is not None
    assert "5xx spike" in call["query"] and "checkout-api" in call["query"]


def test_rag_retriever_without_rerank_bills_only_the_embedder(rag_stub, tmp_path):
    db = tmp_path / "runbooks.db"
    db.write_bytes(b"")
    r = retrieve.RagRetriever(db_path=db, rerank=False)
    out = r.retrieve({"id": "INC-0001", "title": "t", "body": "b"})
    assert r.reranker is None
    assert out.usage == {"voyage-4-lite": {"total_tokens": 11}}


def test_rag_retriever_refuses_to_run_without_a_built_index(rag_stub, tmp_path):
    with pytest.raises(SystemExit) as e:
        retrieve.RagRetriever(db_path=tmp_path / "missing.db")
    assert "index-runbooks" in str(e.value)


def test_index_runbooks_writes_chunk_rows_then_hands_them_to_the_index(rag_stub, tmp_path):
    db = tmp_path / "out" / "runbooks.db"
    stats = retrieve.index_runbooks(db_path=db)
    chunks = db.parent / "runbook_chunks.jsonl"
    rows = [json.loads(l) for l in chunks.read_text(encoding="utf-8").splitlines()]
    assert rows and {"id", "url", "source_url", "text"} <= set(rows[0])
    assert stats["runbooks"] == len({r["source_url"] for r in rows})
    import rag.index as stub_index
    assert stub_index.CALLS[-1]["chunks_path"] == chunks
    assert stub_index.CALLS[-1]["embedder"].model == "voyage-4-lite"
