"""Stage: retrieve — surface the runbook(s) for an incident (design §4.2).

Runbook search is not built here. This stage **reuses the `rag` package**
from [tech-docs-rag](https://github.com/Hal-Hanami/tech-docs-rag) — hybrid dense
+ BM25 retrieval fused by RRF, then a Voyage cross-encoder rerank — over the
synthetic runbook corpus (`triage/runbooks.py`). None of that retrieval logic is
reimplemented here (design §11); `RagRetriever` is a thin adapter that builds the
query, calls `rag.search.search`, and folds the ranked chunks back to ranked
*runbooks* (dedup, first occurrence wins). The top runbook's score is the
retrieval confidence the decider gates on (design §6).

Same seam shape as the classify stage: a `Retriever` Protocol so the eval loop
runs offline with a fake (no `rag`, no Voyage, no key), while `RagRetriever`
plugs the real engine in for live measurement. `rag` is imported lazily from the
sibling tech-docs-rag checkout (default `../tech-docs-rag`, override with
`TECH_DOCS_RAG_PATH`), so the offline core and test suite never need it installed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import observe, runbooks
from .observe import Trace

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "runbooks.db"
DEFAULT_K = 5  # chunks fetched per incident; deduped to runbooks for recall ranking


@dataclass
class RunbookHit:
    """One retrieved runbook: its id + the best chunk that surfaced it. `score` is
    the unified retrieval score (rerank score when reranked) used as the confidence
    signal; `section_path` / `url` / `text` carry the citable section (design §5)."""

    runbook_id: str
    score: float
    section_path: str = ""
    url: str = ""
    text: str = ""
    chunk_id: str = ""


@dataclass
class Retrieval:
    """The retrieve stage's output for one incident: runbooks ranked best-first
    (deduped from the chunk hits), the raw top-k chunks for the drafter, and the
    per-model Voyage token usage for the cost ledger (design §7). A fake retriever
    returns empty `usage`."""

    runbooks: list[RunbookHit]
    chunks: list[dict] = field(default_factory=list)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def runbook_ids(self) -> list[str]:
        """Retrieved runbook ids, rank order — what recall@k scores (design §10)."""
        return [h.runbook_id for h in self.runbooks]

    @property
    def top_score(self) -> float:
        """Top runbook's retrieval score — the decider's confidence signal (§6)."""
        return self.runbooks[0].score if self.runbooks else 0.0


class Retriever(Protocol):
    """LLM/embedding seam — `retrieve(view) -> Retrieval`. Faked offline in tests;
    the real `RagRetriever` (tech-docs-rag's `rag`) plugs in here."""

    def retrieve(self, view: dict[str, str]) -> Retrieval: ...


def build_query(view: dict[str, str]) -> str:
    """The retrieval query from an incident's prompt view (title + body only — same
    no-label-leak contract as classify). Type-conditioning on the predicted type is a
    possible lever (design §4.2); what is measured here is raw text-to-runbook recall."""
    return f"{view['title']}\n\n{view['body']}"


def build_retrieval(chunks: list[dict[str, Any]],
                    usage: dict[str, dict[str, int]] | None = None) -> Retrieval:
    """Fold ranked chunk results into ranked runbook hits (dedup, first occurrence
    wins — a runbook's rank is its best chunk's rank)."""
    hits: list[RunbookHit] = []
    seen: set[str] = set()
    for c in chunks:
        rb = c["source_url"]
        if rb in seen:
            continue
        seen.add(rb)
        hits.append(RunbookHit(
            runbook_id=rb,
            score=float(c.get("score", 0.0)),
            section_path=c.get("section_path", ""),
            url=c.get("url", ""),
            text=c.get("text", ""),
            chunk_id=c.get("chunk_id", ""),
        ))
    return Retrieval(runbooks=hits, chunks=list(chunks), usage=usage or {})


def retrieve_runbooks(view: dict[str, str], retriever: Retriever,
                      *, trace: Trace | None = None) -> Retrieval:
    """Run the retrieve stage for one incident's prompt view.

    Mirrors `classify.classify_incident`: `trace`, if given, times the stage and
    files Voyage's embed/rerank tokens under their model names (design §7)."""
    with observe.span(trace, "retrieve"):
        result = retriever.retrieve(view)
    if trace is not None:
        for model, usage in result.usage.items():
            trace.add_usage(model, usage)
    return result


# --- reuse: tech-docs-rag's `rag` package ------------------------------------

def _rag_root() -> Path:
    """Where the sibling tech-docs-rag checkout lives (for `import rag`)."""
    env = os.environ.get("TECH_DOCS_RAG_PATH")
    return Path(env).expanduser() if env else ROOT.parent / "tech-docs-rag"


def _load_rag():
    """Lazy-import `rag` (search / index / the Voyage clients) off the sibling
    tech-docs-rag checkout. Only the live retrieve/index paths call this — the
    offline core and test suite never import `rag` or its `sqlite-vec` dependency."""
    root = _rag_root()
    if not (root / "rag" / "__init__.py").exists():
        raise SystemExit(
            f"tech-docs-rag's `rag` package not found at {root}.\n"
            "  Clone https://github.com/Hal-Hanami/tech-docs-rag next to this repo, "
            "or set TECH_DOCS_RAG_PATH to its path."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        # The Voyage embedder and reranker share one HTTP/usage implementation and
        # live together in `rag.clients.voyage`; `search`/`index` take them as
        # arguments, so this module never reimplements either.
        from rag import index, search  # noqa: PLC0415  (lazy by design)
        from rag.clients import voyage  # noqa: PLC0415
    except ModuleNotFoundError as e:  # almost always the sqlite-vec runtime dep
        raise SystemExit(
            f"could not import tech-docs-rag's `rag` ({e}). Its deps are missing — run via:\n"
            "  uv run --with sqlite-vec [--with anthropic] python -m triage ..."
        ) from e
    return search, voyage, index


class RagRetriever:
    """Runbook retrieval over tech-docs-rag's `rag`: hybrid (dense+BM25) + Voyage rerank
    against the prebuilt runbook index (design §2/§4.2). Holds the Voyage embedder +
    reranker so each call can snapshot their cumulative token usage and attribute the
    per-query delta per model (the same pattern as tech-docs-rag's `rag.eval`)."""

    def __init__(self, db_path: Path = DEFAULT_DB, *, k: int = DEFAULT_K,
                 hybrid: bool = True, rerank: bool = True) -> None:
        search, voyage, _ = _load_rag()
        self._search = search
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise SystemExit(
                f"runbook index not found at {self.db_path}.\n"
                "  Build it first:  uv run --with sqlite-vec python -m triage index-runbooks"
            )
        self.k = k
        self.hybrid = hybrid
        self.embedder = voyage.VoyageEmbedder()          # reads VOYAGE_API_KEY
        self.reranker = voyage.VoyageReranker() if rerank else None

    def retrieve(self, view: dict[str, str]) -> Retrieval:
        query = build_query(view)
        embed_before = self.embedder.usage["total_tokens"]
        rerank_before = self.reranker.usage["total_tokens"] if self.reranker else 0

        results = self._search.search(query, self.db_path, self.embedder, k=self.k,
                                      hybrid=self.hybrid, reranker=self.reranker)

        usage: dict[str, dict[str, int]] = {}
        d_embed = self.embedder.usage["total_tokens"] - embed_before
        if d_embed:
            observe.merge_usage(usage, self.embedder.model, {"total_tokens": d_embed})
        if self.reranker:
            d_rerank = self.reranker.usage["total_tokens"] - rerank_before
            if d_rerank:
                observe.merge_usage(usage, self.reranker.model, {"total_tokens": d_rerank})
        return build_retrieval(results, usage)


def index_runbooks(*, runbook_dir: Path = runbooks.RUNBOOK_DIR,
                   db_path: Path = DEFAULT_DB,
                   chunks_path: Path | None = None) -> dict[str, Any]:
    """Chunk the runbook corpus and (re)build the `rag` vector+BM25 index over it.

    Writes the chunk rows to a JSONL (mirroring tech-docs-rag's chunks.jsonl -> index flow)
    then hands them to `rag.index.build` with a Voyage embedder. Needs VOYAGE_API_KEY.
    The index/jsonl live under `data/` (gitignored) — rebuilt, never committed."""
    _, voyage, index = _load_rag()

    rows = runbooks.build_runbook_chunks(runbook_dir)
    chunks_path = chunks_path or (db_path.parent / "runbook_chunks.jsonl")
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = index.build(chunks_path, db_path, voyage.VoyageEmbedder())
    stats["runbooks"] = len({r["source_url"] for r in rows})
    return stats
