"""The runbook corpus adapter — turn `fixtures/runbooks/*.md` into index rows.

This is the one piece of the retrieve stage this repo owns: chunking a small,
hand-structured local corpus into the row shape the `rag` index expects.
Retrieval itself (dense + BM25 + RRF + Voyage rerank) is reused wholesale from
`rag`, never reimplemented here (design §11) — chunking a private corpus is the
boundary that reuse leaves to the caller, exactly as tech-docs-rag's own `ingest`
package is separate from its `rag` package.

Each runbook is split at its H2 (`## …`) section boundaries: the H1 + "Applies
to" preamble becomes the first chunk, then one chunk per section (First response
/ Recommended first action / Escalate if). Section-level chunks give the
drafter citable units (design §5). Every row carries `source_url = <runbook id>`
(e.g. `RB-app-5xx`) — the key the retrieve stage dedups to and that the eval's
recall@k scores against (`Incident.expected_runbook`). Pure stdlib, no LLM, no
network: offline-testable without `rag`, Voyage, or a key.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNBOOK_DIR = ROOT / "fixtures" / "runbooks"

# Row keys consumed by rag.store.insert (via rag.index.build): id / url / source_url
# / page_title / section_path / anchor / text. We overload `source_url` to carry the
# runbook id (the corpus here is local files, not URLs) so retrieval results map 1:1
# back to a runbook for recall scoring; `url` keeps an openable file locator for
# citations.
_H1_RE = re.compile(r"^#\s+(.*\S)\s*$")
_H2_RE = re.compile(r"^##\s+(.*\S)\s*$")


def _slug(text: str) -> str:
    """Heading -> URL anchor slug (lowercase, spaces to dashes, punctuation dropped)."""
    t = re.sub(r"[^a-z0-9\s-]", "", text.strip().lower())
    return re.sub(r"-+", "-", re.sub(r"\s+", "-", t)).strip("-")


def chunk_runbook(path: Path) -> list[dict]:
    """Split one runbook markdown file into section-level index rows."""
    runbook_id = path.stem  # e.g. "RB-app-5xx"
    lines = path.read_text(encoding="utf-8").splitlines()

    page_title = runbook_id
    for ln in lines:
        m = _H1_RE.match(ln)
        if m:
            page_title = m.group(1).strip()
            break

    # Partition into the preamble (everything before the first H2) + one part per H2.
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    buf: list[str] = []
    for ln in lines:
        m = _H2_RE.match(ln)
        if m:
            sections.append((heading, buf))
            heading, buf = m.group(1).strip(), [ln]
        else:
            buf.append(ln)
    sections.append((heading, buf))

    rows: list[dict] = []
    for heading, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if not text:
            continue
        anchor = _slug(heading) if heading else ""
        section_path = page_title if heading is None else f"{page_title} > {heading}"
        url = f"fixtures/runbooks/{runbook_id}.md" + (f"#{anchor}" if anchor else "")
        rows.append({
            "id": f"{runbook_id}#{len(rows)}",
            "source_url": runbook_id,          # the recall key (Incident.expected_runbook)
            "url": url,                         # openable locator for citations (§5)
            "page_title": page_title,
            "section_path": section_path,
            "anchor": anchor,
            "text": text,
        })
    return rows


def build_runbook_chunks(runbook_dir: Path = RUNBOOK_DIR) -> list[dict]:
    """Chunk every `RB-*.md` runbook in `runbook_dir` into index rows."""
    rows: list[dict] = []
    for path in sorted(Path(runbook_dir).glob("RB-*.md")):
        rows.extend(chunk_runbook(path))
    return rows


def runbook_ids(runbook_dir: Path = RUNBOOK_DIR) -> list[str]:
    """The set of runbook ids present in the corpus (the answerable universe)."""
    return sorted(p.stem for p in Path(runbook_dir).glob("RB-*.md"))
