"""Check the stand-in `rag` package against the real one, when it is next door.

`tests/test_retrieve.py` drives the adapter through a hand-written stub of the
sibling `tech-docs-rag` package. That keeps the suite offline, but it also means
the suite cannot notice when the real package moves: the stub answers to whatever
the adapter asks, so a rename over there leaves every test green and breaks only a
paid run. That is not hypothetical — the Voyage clients moved into
`rag.clients.voyage`, the stub was written to match the adapter rather than the
package, and the live retrieval path was broken while the suite reported success.

So this file asks the real package the questions the stub cannot: do the calls in
`triage/retrieve.py` still bind, and does the stub promise anything the package no
longer provides. It runs only when the checkout is actually present — CI does not
clone the sibling, so there it skips — which makes it a local guard rather than a
gate. The gate lives on the other side, in that repository's `test_public_api.py`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tests.test_retrieve import RAG_STUB
from triage import retrieve

RAG_ROOT = retrieve._rag_root()

pytestmark = pytest.mark.skipif(
    not (RAG_ROOT / "rag" / "__init__.py").exists(),
    reason=f"no tech-docs-rag checkout at {RAG_ROOT} (set TECH_DOCS_RAG_PATH to run)",
)


@pytest.fixture(scope="module")
def real():
    """The real `rag`, imported off the sibling checkout rather than the stub."""
    try:
        return retrieve._load_rag()  # -> (search, voyage, index)
    except SystemExit as e:  # sqlite-vec absent: its own runtime dependency
        pytest.skip(f"tech-docs-rag is present but not importable here: {e}")


def test_the_real_package_exposes_the_modules_the_adapter_imports(real):
    search, voyage, index = real
    assert callable(search.search) and callable(index.build)
    assert hasattr(voyage, "VoyageEmbedder") and hasattr(voyage, "VoyageReranker")


def test_the_real_search_accepts_the_call_this_repo_makes(real):
    search, _, _ = real
    # RagRetriever.retrieve
    inspect.signature(search.search).bind(
        "query", Path("runbooks.db"), object(), k=4, hybrid=True, reranker=None)


def test_the_real_index_build_accepts_the_call_this_repo_makes(real):
    _, _, index = real
    # index_runbooks
    inspect.signature(index.build).bind(Path("chunks.jsonl"), Path("runbooks.db"), object())


def test_the_real_voyage_clients_expose_the_model_and_meter_we_bill_from(real):
    _, voyage, _ = real
    embedder = voyage.VoyageEmbedder(api_key="not-a-real-key")
    reranker = voyage.VoyageReranker(api_key="not-a-real-key")
    # the per-model cost split in `RagRetriever.retrieve` reads exactly these two
    assert embedder.model and embedder.usage["total_tokens"] == 0
    assert reranker.model and reranker.usage["total_tokens"] == 0


def test_the_stub_promises_nothing_the_real_package_has_dropped(real):
    """The stub is only trustworthy while it is a subset of the real thing."""
    search, voyage, index = real
    modules = {"rag/search.py": search, "rag/index.py": index,
               "rag/clients/voyage.py": voyage}
    for rel, module in modules.items():
        for name in _public_names(RAG_STUB[rel]):
            assert hasattr(module, name), (
                f"the stub in tests/test_retrieve.py defines {rel}:{name}, which the real "
                f"package no longer has — the stub is now lying to every test that uses it"
            )


def _public_names(source: str) -> list[str]:
    """Top-level `def`/`class` names declared in a stub module's source."""
    import ast
    return [n.name for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
