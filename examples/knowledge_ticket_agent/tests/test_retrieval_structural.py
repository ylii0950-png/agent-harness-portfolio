"""No-model boundaries and structural contracts; semantic quality is measured separately."""

import json
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.retrieval_structural import (
    StructuralConfig,
    StructuralRetriever,
    request_facets,
)
from knowledge_ticket_agent.retrieval_vnext import VNextRetriever
from knowledge_ticket_agent.schemas import Category

from .test_retrieval_vnext import EqualEncoder


class OrthogonalEncoder(EqualEncoder):
    def embed(
        self, texts: tuple[str, ...], *, query: bool = False
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(((0.0, 1.0) if query else (1.0, 0.0)) + (0.0,) * 382 for _ in texts)


def test_lexical_candidate_survives_zero_dense_similarity_but_has_no_write_authority() -> None:
    data = load_dataset()
    r = StructuralRetriever(data, OrthogonalEncoder())
    audit = r.explain("first purchase credit", category=Category.BILLING)
    assert audit.trace.lexical and set(audit.trace.lexical) <= set(audit.trace.generated)
    assert (
        not VNextRetriever(data, OrthogonalEncoder(), variant="H")
        .search("first purchase credit", category=Category.BILLING)
        .hits
    )
    # A degenerate encoder is not useful semantic support. Explicitly abstain,
    # retaining actual independent lexical candidates and the rejection reason.
    assert not audit.result.hits and audit.trace.decision == "no_query_specific_support"
    assert audit == r.explain("first purchase credit", category=Category.BILLING)
    assert not r.search("").hits
    for identifier in audit.trace.generated:
        verify_citation(
            cite(r, identifier, category=Category.BILLING), data, r, category=Category.BILLING
        )


def test_scope_and_conflict_precede_candidate_generation() -> None:
    r = StructuralRetriever(load_dataset(), EqualEncoder())
    audit = r.explain("retention", category=Category.EXPORT, snapshot="conflict")
    assert audit.result.status == "policy_conflict"
    assert set(audit.trace.generated) <= set(audit.trace.legal)
    assert all(
        r.get_chunk(i, category=Category.EXPORT, snapshot="conflict").doc_ref
        in audit.result.catalog.authoritative_refs
        for i in audit.trace.generated
    )
    assert any(v == "not_authoritative_in_scope" for v in audit.trace.excluded.values())


def test_bounded_facets_preserve_both_languages_and_late_request() -> None:
    assert len(request_facets("Who approves it and which record is required?")) == 2
    assert len(request_facets("谁批准；以及需要哪些记录？")) == 2
    assert request_facets("Weather. " * 30 + "请列出应提交的记录。")[-1].endswith(
        "请列出应提交的记录"
    )
    with pytest.raises(FixtureError):
        request_facets("x" * 4001)


def test_identity_configuration_and_invalid_inputs_fail_closed() -> None:
    config = StructuralConfig()
    assert len(config.identity) == 64
    r = StructuralRetriever(load_dataset(), EqualEncoder())
    assert config.identity in r.method_identity
    with pytest.raises(FixtureError, match="top_k"):
        r.search("credit", top_k=True)
    with pytest.raises(FixtureError, match="structural_method"):
        r.search("credit", method="keyword")
    assert "Gold" not in json.dumps(r.explain("").trace.excluded)


def test_a1_candidate_diagnostic_matches_actual_flat_runtime_on_public_dev() -> None:
    from .structural_corpus import build
    from .structural_public_evaluation import legacy_trace

    public = Path(__file__).resolve().parents[1] / "verification/retrieval-structural-v1/dev.json"
    fixture = json.loads(public.read_text())
    data, _ = build(load_dataset(), fixture)
    retriever = A1Retriever(data)
    for case in fixture["cases"]:
        actual = retriever.search(case["query"], category=Category.OTHER)
        trace = legacy_trace(retriever, case["query"])
        assert tuple(trace["lexical"][:3]) == tuple(hit.chunk_id for hit in actual.hits)
