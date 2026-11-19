"""No-model default tests of source boundaries, explicit errors and real tool flow."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.business_demo import run_request, run_ticket, summarize
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.evaluation_runtime import CaseInput, execute_public_task
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.retrieval_vnext import VNextRetriever, segments
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.semantic_model import _windows, model_identity, verify_model


class EqualEncoder:
    """A control stub for contract tests; never semantic-quality evidence."""

    identity = "1" * 64

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def embed(
        self, texts: tuple[str, ...], *, query: bool = False
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append((texts, query))
        return tuple((1.0, *(0.0 for _ in range(383))) for _ in texts)


def test_segments_retain_late_intent_and_cjk_without_prefix_truncation() -> None:
    query = "Unrelated weather information. " * 60 + "请解释首次购买额度的申请条件"
    parts = segments(query)
    assert any("首次购买" in p for p in parts)
    assert all(len(p) <= 256 for p in parts)
    assert segments(" ?! ") == ()
    assert len(segments("x" * 4000)) <= 32
    for bad in ("x" * 4001, "bad\x00query"):
        with pytest.raises(FixtureError):
            segments(bad)


@pytest.mark.parametrize("length", [0, 470, 490, 508, 984, 1300])
def test_encoder_windows_cover_tokens_without_a_redundant_tail(length: int) -> None:
    body = list(range(10, 10 + length))
    windows = _windows([0, 4, 5, *body, 2], [4, 5])
    assert all(len(w) <= 512 for w in windows)
    observed = [t for window in windows for t in window[3:-1]]
    assert set(observed) == set(body)
    if length <= 508:
        assert len(windows) == 1
    else:
        assert not set(windows[-1][3:-1]) <= set(windows[-2][3:-1])


def test_encoder_prefix_never_replaces_the_first_body_token() -> None:
    complete = [0, 41, 1294, 12, 6226, 100, 2]
    assert _windows(complete, [41, 1294, 12]) == (tuple(complete),)
    with pytest.raises(FixtureError, match="semantic_token_prefix"):
        _windows(complete, [41, 1294, 12, 6])


@pytest.mark.parametrize("variant", ["D", "H"])
def test_filtered_verified_deterministic_sources_and_versioned_method(variant: str) -> None:
    data, encoder = load_dataset(), EqualEncoder()
    r = VNextRetriever(data, encoder, variant=variant)
    a1 = A1Retriever(data).search("first purchase")
    result = r.search("first purchase", category=Category.BILLING, top_k=5)
    assert len(result.hits) <= 3 and result.method == r.method_identity
    assert r.search("first purchase", category=Category.BILLING) == result
    assert a1 == A1Retriever(data).search("first purchase")
    assert r.search(" ").status == "no_match"
    for h in result.hits:
        assert h.doc_ref in result.catalog.authoritative_refs
        verify_citation(
            cite(r, h.chunk_id, category=Category.BILLING), data, r, category=Category.BILLING
        )
    assert (
        r.search("retention", category=Category.EXPORT, snapshot="conflict").status
        == "policy_conflict"
    )
    assert all("FAQ" not in text for texts, query in encoder.calls if not query for text in texts)
    assert r.config.identity in result.method and encoder.identity in result.method


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0])
def test_invalid_model_vectors_fail_closed_without_lexical_fallback(bad: float) -> None:
    class Broken(EqualEncoder):
        def embed(
            self, texts: tuple[str, ...], *, query: bool = False
        ) -> tuple[tuple[float, ...], ...]:
            return tuple(tuple(bad for _ in range(384)) for _ in texts)

    with pytest.raises(FixtureError, match="embedding"):
        VNextRetriever(load_dataset(), Broken()).search("first purchase")


def test_dimension_and_input_errors_are_explicit() -> None:
    class Broken(EqualEncoder):
        def embed(
            self, texts: tuple[str, ...], *, query: bool = False
        ) -> tuple[tuple[float, ...], ...]:
            return tuple((1.0,) for _ in texts)

    with pytest.raises(FixtureError, match="embedding_dimensions"):
        VNextRetriever(load_dataset(), Broken()).search("credit")
    r = VNextRetriever(load_dataset(), EqualEncoder())
    with pytest.raises(FixtureError, match="top_k"):
        r.search("credit", top_k=True)
    with pytest.raises(FixtureError, match="vnext_method"):
        r.search("credit", method="keyword")


def test_pinned_model_manifest_and_missing_model_never_download(tmp_path: Path) -> None:
    assert len(model_identity()) == 64
    with pytest.raises(FixtureError, match="semantic_model_unavailable"):
        verify_model(tmp_path / "missing")


@pytest.mark.parametrize("approval", ["approve", "refuse"])
def test_actual_agent_ask_atomic_commit_journal_readback(approval: str) -> None:
    data = load_dataset()
    run = run_ticket(
        data, "KT-D01", approval=approval, retriever=VNextRetriever(data, EqualEncoder())
    )
    report = summarize(run, "new-profile-control")
    assert report["verification"] == "PASS"
    assert report["commits"] == int(approval == "approve")
    assert report["handler_entries"] == int(approval == "approve")
    assert report["activity"]["initial_retriever"] == report["retrieval_method"]
    assert (
        report["postwrite_readback_verified"]
        if approval == "approve"
        else report["unchanged_state_verified"]
    )
    searches = [r for r in run.app.registry.records if r.call.name == "kb_search" and r.result.ok]
    assert searches and all(
        json.loads(r.result.content)["result"]["method"].startswith("lqd-semantic-hybrid-")
        for r in searches
    )
    assert any(r.call.name == "kb_get" and r.result.ok for r in run.app.registry.records)


@pytest.mark.parametrize(
    "text",
    [
        "请按首次购买的购买额度调整政策处理这张模拟工单。",
        "请依据 first purchase credit 处理模拟工单。",
    ],
)
def test_bilingual_entry_uses_selected_model_profile(text: str) -> None:
    data = load_dataset()
    run = run_request(
        data, "KT-D01", text, approval="refuse", retriever=VNextRetriever(data, EqualEncoder())
    )
    report = summarize(run, "mixed-control")
    assert report["verification"] == "PASS" and report["commits"] == 0
    assert report["retrieval_method"].startswith("lqd-semantic-hybrid-")
    assert report["request"]["input_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_new_public_task_query_reaches_first_real_search_and_old_sealed_route_rejects() -> None:
    data = load_dataset()
    task = CaseInput("KT-D01", "first purchase credit", "normal", "SYN-STAFF-02", "refuse")

    def factory(d: Dataset) -> VNextRetriever:
        return VNextRetriever(d, EqualEncoder())

    run = execute_public_task(data, task, retriever_factory=factory)
    search = next(r for r in run.app.registry.records if r.call.name == "kb_search")
    assert json.loads(search.call.arguments)["query"] == task.query
    assert run.app.workflow.store.snapshot == run.app.initial
    with pytest.raises(PermissionError):
        execute_public_task(
            data, replace(task, fixture="sealed:a1/normal"), retriever_factory=factory
        )
