"""No-model safety and stage tests; stubs do not establish semantic quality."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.business_demo import run_ticket, summarize
from knowledge_ticket_agent.citations import cite
from knowledge_ticket_agent.evidence_contract import EvidenceVerdict, Verdict
from knowledge_ticket_agent.pair_model import pair_identity, token_windows, verify_pair_model
from knowledge_ticket_agent.relevance_evidence import verify_business_sources, verify_support
from knowledge_ticket_agent.retrieval import lexical_rank
from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever, plan_query
from knowledge_ticket_agent.schemas import Category

from .test_retrieval_structural import OrthogonalEncoder
from .test_retrieval_vnext import EqualEncoder


class FakePair:
    identity = "2" * 64

    def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]:
        return tuple(
            5.0 if "No supporting" not in text and "do not specify" not in text else -5.0
            for _, text in pairs
        )


def test_read_only_documentary_subset_does_not_require_business_transition_sources() -> None:
    from knowledge_ticket_agent.quality_evidence import AnswerSpec, resolve_policy

    data = load_dataset()
    result = resolve_policy(
        data,
        "KT-D01",
        "first purchase credit eligibility",
        AnswerSpec(Category.BILLING, "policy_explanation", (("POL-BILL-01@1.0", "eligibility"),)),
        retriever=RelevanceRetriever(data, EqualEncoder(), pair=FakePair()),
    )
    assert result.decision.status == "answerable"
    assert result.commits == result.approval_calls == 0 and result.unchanged_state_verified


def test_request_plan_keeps_middle_negations_conditions_and_all_facets() -> None:
    q = "Who approves? When not eligible? Which record? Where to send? How long?"
    plan = plan_query(q)
    assert len(plan.facets) == 5 and "not eligible" in plan.facets[1]
    assert plan_query("谁批准？如果未完成，不得执行；需要什么记录？").facets == (
        "谁批准?",
        "如果未完成,不得执行",
        "需要什么记录?",
    )
    assert len(plan_query("A and B must not proceed unless C and D agree").facets) == 1
    with pytest.raises(FixtureError, match="request_plan_capacity"):
        plan_query(" ".join(f"Who approves {i}?" for i in range(17)))
    assert not plan_query(" ！？ ").facets


def test_pair_windows_preserve_all_tokens_and_do_not_duplicate_the_tail() -> None:
    for length in (0, 252, 253, 1000):
        tokens = list(range(length))
        windows = token_windows(tokens, 252)
        assert set(t for w in windows for t in w) == set(tokens)
        assert all(len(w) <= 252 for w in windows)
        if len(windows) > 1:
            assert not set(windows[-1]) <= set(windows[-2])
    assert len(pair_identity()) == 64
    with pytest.raises(FixtureError):
        verify_pair_model(Path("missing-explicit-pair-assets"))


@pytest.mark.parametrize("variant", ["R0", "R1"])
def test_independent_candidates_and_ranked_hits_do_not_assert_answerability(variant: str) -> None:
    data = load_dataset()
    r = RelevanceRetriever(
        data, OrthogonalEncoder(), variant=variant, pair=FakePair() if variant == "R1" else None
    )
    audit = r.explain("first purchase credit", category=Category.BILLING)
    assert set(audit.trace.lexical) <= set(audit.trace.generated)
    assert audit.result.hits and audit.trace.decision == "relevance_only_not_answerability"
    assert audit.result == r.search("first purchase credit", category=Category.BILLING)
    assert audit.result.method == r.method_identity
    assert len(audit.result.method) <= 256
    assert all(h.score > 0 for h in audit.result.hits)
    assert not r.search("").hits
    assert (
        r.search("export", category=Category.EXPORT, snapshot="conflict").status
        == "policy_conflict"
    )
    for h in audit.result.hits:
        assert h.doc_ref in audit.result.catalog.authoritative_refs
    with pytest.raises(FixtureError):
        r.search("credit", top_k=True)


def test_pair_failure_is_explicit_without_dense_or_lexical_fallback() -> None:
    class Broken(FakePair):
        def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]:
            return (math.nan,) * len(pairs)

    with pytest.raises(FixtureError, match="pair_scores"):
        RelevanceRetriever(load_dataset(), EqualEncoder(), pair=Broken()).search("credit")
    with pytest.raises(FixtureError, match="explicit_pair_backend"):
        RelevanceRetriever(load_dataset(), EqualEncoder())


def test_full_canonical_evidence_only_and_no_entailment_or_authorization_claim() -> None:
    data = load_dataset()
    r = RelevanceRetriever(data, EqualEncoder(), pair=FakePair())
    hit = r.search("first purchase", category=Category.BILLING).hits[0]
    citation = cite(r, hit.chunk_id, category=Category.BILLING)
    verdict = verify_support(data, r, "Who approves?", (citation,), category=Category.BILLING)
    assert verdict.status is Verdict.NOT_ASSESSABLE
    assert verify_support(data, r, "Who approves?", ()).status is Verdict.NOT_ASSESSABLE
    with pytest.raises(FixtureError):
        verify_support(data, r, "Who approves?", (replace(citation, quote="forged"),))


@pytest.mark.parametrize("approval", [False, True])
def test_actual_agent_keeps_exact_ask_atomic_journal_and_readback(approval: bool) -> None:
    data = load_dataset()

    class LexicalPair(FakePair):
        def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]:
            return tuple(
                dict(lexical_rank((("doc", passage),), query[:256])).get("doc", 0.0)
                for query, passage in pairs
            )

    r = RelevanceRetriever(data, EqualEncoder(), pair=LexicalPair())
    run = run_ticket(data, "KT-D01", approval="approve" if approval else "refuse", retriever=r)
    result = summarize(run, "relevance-control")
    assert result["verification"] == "PASS"
    assert result["commits"] == result["handler_entries"] == int(approval)


def test_actual_conflict_verdict_blocks_answer_assessment_and_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Source opposition itself is tested above and on canonical public fixtures.
    # Inject only the detector's veto here: frozen M3 policy bytes cannot be changed
    # to create a business test without first violating its existing source lock.
    data = load_dataset()

    def denied(*args: object, **kwargs: object) -> EvidenceVerdict:
        # The injected outcome exercises the real assessment/write gate. Semantic
        # claim opposition is independently tested by the domain contract suite.
        result = verify_business_sources(
            data,
            RelevanceRetriever(data, EqualEncoder(), pair=FakePair()),
            (),
            (("POL-BILL-01@1.0", "eligibility"),),
            category=Category.BILLING,
        )
        return replace(result, status=Verdict.NOT_ASSESSABLE, unknown=("synthetic_conflict",))

    monkeypatch.setattr("knowledge_ticket_agent.relevance_evidence.verify_business_sources", denied)

    class LexicalPair(FakePair):
        def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]:
            return tuple(
                dict(lexical_rank((("doc", p),), q[:256])).get("doc", 0.0) for q, p in pairs
            )

    run = run_ticket(
        data,
        "KT-D01",
        approval="approve",
        retriever=RelevanceRetriever(data, EqualEncoder(), pair=LexicalPair()),
    )
    report = summarize(run, "directive-conflict")
    assert report["commits"] == report["handler_entries"] == 0
    assert report["before"] == report["after"]
    assert report["evidence"]["status"] == "not_assessable"
    assert any(
        "source_directive_conflict" in r.result.content
        for r in run.app.registry.records
        if r.call.name == "ticket_assess"
    )


def test_pair_assets_symlinks_and_modified_bytes_fail_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knowledge_ticket_agent.pair_model import prepare_pair_model

    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FixtureError, match="model_symlink"):
        verify_pair_model(link)
    (tmp_path / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(FixtureError, match="semantic_model_size"):
        verify_pair_model(tmp_path)
    monkeypatch.setenv("CI", "true")
    with pytest.raises(FixtureError, match="model_download_disabled_in_ci"):
        prepare_pair_model(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    # Existing invalid assets are never overwritten or downloaded over.
    with pytest.raises(FixtureError, match="semantic_model_size"):
        prepare_pair_model(tmp_path)
    assert (tmp_path / "model.onnx").read_bytes() == b"tampered"


def test_prospective_navigation_batches_only_independent_reads_without_raising_limits() -> None:
    from knowledge_ticket_agent.quality_evidence import EvidenceNavigator

    targets = tuple(((f"source-{i}",), f"question-{i}") for i in range(5))
    navigator = EvidenceNavigator("initial", targets, batch_followups=3)
    assert len(navigator.step(()).tool_calls) == 1
    batch = navigator.step(())
    assert len(batch.tool_calls) == 3 and {c.name for c in batch.tool_calls} == {"kb_search"}
    assert len(navigator.step(()).tool_calls) == 2
    assert not navigator.step(()).tool_calls
    with pytest.raises(FixtureError, match="navigation_batch"):
        EvidenceNavigator("initial", targets, batch_followups=True)
