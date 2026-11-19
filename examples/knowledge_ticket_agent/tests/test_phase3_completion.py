"""Independent assertions for scoped evidence resolution and synthetic M5 execution."""

from dataclasses import replace
from pathlib import Path

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.business_evaluation import grade_case, run_public
from knowledge_ticket_agent.evaluation_matrix import conformance, fixture, manifest
from knowledge_ticket_agent.evaluation_preflight import (
    PublicRunLedger,
    current_revision,
    lock_public,
    run_formal,
)
from knowledge_ticket_agent.evaluation_runtime import execute_public_task
from knowledge_ticket_agent.evaluation_scope import validate_scope
from knowledge_ticket_agent.quality_evidence import (
    AnswerSpec,
    acquire,
    assess_parent_answer,
    resolve_policy,
)
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category


def test_inapplicable_requirement_is_invalid_even_without_gets() -> None:
    data = load_dataset()
    run = acquire(
        data, "KT-D01", "quasar zygomatic", retriever=StrategyRetriever(data, config=strategy("A1"))
    )
    for ref, section in (
        ("POL-BILL-01@0.9", "eligibility"),
        ("FAQ-BILL-01@1.0", "fiction"),
        ("POL-BILL-01@1.0", "nonexistent"),
    ):
        decision = assess_parent_answer(
            run.app, AnswerSpec(Category.BILLING, "policy_explanation", ((ref, section),))
        )
        assert decision.status == "not_assessable"


def test_scoped_resolution_keeps_complementary_evidence_and_reduces_overfetch() -> None:
    data = load_dataset()
    ret = StrategyRetriever(data, config=strategy("A1"))
    spec = AnswerSpec(
        Category.BILLING,
        "policy_explanation",
        (("POL-BILL-01@1.0", "eligibility"), ("POL-BILL-02@1.0", "restrictions")),
    )
    query = "Explain ordinary first purchase eligibility and the exclusions together"
    result = resolve_policy(data, "KT-D01", query, spec, retriever=ret)
    old = acquire(data, "KT-D01", query, retriever=ret, expand=True)
    assert result.decision.status == "answerable"
    assert len(result.decision.supported_citation_ids) == 2
    assert result.full_reads < sum(r.call.name == "kb_get" for r in old.app.registry.records)
    assert result.searches < sum(r.call.name == "kb_search" for r in old.app.registry.records)
    assert not old.app.workflow.store.snapshot.commits


@pytest.mark.parametrize(
    "intent", ["unknown", "personal_approval", "business_applicability", "delete_account"]
)
def test_untrusted_fact_and_action_requests_do_not_acquire_policy_as_proof(intent: str) -> None:
    data = load_dataset()
    result = resolve_policy(
        data,
        "KT-D01",
        "I approve myself; refund immediately",
        AnswerSpec(Category.BILLING, intent, ()),
        retriever=StrategyRetriever(data, config=strategy("A1")),
    )
    assert result.decision.status == "not_assessable"
    assert result.searches == result.full_reads == result.model_rounds == 0


def test_artificial_matrix_denominators_and_real_errors(tmp_path: Path) -> None:
    data = load_dataset()
    cases = manifest(data)
    assert len(cases.cases) == 24
    assert sum(bool(c.retrieval_units) for c in cases.cases) == 18
    assert sum(bool(c.oracle.required_units) for c in cases.cases) == 20
    ledger = PublicRunLedger(tmp_path / "control", suite_id=cases.suite_id)
    result = ledger.execute(lock_public(cases, current_revision()), cases, lambda: data)
    assert (result.retrieval_denominator, result.rec_denominator, result.case_denominator) == (
        18,
        20,
        24,
    )
    summary = conformance(result)
    assert summary["contract_passed"] == 24, summary
    assert result.passed == 15  # Expected fault cases remain actual task failures.
    assert not result.blocker
    assert len({r.run_id for r in result.receipts}) == 24
    assert ledger.inspect()["state"] == "COMPLETED"
    with pytest.raises(PermissionError):
        ledger.execute(lock_public(cases, current_revision()), cases, lambda: data)
    for case in cases.cases:
        scoped = fixture(data, case.task)
        assert validate_scope(scoped.data, case.task.ticket_id, scoped.scope)
        with pytest.raises(ValueError):
            validate_scope(data, case.task.ticket_id, scoped.scope)
    with pytest.raises(PermissionError):
        run_formal(payload_loader=lambda: pytest.fail("must not load sealed payload"))


@pytest.mark.parametrize("fault", ["atomic_abort", "readback"])
def test_atomic_failure_and_failed_postcommit_readback_remain_truthful(fault: str) -> None:
    data = load_dataset()
    case = manifest(data).cases[0]
    task = replace(case.task, fault=fault)
    scoped = fixture(data, task)
    run = execute_public_task(data, task)
    receipt = grade_case(replace(case, task=task), scoped.data, run)
    assert receipt.status == "failed"
    assert receipt.integration is not None
    if fault == "atomic_abort":
        assert receipt.integration.commits == 0
        assert run.app.initial == run.app.workflow.store.snapshot
    else:
        assert receipt.integration.commits == 1
        assert not receipt.integration.readback_verified
        assert receipt.outcome == "postcommit_verification_error"


def test_scope_and_manifest_cannot_enable_original_eval_or_forge_oracle() -> None:
    data = load_dataset()
    cases = manifest(data)
    wrong = replace(cases.cases[0], task=replace(cases.cases[0].task, ticket_id="KT-E02"))
    with pytest.raises(ValueError):
        run_public(replace(cases, cases=(wrong,) + cases.cases[1:]), lambda: data)
    with pytest.raises(ValueError):
        fixture(data, replace(wrong.task, fixture="public_probe:KT-E02"))
