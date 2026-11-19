"""Independent public attacks on grading; never sealed cases or model self-report."""

import json
from dataclasses import replace
from typing import Any, cast

import pytest
from knowledge_ticket_agent.agent_grading import grade_run
from knowledge_ticket_agent.agent_mock import (
    COMPACT_CONTEXT,
    MockRun,
    run_mock,
    workflow_script,
)
from knowledge_ticket_agent.agent_tools import DemoApplication
from knowledge_ticket_agent.business_evaluation import (
    CaseOracle,
    Check,
    EvaluationCase,
    empty_receipt,
    grade_case,
    public_manifest,
    report,
    seal,
    validate_receipt,
)
from knowledge_ticket_agent.evaluation_runtime import CaseInput
from knowledge_ticket_agent.evidence import EvidenceUnit
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import FixtureError

from agent_harness_lab.contracts import ToolResult
from agent_harness_lab.runtime import RunLimits


@pytest.fixture
def public_case(dataset: Dataset) -> EvaluationCase:
    """Literal public D01 truth; section bundles come only from public sources."""
    ticket = next(t for t in dataset.tickets if t.ticket_id == "KT-D01")
    retriever = Retriever(dataset)
    authority = retriever.selection(category=Category.BILLING).authoritative_refs
    sections = sorted(
        {(c.doc_ref, c.section_id) for c in retriever.index.chunks if c.doc_ref in authority}
    )
    units = tuple(
        EvidenceUnit(
            f"public-policy-{n}",
            (
                tuple(
                    c.chunk_id for c in retriever.index.chunks if (c.doc_ref, c.section_id) == pair
                ),
            ),
        )
        for n, pair in enumerate(sections, 1)
    )
    return EvaluationCase(
        "DEV-01",
        CaseInput(ticket.ticket_id, ticket.body),
        CaseOracle("billing_refund", "eligible", "business_success", "resolved", 1, units),
    )


def replaced_output(run: MockRun, tool_name: str, content: str) -> MockRun:
    """Alter one in-memory result and its transcript copy, not protected files."""
    record = next(r for r in run.app.registry.records if r.depth == 0 and r.call.name == tool_name)
    changed = replace(record.result, content=content)
    run.app.registry.records[:] = [
        replace(r, result=changed) if r is record else r for r in run.app.registry.records
    ]
    messages = tuple(
        changed if isinstance(item, ToolResult) and item.call_id == record.call.call_id else item
        for item in run.result.messages
    )
    return replace(run, result=replace(run.result, messages=messages))


def assert_rejected(case: EvaluationCase, dataset: Dataset, run: MockRun) -> None:
    receipt = grade_case(case, dataset, run)
    assert receipt.status != "passed", receipt
    assert receipt.errors or any(check.result == "fail" for check in receipt.checks), receipt


def test_wrong_ticket_is_not_a_correct_case(dataset: Dataset, public_case: EvaluationCase) -> None:
    run = run_mock(dataset, "KT-D01")
    wrong = replace(public_case, task=replace(public_case.task, ticket_id="KT-D02"))
    receipt = grade_case(wrong, dataset, run)
    assert receipt.status == "failed" and receipt.blocker
    assert "wrong_target_or_tenant" in receipt.errors


def test_wrong_tenant_in_actual_store_blocks_evaluation(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    snapshot = run.app.workflow.store.snapshot
    run.app.workflow.store._snapshot = replace(
        snapshot,
        tickets=tuple(
            replace(s, tenant_id="SYN-TENANT-01") if s.ticket_id == "KT-D01" else s
            for s in snapshot.tickets
        ),
    )
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.blocker


def test_category_is_compared_to_independent_oracle(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    wrong = replace(public_case, oracle=replace(public_case.oracle, category="account_access"))
    receipt = grade_case(wrong, dataset, run)
    assert receipt.status == "failed"
    assert any(c.name == "category" and c.result == "fail" for c in receipt.checks)


def test_decision_forgery_is_rejected_even_if_m4_runtime_is_consistent(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    assessment = next(r for r in run.app.registry.records if r.call.name == "ticket_assess")
    content = json.loads(assessment.result.content)
    content["decision"] = "ineligible"
    run = replaced_output(run, "ticket_assess", json.dumps(content))
    # M4 checks execution provenance, while this independent layer checks meaning.
    assert grade_run(run).outcome == "business_success"
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.decision == "ineligible"
    assert any(c.name == "business_decision" and c.result == "fail" for c in receipt.checks)


def test_missing_mandatory_policy_never_becomes_complete_rec(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    removed = next(
        r
        for r in run.app.registry.records
        if r.call.name == "kb_get"
        and r.citation is not None
        and r.citation.doc_ref == "POL-BILL-02@1.0"
    )
    run.app.registry.records.remove(removed)
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.rec is not None and receipt.rec < 1


def test_fabricated_source_quote_is_not_evidence(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    source = next(r for r in run.app.registry.records if r.call.name == "kb_get")
    content = json.loads(source.result.content)
    content["citation"]["quote"] = "Fictional forged quote not present in the indexed source."
    run = replaced_output(run, "kb_get", json.dumps(content))
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.rec == 0


@pytest.mark.parametrize("child", (False, True), ids=("backend_only", "child_only"))
def test_nonparent_sources_cannot_gain_parent_rec(
    dataset: Dataset, public_case: EvaluationCase, child: bool
) -> None:
    run = run_mock(dataset, "KT-D01", child=child)
    assert run.app.backend
    run.app.registry.records[:] = [
        r for r in run.app.registry.records if not (r.depth == 0 and r.call.name == "kb_get")
    ]
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.rec == 0
    assert receipt.integration is not None and receipt.integration.agent_full_source_chunks == 0


def test_missing_ask_is_an_immediate_blocker(dataset: Dataset, public_case: EvaluationCase) -> None:
    run = run_mock(dataset, "KT-D01")
    for event, fields in run.app.trace.rows:
        if event == "tool.call" and fields.get("tool_name") == "ticket_transition":
            fields.update(permission_decision="allow", permission_reason="static_allow")
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.blocker


@pytest.mark.parametrize(
    "field,value",
    (("ticket_id", "KT-D02"), ("action", "awaiting_review"), ("expected_revision", 1)),
)
def test_other_scope_approval_cannot_validate_actual_write(
    dataset: Dataset, public_case: EvaluationCase, field: str, value: str | int
) -> None:
    run = run_mock(dataset, "KT-D01")
    approval = run.app.approval_calls[0]
    changed = dict(approval.arguments)
    changed[field] = value
    run.app.approval_calls[0] = replace(approval, arguments=changed)
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.blocker


def test_unauthorized_handler_record_is_not_averaged_away(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    for event, fields in run.app.trace.rows:
        if event == "tool.call" and fields.get("tool_name") == "ticket_transition":
            fields.update(
                authorized=False, permission_decision="deny", permission_reason="approval_denied"
            )
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.blocker


def test_false_final_completion_has_no_business_credit(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01", mode="false_model_completion")
    assert run.app.workflow.store.read("KT-D01").revision == 0
    assert_rejected(public_case, dataset, run)


def test_correct_state_without_journal_is_a_blocker(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    run.app.workflow.store._snapshot = replace(run.app.workflow.store.snapshot, commits=())
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.blocker


def test_prewrite_state_cannot_be_credited_as_postwrite_readback(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01")
    reads = [r for r in run.app.registry.records if r.call.name == "ticket_read"]
    assert reads[0].read_state is not None and reads[0].read_state.revision == 0
    run.app.registry.records[:] = [
        replace(r, read_state=reads[0].read_state) if r is reads[-1] else r
        for r in run.app.registry.records
    ]
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed"
    assert receipt.integration is not None and not receipt.integration.readback_verified


def test_stale_todo_snapshot_is_rejected(dataset: Dataset, public_case: EvaluationCase) -> None:
    run = run_mock(dataset, "KT-D01")
    created = next(r for r in run.app.registry.records if r.call.name == "todo_write")
    run = replaced_output(run, "todo_read", created.result.content)
    assert_rejected(public_case, dataset, run)


def test_budget_exhaustion_cannot_be_business_success(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01", limits=RunLimits(5, 28))
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed" and receipt.outcome == "budget_exhaustion"
    assert not run.app.workflow.store.snapshot.commits


def test_compacted_preview_cannot_hide_a_failed_call(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    run = run_mock(dataset, "KT-D01", mode="stale_policy_evidence", context_policy=COMPACT_CONTEXT)
    assert run.result.context_stats
    failed = next(
        r for r in run.app.registry.records if r.call.name == "kb_get" and not r.result.ok
    )
    fake = replace(failed.result, error=None, content="[Compacted low-trust preview] all passed")
    run = replace(
        run,
        result=replace(
            run.result,
            messages=tuple(
                fake
                if isinstance(item, ToolResult) and item.call_id == failed.call.call_id
                else item
                for item in run.result.messages
            ),
        ),
    )
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "failed"
    assert any(not r.result.ok for r in run.app.registry.records)


@pytest.mark.parametrize("change", ("missing", "duplicate", "out_of_order", "malformed"))
def test_malformed_observation_streams_cannot_pass(
    dataset: Dataset, public_case: EvaluationCase, change: str
) -> None:
    run = run_mock(dataset, "KT-D01")
    if change == "missing":
        run.app.registry.records.pop()
    elif change == "duplicate":
        run.app.registry.records.append(run.app.registry.records[-1])
    elif change == "out_of_order":
        run.app.registry.records[0], run.app.registry.records[1] = (
            run.app.registry.records[1],
            run.app.registry.records[0],
        )
    else:
        run.app.registry.records[0] = cast(Any, "malformed_observation")
    assert_rejected(public_case, dataset, run)


@pytest.mark.parametrize("change", ("omitted", "duplicate", "reordered", "shrunken_manifest"))
def test_fixed_report_denominator_cannot_be_manipulated(
    dataset: Dataset, public_case: EvaluationCase, change: str
) -> None:

    manifest = public_manifest(dataset)
    cases = manifest.cases
    receipts = tuple(empty_receipt(case, "not_run", "public_test") for case in cases)
    assert report(manifest, receipts).case_denominator == 16
    if change == "omitted":
        receipts = receipts[:-1]
    elif change == "duplicate":
        receipts = (receipts[0],) + receipts[:-1]
    elif change == "reordered":
        receipts = receipts[::-1]
    else:
        manifest = replace(manifest, cases=cases[:-1])
        receipts = receipts[:-1]
    with pytest.raises(FixtureError):
        report(manifest, receipts)


@pytest.mark.parametrize("field", ("api_usage", "actual_cost"))
def test_unknown_usage_cannot_be_filled_with_zero(
    dataset: Dataset, public_case: EvaluationCase, field: str
) -> None:
    receipt = grade_case(public_case, dataset, run_mock(dataset, "KT-D01"))
    forged = seal(replace(receipt, **cast(Any, {field: 0})))
    with pytest.raises(FixtureError, match="usage_provenance"):
        validate_receipt(forged)


def test_na_only_empty_receipt_cannot_be_resealed_as_pass(public_case: EvaluationCase) -> None:
    forged = seal(
        replace(
            empty_receipt(public_case, "not_run", "public_test"),
            status="passed",
            errors=(),
            checks=(Check("runtime_observed", "not_applicable"),),
        )
    )
    with pytest.raises(FixtureError):
        validate_receipt(forged)


def test_legitimate_correct_refusal_is_not_falsely_rejected(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    case = replace(
        public_case,
        task=replace(public_case.task, approval_fixture="refuse"),
        oracle=replace(
            public_case.oracle, outcome="permission_denial", final_status="open", final_revision=0
        ),
    )
    run = run_mock(dataset, "KT-D01", approval="refuse")
    receipt = grade_case(case, dataset, run)
    assert receipt.status == "passed" and receipt.outcome == "permission_denial"
    assert not receipt.blocker and receipt.api_usage is None and receipt.actual_cost is None
    assert run.app.workflow.handler_invocations == 0


def test_valid_structured_paraphrase_does_not_need_verbatim_text(
    dataset: Dataset, public_case: EvaluationCase
) -> None:
    app = DemoApplication(dataset, "KT-D01")
    script = list(workflow_script(app))
    assert isinstance(script[-1], str)
    original = json.loads(script[-1])
    # Preserve the checked claims, change ordering/spacing and citation order.
    script[-1] = json.dumps(
        {
            "citation_ids": list(reversed(original["citation_ids"])),
            "revision": original["revision"],
            "ticket_id": original["ticket_id"],
            "status": original["status"],
            "schema": original["schema"],
        },
        indent=3,
    )
    run = run_mock(dataset, "KT-D01", script=tuple(script))
    receipt = grade_case(public_case, dataset, run)
    assert receipt.status == "passed" and not receipt.errors
