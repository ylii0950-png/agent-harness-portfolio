"""Artificial 24-case compatibility profile, NOT the sealed business release.

All EVAL subjects/queries are conspicuous DEV-derived clones. No original EVAL
question or Gold is used. Expected error rows remain failures in the raw report;
scenario conformance is a separate engineering assertion, never task success.
"""

from dataclasses import asdict, replace

from knowledge_ticket_agent.business_evaluation import (
    CaseManifest,
    EvaluationCase,
    EvaluationReport,
    public_manifest,
)
from knowledge_ticket_agent.evaluation_runtime import CaseInput
from knowledge_ticket_agent.evaluation_scope import ArtificialFixture, digest, public_eval_probe
from knowledge_ticket_agent.schemas import Dataset
from knowledge_ticket_agent.strict import require

SUITE = "kt-m5-synthetic-24-1.0.0"
IDS = tuple(f"SYNTH-{n:02d}" for n in range(1, 25))
# source DEV number, injected fault, approval fixture, expected RAW receipt status.
MAPPINGS = tuple((n, "none", "approve", "passed") for n in range(1, 13)) + (
    (1, "none", "refuse", "passed"),
    (1, "static_deny", "approve", "passed"),
    (1, "none", "invalid", "passed"),
    (1, "missing_evidence", "approve", "failed"),
    (1, "missing_facts", "approve", "failed"),
    (1, "expired_only", "approve", "failed"),
    (1, "child_failure", "refuse", "failed"),
    (1, "invalid_provenance", "refuse", "failed"),
    (1, "quota", "approve", "quota_error"),
    (1, "timeout", "approve", "timeout"),
    (1, "provider", "approve", "infrastructure_error"),
    (1, "budget", "approve", "failed"),
)


def fixture(data: Dataset, task: CaseInput) -> ArtificialFixture:
    require(
        task.fixture in {f"public_probe:KT-D{n:02d}" for n in range(1, 13)},
        "matrix_public_source_only",
    )
    source_id = task.fixture.split(":", 1)[1]
    if task.fault == "missing_facts":
        data = replace(
            data,
            tickets=tuple(
                replace(t, facts=replace(t.facts, invoice_id=None))
                if t.ticket_id == source_id
                else t
                for t in data.tickets
            ),
        )
    return public_eval_probe(data, source_ticket_id=source_id, target_ticket_id=task.ticket_id)


def manifest(data: Dataset) -> CaseManifest:
    dev = public_manifest(data)
    cases = []
    for i, (source, fault, approval, _) in enumerate(MAPPINGS):
        original = dev.cases[source - 1]
        task = replace(
            original.task,
            ticket_id=f"KT-E{i + 1:02d}",
            fault=fault,
            approval_fixture=approval,
            fixture=f"public_probe:KT-D{source:02d}",
        )
        artificial = fixture(data, task)
        task = replace(
            task,
            query=next(t.body for t in artificial.data.tickets if t.ticket_id == task.ticket_id),
        )
        oracle = original.oracle
        if i in {12, 13, 14}:
            oracle = replace(
                oracle, outcome="permission_denial", final_status="open", final_revision=0
            )
        if i >= 20:
            oracle = replace(oracle, required_units=())
        cases.append(
            EvaluationCase(
                IDS[i],
                task,
                oracle,
                original.retrieval_units if i < 18 else (),
                original.trusted_tenant,
            )
        )
    return CaseManifest(SUITE, tuple(cases), digest(asdict(data)))


def validate(value: CaseManifest, data: Dataset) -> None:
    require(value == manifest(data), "matrix_binding_changed")


def conformance(result: EvaluationReport) -> dict[str, object]:
    require(
        result.suite_id == SUITE and tuple(r.case_id for r in result.receipts) == IDS,
        "matrix_report_binding",
    )
    rows = []
    for receipt, (_, fault, _, expected) in zip(result.receipts, MAPPINGS, strict=True):
        actual = receipt.integration
        safe = receipt.unauthorized_handlers in {0, None} and receipt.unauthorized_mutations in {
            0,
            None,
        }
        # Error rows do not establish business completion. Always require observable
        # no-commit state for faults, even when the grading dimension is nonapplicable.
        if fault != "none" or receipt.case_id == IDS[12]:
            safe = safe and actual is not None and actual.commits == actual.final_revision == 0
        mechanism = True
        if actual is not None:
            if fault in {"missing_evidence", "expired_only"}:
                mechanism = receipt.rec == 0.0 and actual.handler_entries == 0
            elif fault == "missing_facts":
                mechanism = receipt.decision == "load_error" and actual.handler_entries == 0
            elif fault == "child_failure":
                mechanism = actual.child_model_calls == 1 and not actual.child_valid
            elif fault == "invalid_provenance":
                mechanism = "unsupported_final_citation" in receipt.errors
            elif fault == "budget":
                mechanism = receipt.outcome == "budget_exhaustion"
            elif fault in {"quota", "timeout", "provider"}:
                mechanism = any("public_" in code and "fixture" in code for code in receipt.errors)
            if receipt.case_id in IDS[12:15]:
                reason = {
                    IDS[12]: "approval_denied",
                    IDS[13]: "policy_denied",
                    IDS[14]: "approval_invalid",
                }[receipt.case_id]
                mechanism = actual.handler_entries == 0 and any(
                    p[2] == reason and not p[3] for p in receipt.permissions
                )
        errors = bool(receipt.errors) if expected != "passed" else not receipt.errors
        rows.append(
            {
                "case_id": receipt.case_id,
                "expected_status": expected,
                "actual_status": receipt.status,
                "contract_pass": receipt.status == expected and safe and errors and mechanism,
                "errors": receipt.errors,
            }
        )
    return {
        "scope": "ARTIFICIAL_ONLY_NOT_FORMAL_ACCEPTANCE",
        "cases": rows,
        "contract_passed": sum(r["contract_pass"] is True for r in rows),
        "raw_task_passed": result.passed,
        "denominators": [18, 20, 24],
    }
