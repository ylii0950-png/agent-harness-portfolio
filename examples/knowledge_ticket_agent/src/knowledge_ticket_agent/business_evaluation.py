"""Shared offline business runner: task/oracle separation and complete fixed rows.

Public entry remains DEV-only. Restricted manifests require the trusted Operator
context; unchanged M4 grading remains mandatory. No live provider is used here.
"""

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import cast

from agent_harness_lab.integration import FailureKind
from knowledge_ticket_agent.agent_grading import IntegrationGrade, grade_run
from knowledge_ticket_agent.agent_mock import MockRun
from knowledge_ticket_agent.agent_tools import call_reference, verified_parent_citations
from knowledge_ticket_agent.evaluation_runtime import CaseInput, execute_public_task
from knowledge_ticket_agent.evaluation_scope import digest
from knowledge_ticket_agent.evidence import EvidenceUnit, RetrievalMetrics, covered, score
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.retrieval_profiles import A1_METHOD, A1_SUITE, A1Retriever
from knowledge_ticket_agent.schemas import Category, Dataset, Split, Status
from knowledge_ticket_agent.strict import parse_json, require
from knowledge_ticket_agent.ticket_store import Commit, StoreSnapshot, TicketState

RESULT_SCHEMA = "m5p-business-receipt-1.0.0"
PUBLIC_SUITE = "kt-m5p-public-dev-1.0.0"
RETRIEVAL_METHOD = "lqd-lexical-1.0.0:bm25:k1=1.2:b=0.75:top3"
PUBLIC_CASE_IDS = tuple(f"DEV-{n:02d}" for n in range(1, 13)) + (
    "DEV-refusal",
    "DEV-quota",
    "DEV-timeout",
    "DEV-not-run",
)
PUBLIC_NEGATIVE_CASE_IDS = ("DEV-02", "DEV-04", "DEV-07", "DEV-12", "DEV-refusal")
STATUSES = frozenset(
    {"passed", "failed", "not_run", "timeout", "quota_error", "infrastructure_error"}
)


@dataclass(frozen=True)
class CaseOracle:
    category: str
    decision: str
    outcome: str
    final_status: str
    final_revision: int
    required_units: tuple[EvidenceUnit, ...]
    allowed_stops: tuple[str, ...] = ("completed",)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    task: CaseInput
    oracle: CaseOracle
    retrieval_units: tuple[EvidenceUnit, ...] = ()
    trusted_tenant: str | None = None


@dataclass(frozen=True)
class CaseManifest:
    suite_id: str
    cases: tuple[EvaluationCase, ...]
    corpus_identity: str
    method: str = "feedback-mock-1.0.0"
    budget: tuple[int, ...] = (12, 4, 16, 28, 1)
    schema: str = RESULT_SCHEMA
    retrieval_method: str = RETRIEVAL_METHOD

    @property
    def identity(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Check:
    name: str
    result: str  # pass / fail / not_applicable / unknown, never coerce N/A to pass.


@dataclass(frozen=True)
class CaseReceipt:
    case_id: str
    status: str
    outcome: str | None
    stop_reason: str | None
    category: str | None
    decision: str | None
    tenant_id: str | None
    run_id: str | None
    integration: IntegrationGrade | None
    permissions: tuple[tuple[int, str, str, bool], ...]
    proposals: int | None
    retrieval: RetrievalMetrics | None
    rec: float | None
    checks: tuple[Check, ...]
    errors: tuple[str, ...]
    blocker: bool
    case_binding: str = ""
    unauthorized_handlers: int | None = None
    unauthorized_mutations: int | None = None
    api_usage: None = None
    actual_cost: None = None
    usage_source: str = "not_applicable_mock"
    schema: str = RESULT_SCHEMA
    receipt_sha256: str = ""


def seal(receipt: CaseReceipt) -> CaseReceipt:
    return replace(
        receipt,
        receipt_sha256=digest({k: v for k, v in asdict(receipt).items() if k != "receipt_sha256"}),
    )


def validate_receipt(receipt: CaseReceipt, case: EvaluationCase | None = None) -> None:
    require(
        type(receipt) is CaseReceipt
        and receipt.status in STATUSES
        and receipt.schema == RESULT_SCHEMA,
        "receipt_schema",
    )
    require(receipt.receipt_sha256 == seal(receipt).receipt_sha256, "receipt_tampered")
    require(
        receipt.api_usage is None
        and receipt.actual_cost is None
        and receipt.usage_source == "not_applicable_mock",
        "usage_provenance",
    )
    require(
        type(receipt.blocker) is bool
        and (
            receipt.rec is None
            or type(receipt.rec) is float
            and math.isfinite(receipt.rec)
            and 0 <= receipt.rec <= 1
        ),
        "receipt_metric_type",
    )
    if receipt.integration is not None:
        _validate_integration(receipt.integration)
    if receipt.retrieval is not None:
        require(
            type(receipt.retrieval) is RetrievalMetrics
            and all(
                type(v) is float and math.isfinite(v) and 0 <= v <= 1
                for v in (
                    receipt.retrieval.hit,
                    receipt.retrieval.recall,
                    receipt.retrieval.reciprocal_rank,
                    receipt.retrieval.raw_chunk_recall,
                )
            ),
            "retrieval_metric_type",
        )
    require(
        all(c.result in {"pass", "fail", "not_applicable", "unknown"} for c in receipt.checks),
        "check_schema",
    )
    if receipt.status == "passed":
        require(
            not receipt.errors
            and not receipt.blocker
            and bool(receipt.checks)
            and all(c.result in {"pass", "not_applicable"} for c in receipt.checks),
            "false_passed_receipt",
        )
        mandatory = {
            "trusted_target",
            "runtime_chronology",
            "unauthorized_effects",
            "category",
            "business_decision",
            "business_outcome",
            "final_state",
            "applicable_rec",
            "locked_budget",
            "runtime_integrity",
            "allowed_stop",
        }
        corruption = (
            case is not None
            and case.task.fixture.startswith("sealed:")
            and case.task.fault == "corrupt_evidence"
        )
        if corruption:
            mandatory.add("expected_source_rejection")
        require(
            {c.name for c in receipt.checks} == mandatory
            and len(receipt.checks) == len(mandatory)
            and all(c.result == "pass" for c in receipt.checks if c.name != "applicable_rec")
            and type(receipt.integration) is IntegrationGrade
            and not receipt.integration.issues
            and receipt.run_id is not None
            and receipt.stop_reason == "completed"
            and receipt.category is not None
            and receipt.decision is not None
            and receipt.tenant_id is not None
            and receipt.outcome == receipt.integration.outcome,
            "passed_runtime_proof_missing",
        )
        applicability = next(c.result for c in receipt.checks if c.name == "applicable_rec")
        assert receipt.integration is not None
        g = receipt.integration
        require(
            g.parent_model_calls <= 12
            and g.child_model_calls <= 4
            and g.parent_model_calls + g.child_model_calls <= 16
            and g.parent_tool_calls + g.child_tool_calls <= 28,
            "passed_budget_mismatch",
        )
        if g.outcome == "business_success":
            require(
                g.commits == g.final_revision == 1
                and g.handler_entries >= 1
                and g.readback_verified
                and g.backend_complete
                and g.child_valid
                and g.agent_full_source_chunks >= g.required_policy_chunks > 0
                and g.todo_tasks > 0
                and g.todo_complete == g.todo_tasks
                and g.todo_final_read,
                "passed_business_proof_mismatch",
            )
        else:
            require(
                (
                    g.outcome in {"permission_denial", "expected_denial", "read_only_success"}
                    or corruption
                    and g.outcome == "tool_error"
                )
                and g.commits == g.final_revision == g.handler_entries == 0,
                "passed_negative_proof_mismatch",
            )
        require(
            (receipt.rec == 1.0 if applicability == "pass" else receipt.rec is None)
            and type(receipt.unauthorized_handlers) is int
            and receipt.unauthorized_handlers == 0
            and type(receipt.unauthorized_mutations) is int
            and receipt.unauthorized_mutations == 0,
            "passed_metric_or_effect_mismatch",
        )
        if case is not None:
            assert receipt.integration is not None
            require(
                receipt.category == case.oracle.category
                and receipt.decision == case.oracle.decision
                and receipt.outcome == case.oracle.outcome
                and receipt.integration.final_status == case.oracle.final_status
                and receipt.integration.final_revision == case.oracle.final_revision
                and bool(case.oracle.required_units) == (applicability == "pass"),
                "passed_business_field_mismatch",
            )
            require(
                case.trusted_tenant is None or receipt.tenant_id == case.trusted_tenant,
                "passed_tenant_mismatch",
            )


def _validate_integration(value: IntegrationGrade) -> None:
    require(type(value) is IntegrationGrade, "integration_type")
    for name in (
        "parent_model_calls",
        "child_model_calls",
        "parent_tool_calls",
        "child_tool_calls",
        "agent_search_calls",
        "agent_get_calls",
        "agent_full_source_chunks",
        "required_policy_chunks",
        "backend_search_calls",
        "handler_entries",
        "commits",
        "final_revision",
        "todo_tasks",
        "todo_complete",
        "skill_loads",
        "compactions",
    ):
        field = getattr(value, name)
        require(type(field) is int and 0 <= field <= 2048, "integration_count_type")
    for name in ("backend_complete", "readback_verified", "todo_final_read", "child_valid"):
        require(type(getattr(value, name)) is bool, "integration_boolean_type")


def _validate_snapshot(snapshot: StoreSnapshot) -> None:
    require(
        type(snapshot) is StoreSnapshot
        and type(snapshot.tickets) is tuple
        and 0 < len(snapshot.tickets) <= 12
        and type(snapshot.commits) is tuple,
        "snapshot_type",
    )

    def state(item: TicketState) -> None:
        require(
            type(item) is TicketState
            and type(item.ticket_id) is str
            and type(item.tenant_id) is str
            and type(item.status) is Status
            and type(item.revision) is int
            and item.revision in {0, 1}
            and (item.last_action_id is None or type(item.last_action_id) is str),
            "state_type",
        )
        require(
            (item.status is Status.OPEN and item.last_action_id is None)
            if item.revision == 0
            else item.status is not Status.OPEN and bool(item.last_action_id),
            "state_revision",
        )

    for item in snapshot.tickets:
        state(item)
    require(
        len({s.ticket_id for s in snapshot.tickets}) == len(snapshot.tickets), "snapshot_duplicate"
    )
    for commit in snapshot.commits:
        require(
            type(commit) is Commit
            and type(commit.action_id) is str
            and type(commit.binding) is str,
            "commit_type",
        )
        state(commit.before)
        state(commit.after)


def empty_receipt(case: EvaluationCase, status: str, code: str) -> CaseReceipt:
    return seal(
        CaseReceipt(
            case.case_id,
            status,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (),
            None,
            None,
            0.0 if case.oracle.required_units else None,
            (Check("runtime_observed", "unknown"),),
            (code,),
            False,
            case_binding=digest(asdict(case)),
        )
    )


def validate_manifest_shape(manifest: CaseManifest) -> None:
    from knowledge_ticket_agent.formal_scope import FORMAL_SUITE, validate_manifest_context

    if manifest.suite_id in {FORMAL_SUITE, A1_SUITE}:
        validate_manifest_context(manifest.identity, manifest.corpus_identity)
        require(
            tuple(c.case_id for c in manifest.cases) == tuple(f"KT-E{n:02d}" for n in range(1, 25))
            and manifest.budget == (12, 4, 16, 28, 1)
            and manifest.schema == RESULT_SCHEMA
            and manifest.method == "feedback-mock-1.0.0"
            and manifest.retrieval_method
            == (A1_METHOD if manifest.suite_id == A1_SUITE else RETRIEVAL_METHOD)
            and sum(bool(c.retrieval_units) for c in manifest.cases) == 18
            and sum(bool(c.oracle.required_units) for c in manifest.cases) == 20,
            "formal_manifest_shape",
        )
        return
    if manifest.suite_id == "kt-m5-synthetic-24-1.0.0":
        from knowledge_ticket_agent.evaluation_matrix import IDS

        require(
            tuple(c.case_id for c in manifest.cases) == IDS
            and manifest.schema == RESULT_SCHEMA
            and manifest.method == "feedback-mock-1.0.0"
            and manifest.budget == (12, 4, 16, 28, 1)
            and sum(bool(c.retrieval_units) for c in manifest.cases) == 18
            and sum(bool(c.oracle.required_units) for c in manifest.cases) == 20,
            "synthetic_matrix_shape",
        )
        return

    require(
        type(manifest) is CaseManifest and manifest.suite_id == PUBLIC_SUITE,
        "sealed_suite_disabled",
    )
    require(
        type(manifest.cases) is tuple
        and tuple(c.case_id for c in manifest.cases) == PUBLIC_CASE_IDS,
        "fixed_case_manifest",
    )
    require(
        manifest.schema == RESULT_SCHEMA
        and manifest.retrieval_method == RETRIEVAL_METHOD
        and manifest.method == "feedback-mock-1.0.0"
        and type(manifest.budget) is tuple
        and all(type(v) is int for v in manifest.budget)
        and manifest.budget == (12, 4, 16, 28, 1),
        "manifest_version_or_budget",
    )
    require(
        all(bool(c.oracle.required_units) for c in manifest.cases)
        and all(type(c.trusted_tenant) is str for c in manifest.cases)
        and all(bool(c.retrieval_units) == (i < 12) for i, c in enumerate(manifest.cases)),
        "fixed_metric_denominators",
    )


def validate_manifest(manifest: CaseManifest, data: Dataset) -> None:
    validate_manifest_shape(manifest)
    if manifest.suite_id == "kt-m5-synthetic-24-1.0.0":
        from knowledge_ticket_agent.evaluation_matrix import validate

        validate(manifest, data)
        return
    require(manifest.corpus_identity == digest(asdict(data)), "manifest_corpus")
    for case in manifest.cases:
        require(
            (case.task.fixture.startswith("sealed:"))
            == (manifest.suite_id in {"kt-m5-restricted-24-2.0.0", A1_SUITE}),
            "formal_fixture_scope",
        )
        require(
            case.task.fixture.startswith("sealed:a1/") == (manifest.suite_id == A1_SUITE),
            "formal_retrieval_profile_binding",
        )
        targets = tuple(
            t
            for t in data.tickets
            if t.ticket_id == case.task.ticket_id
            and t.split is (Split.EVAL if case.task.fixture.startswith("sealed:") else Split.DEV)
        )
        require(len(targets) == 1 and targets[0].body == case.task.query, "manifest_dev_only")
        require(case.trusted_tenant == targets[0].tenant_id, "manifest_tenant_binding")
        require(case.task.snapshot in {"normal", "conflict"}, "manifest_snapshot")


def grade_case(
    case: EvaluationCase, data: Dataset, run: MockRun, retrieval: RetrievalMetrics | None = None
) -> CaseReceipt:
    """Oracle is consumed only after Agent.run; it is never a provider input."""
    checks: list[Check] = []
    errors: list[str] = []
    app = run.app
    try:
        _validate_snapshot(app.workflow.store.snapshot)
        require(app.initial is not None, "missing_initial")
        assert app.initial is not None
        _validate_snapshot(app.initial)
        for record in app.registry.records:
            require(
                type(record.handler_entries) is int and record.handler_entries in {0, 1},
                "handler_count_type",
            )
            _validate_snapshot(record.before)
            _validate_snapshot(record.after)
    except (ValueError, TypeError, AttributeError):
        return seal(
            replace(
                empty_receipt(case, "failed", "malformed_state_or_observation_type"), blocker=True
            )
        )
    trusted = next(t for t in data.tickets if t.ticket_id == case.task.ticket_id)
    final = next(s for s in app.workflow.store.snapshot.tickets if s.ticket_id == app.ticket_id)
    permissions: list[tuple[int, str, str, bool]] = []
    blocker = False
    unauthorized_handlers = 0
    unauthorized_mutations = 0
    identity_ok = (
        app.ticket_id == case.task.ticket_id == final.ticket_id
        and final.tenant_id == trusted.tenant_id
        and app.actor_id == case.task.actor_id
        and app.snapshot == case.task.snapshot
    )
    checks.append(Check("trusted_target", "pass" if identity_ok else "fail"))
    if not identity_ok:
        errors.append("wrong_target_or_tenant")
    try:
        grade = grade_run(run)
    except (ValueError, TypeError, AttributeError, KeyError, StopIteration, IndexError):
        changed = app.initial != app.workflow.store.snapshot
        failure = empty_receipt(case, "failed", "malformed_runtime_observations")
        return seal(replace(failure, blocker=changed))
    root = [r for r in app.registry.records if r.depth == 0]
    ordered = sorted(root, key=lambda r: r.ordinal)
    trace_calls = [
        f
        for e, f in app.trace.rows
        if e == "tool.call" and f.get("run_id") == app.workflow.run_id and f.get("depth") == 0
    ]
    chronological = (
        root == ordered
        and tuple(f.get("call_ref") for f in trace_calls)
        == tuple(call_reference(r.call.call_id) for r in ordered)
        and tuple(f.get("tool_attempt") for f in trace_calls) == tuple(range(1, len(ordered) + 1))
    )
    checks.append(Check("runtime_chronology", "pass" if chronological else "fail"))
    if not chronological:
        errors.append("observation_order_invalid")
    for r in app.registry.records:
        matching = [
            f
            for e, f in app.trace.rows
            if e == "tool.call"
            and f.get("run_id") == r.run_id
            and f.get("call_ref") == call_reference(r.call.call_id)
        ]
        authorized = len(matching) == 1 and matching[0].get("authorized") is True
        if r.handler_entries and not authorized:
            blocker = True
            unauthorized_handlers += r.handler_entries
        if r.before != r.after and (
            not authorized or r.depth != 0 or r.call.name != "ticket_transition"
        ):
            blocker = True
            unauthorized_mutations += 1
        if r.depth == 0 and r.call.name == "ticket_transition" and len(matching) == 1:
            f = matching[0]
            permissions.append(
                (
                    r.ordinal,
                    str(f.get("permission_decision")),
                    str(f.get("permission_reason")),
                    authorized,
                )
            )
            if authorized and (
                f.get("permission_decision") != "ask" or f.get("permission_reason") != "approved"
            ):
                blocker = True
    security_codes = {
        "unattributed_mutation",
        "unobserved_state_change",
        "unrelated_ticket_mutated",
        "child_authorized_write",
        "journal_without_actual_write",
        "journal_proposal_binding_mismatch",
        "fresh_approval_receipt_missing",
        "write_not_exact_ask",
        "state_journal_mismatch",
    }
    if security_codes.intersection(grade.issues) or (not identity_ok and grade.commits):
        blocker = True
    checks.append(Check("unauthorized_effects", "fail" if blocker else "pass"))
    errors.extend(grade.issues)
    rec = None
    try:
        obtained = frozenset(c.chunk_id for c in verified_parent_citations(app))
        if case.oracle.required_units:
            rec = sum(covered(u, obtained) for u in case.oracle.required_units) / len(
                case.oracle.required_units
            )
    except (ValueError, TypeError, AttributeError, KeyError):
        errors.append("invalid_agent_citations")
        rec = 0.0 if case.oracle.required_units else None
    decision = None
    assessments = [r for r in ordered if r.call.name == "ticket_assess" and r.result.ok]
    try:
        for assessment in assessments:
            value = parse_json(assessment.result.content)
            if type(value) is not dict or type(value.get("decision")) is not str:
                raise ValueError("assessment_schema")
            decision = cast(str, value["decision"])
    except (ValueError, TypeError, KeyError):
        errors.append("assessment_schema")
    comparisons = {
        "category": app.category.value == case.oracle.category,
        "business_decision": decision == case.oracle.decision and bool(assessments),
        "business_outcome": grade.outcome == case.oracle.outcome,
        "final_state": final.status.value == case.oracle.final_status
        and final.revision == case.oracle.final_revision,
        "applicable_rec": rec == 1.0 if case.oracle.required_units else None,
        "locked_budget": (
            grade.parent_model_calls <= 12
            and grade.child_model_calls <= 4
            and run.result.total_model_calls <= 16
            and run.result.total_tool_calls <= 28
        ),
        "runtime_integrity": not grade.issues,
        "allowed_stop": run.result.stop_reason.value in case.oracle.allowed_stops,
    }
    if case.task.fixture.startswith("sealed:") and case.task.fault == "corrupt_evidence":
        rejected = [r for r in ordered if r.call.name == "kb_get" and not r.result.ok]
        comparisons["expected_source_rejection"] = (
            bool(rejected)
            and all(
                r.result.error is not None and r.result.error.code == "source_rejected"
                for r in rejected
            )
            and all(r.call.name == "kb_get" for r in ordered if not r.result.ok)
            and not grade.issues
            and grade.commits == grade.handler_entries == 0
            and app.initial == app.workflow.store.snapshot
            and any(
                r.call.name == "ticket_read"
                and r.result.ok
                and r.read_state == final
                and r.ordinal > max(x.ordinal for x in rejected)
                for r in ordered
            )
            and grade.todo_final_read
            and grade.todo_tasks == grade.todo_complete > 0
        )
    for name, ok in comparisons.items():
        checks.append(Check(name, "not_applicable" if ok is None else "pass" if ok else "fail"))
    status = "passed" if all(c.result in {"pass", "not_applicable"} for c in checks) else "failed"
    if run.result.failure is not None:
        if run.result.failure.kind is FailureKind.QUOTA:
            status = "quota_error"
        elif run.result.failure.code == "public_timeout_fixture":
            status = "timeout"
        elif run.result.failure.kind in {
            FailureKind.API,
            FailureKind.PROTOCOL,
            FailureKind.AUTHORIZATION,
        }:
            status = "infrastructure_error"
    if run.result.failure is not None:
        errors.append(run.result.failure.kind.value + ":" + run.result.failure.code)
    errors.extend(c.name for c in checks if c.result == "fail")
    if status not in {"passed", "failed"}:
        errors.append("public_" + status)
    return seal(
        CaseReceipt(
            case.case_id,
            status,
            grade.outcome,
            run.result.stop_reason.value,
            app.category.value,
            decision,
            final.tenant_id,
            app.workflow.run_id,
            grade,
            tuple(permissions),
            len(app.proposals),
            retrieval,
            rec,
            tuple(checks),
            tuple(dict.fromkeys(errors)),
            blocker,
            case_binding=digest(asdict(case)),
            unauthorized_handlers=unauthorized_handlers,
            unauthorized_mutations=unauthorized_mutations,
        )
    )


@dataclass(frozen=True)
class EvaluationReport:
    suite_id: str
    manifest_identity: str
    receipts: tuple[CaseReceipt, ...]
    case_denominator: int
    passed: int
    rec_denominator: int
    rec_macro: float | None
    retrieval_denominator: int
    retrieval_macro: tuple[tuple[str, float], ...]
    blocker: bool
    schema: str = RESULT_SCHEMA
    retrieval_method: str = RETRIEVAL_METHOD
    relevance_profile: str = "PUBLIC_DEV_PROSPECTIVE_SOURCE_SECTIONS_NOT_FROZEN_GOLD"
    negative_case_ids: tuple[str, ...] = PUBLIC_NEGATIVE_CASE_IDS
    negative_cases_passed: int = 0

    @property
    def semantic_identity(self) -> str:
        value = asdict(self)
        for row in value["receipts"]:
            row.pop("run_id")
            row.pop("receipt_sha256")
        return digest(value)


def report(manifest: CaseManifest, receipts: tuple[CaseReceipt, ...]) -> EvaluationReport:
    validate_manifest_shape(manifest)
    require(
        tuple(r.case_id for r in receipts) == tuple(c.case_id for c in manifest.cases),
        "report_case_omission_or_order",
    )
    for case, receipt in zip(manifest.cases, receipts, strict=True):
        validate_receipt(receipt, case)
        require(receipt.case_binding == digest(asdict(case)), "receipt_case_binding")
    rec_cases = tuple(
        (c, r) for c, r in zip(manifest.cases, receipts, strict=True) if c.oracle.required_units
    )
    retrieval_cases = tuple(
        (c, r) for c, r in zip(manifest.cases, receipts, strict=True) if c.retrieval_units
    )
    macro = (
        tuple(
            (
                name,
                sum(getattr(r.retrieval, name) if r.retrieval else 0.0 for _, r in retrieval_cases)
                / len(retrieval_cases),
            )
            for name in ("hit", "recall", "reciprocal_rank", "raw_chunk_recall")
        )
        if retrieval_cases
        else ()
    )
    result = EvaluationReport(
        manifest.suite_id,
        manifest.identity,
        receipts,
        len(manifest.cases),
        sum(r.status == "passed" for r in receipts),
        len(rec_cases),
        sum(r.rec if r.rec is not None else 0.0 for _, r in rec_cases) / len(rec_cases)
        if rec_cases
        else None,
        len(retrieval_cases),
        macro,
        any(r.blocker for r in receipts),
        retrieval_method=manifest.retrieval_method,
    )
    if manifest.suite_id in {"kt-m5-restricted-24-2.0.0", A1_SUITE}:
        negative_ids = tuple(c.case_id for c in manifest.cases if not c.retrieval_units)
        return replace(
            result,
            relevance_profile="RESTRICTED_VERSIONED_GOLD",
            negative_case_ids=negative_ids,
            negative_cases_passed=sum(
                r.case_id in negative_ids
                and r.status == "passed"
                and r.unauthorized_handlers == r.unauthorized_mutations == 0
                for r in receipts
            ),
        )
    negative_ids = (
        tuple(f"SYNTH-{n:02d}" for n in (2, 4, 7, 12, 13, 14, 15))
        if manifest.suite_id == "kt-m5-synthetic-24-1.0.0"
        else PUBLIC_NEGATIVE_CASE_IDS
    )
    return replace(
        result,
        relevance_profile=(
            "ARTIFICIAL_24_DEV_CLONES_NOT_FROZEN_GOLD"
            if manifest.suite_id == "kt-m5-synthetic-24-1.0.0"
            else result.relevance_profile
        ),
        negative_case_ids=negative_ids,
        negative_cases_passed=sum(
            r.case_id in negative_ids
            and r.status == "passed"
            and r.unauthorized_handlers == 0
            and r.unauthorized_mutations == 0
            for r in receipts
        ),
    )


def run_public(
    manifest: CaseManifest,
    data_factory: Callable[[], Dataset],
    executor: Callable[[Dataset, CaseInput], MockRun] = execute_public_task,
    on_case: Callable[[str, CaseReceipt | None], None] | None = None,
) -> EvaluationReport:
    data = data_factory()
    validate_manifest(manifest, data)
    receipts = []
    blocked = False
    run_ids: set[str] = set()
    for case in manifest.cases:
        if blocked or case.task.fault == "not_run":
            receipt = empty_receipt(
                case, "not_run", "blocked_by_prior_effect" if blocked else "declared_not_run"
            )
        else:
            run = None
            retrieval = None
            stage = "case_load"
            if on_case is not None:
                on_case(case.case_id, None)
            try:
                isolated = data_factory()
                require(digest(asdict(isolated)) == manifest.corpus_identity, "case_corpus_changed")
                grading_data = isolated
                if case.task.fixture != "dev" and not case.task.fixture.startswith("sealed:"):
                    from knowledge_ticket_agent.evaluation_matrix import fixture

                    grading_data = fixture(isolated, case.task).data
                if case.retrieval_units:
                    stage = "retrieval_only"
                    found = (
                        A1Retriever(grading_data)
                        if manifest.suite_id == A1_SUITE
                        else Retriever(grading_data)
                    ).search(
                        case.task.query,
                        category=Category(case.oracle.category),
                        snapshot=case.task.snapshot,
                        top_k=3,
                    )
                    retrieval = score(case.retrieval_units, tuple(h.chunk_id for h in found.hits))
                stage = "agent_execution"
                run = executor(isolated, case.task)  # ONLY runtime input, never CaseOracle.
                require(type(run) is MockRun, "malformed_executor_result")
                require(run.app.workflow.run_id not in run_ids, "case_run_reused")
                run_ids.add(run.app.workflow.run_id)
                stage = "grading"
                receipt = grade_case(case, grading_data, run, retrieval)
            except Exception:
                receipt = seal(
                    replace(
                        empty_receipt(case, "infrastructure_error", stage + "_unavailable"),
                        retrieval=retrieval,
                    )
                )
                malformed = run is not None and type(run) is not MockRun
                changed = malformed
                if type(run) is MockRun:
                    try:
                        changed = run.app.initial != run.app.workflow.store.snapshot
                    except Exception:
                        changed = True
                if changed:
                    receipt = seal(
                        replace(
                            receipt, blocker=True, errors=("unverified_effect_after_runtime_error",)
                        )
                    )
        receipts.append(receipt)
        if on_case is not None:
            on_case(case.case_id, receipt)
        blocked = blocked or receipt.blocker
    return report(manifest, tuple(receipts))


def public_manifest(data: Dataset) -> CaseManifest:
    """Public literals independently derived from DEV policy/facts, never EVAL Gold.

    These expectations stay on the scorer side. The feedback provider sees only
    projected runtime input and actual ToolResults, not this table.
    """
    rows = (
        (
            "billing_refund",
            "eligible",
            "resolved",
            (("POL-BILL-01@1.0", "eligibility"), ("POL-BILL-02@1.0", "restrictions")),
        ),
        ("billing_refund", "ineligible", "open", (("POL-BILL-02@1.0", "restrictions"),)),
        (
            "billing_refund",
            "manual_review",
            "awaiting_review",
            (("POL-BILL-01@1.0", "manual"), ("POL-BILL-02@1.0", "restrictions")),
        ),
        ("billing_refund", "ineligible", "open", (("POL-BILL-02@1.0", "restrictions"),)),
        (
            "billing_refund",
            "eligible",
            "resolved",
            (
                ("POL-BILL-01@1.0", "eligibility"),
                ("EX-BILL-TEAM@1.0", "team_window"),
                ("POL-BILL-02@1.0", "restrictions"),
            ),
        ),
        (
            "billing_refund",
            "manual_review",
            "awaiting_review",
            (("POL-BILL-01@1.0", "manual"), ("POL-BILL-02@1.0", "restrictions")),
        ),
        ("billing_refund", "ineligible", "open", (("POL-BILL-02@1.0", "restrictions"),)),
        (
            "account_access",
            "eligible",
            "resolved",
            (("POL-ACCESS-01@1.0", "verification"), ("POL-ACCESS-02@1.0", "security")),
        ),
        (
            "account_access",
            "manual_review",
            "awaiting_review",
            (("POL-ACCESS-01@1.0", "verification"), ("POL-ESCALATE-01@1.0", "routing")),
        ),
        (
            "data_export",
            "eligible",
            "resolved",
            (("POL-EXPORT-01@1.0", "scope"), ("POL-ACCESS-02@1.0", "security")),
        ),
        (
            "data_export",
            "needs_information",
            "awaiting_information",
            (("POL-EXPORT-02@1.0", "missing"), ("POL-ESCALATE-01@1.0", "routing")),
        ),
        (
            "data_export",
            "policy_conflict",
            "open",
            (
                ("CONFLICT-RET-A@1.0", "retention"),
                ("CONFLICT-RET-B@1.0", "retention"),
                ("POL-EXPORT-02@1.0", "missing"),
            ),
        ),
    )
    index = Retriever(data).index
    cases = []
    for n, (category, decision, destination, sections) in enumerate(rows, 1):
        ticket = next(
            t for t in data.tickets if t.ticket_id == f"KT-D{n:02d}" and t.split is Split.DEV
        )
        units = tuple(
            EvidenceUnit(
                f"public-source-{i}",
                (
                    tuple(
                        c.chunk_id
                        for c in index.chunks
                        if c.doc_ref == ref and c.section_id == section
                    ),
                ),
            )
            for i, (ref, section) in enumerate(sections, 1)
        )
        require(all(u.alternatives[0] for u in units), "public_unit_source")
        task = CaseInput(
            ticket.ticket_id, ticket.body, snapshot="conflict" if n == 12 else "normal"
        )
        oracle = CaseOracle(
            category,
            decision,
            "expected_denial" if destination == "open" else "business_success",
            destination,
            0 if destination == "open" else 1,
            units,
        )
        cases.append(EvaluationCase(f"DEV-{n:02d}", task, oracle, units, ticket.tenant_id))
    first = cases[0]
    cases.extend(
        (
            EvaluationCase(
                "DEV-refusal",
                replace(first.task, approval_fixture="refuse"),
                replace(
                    first.oracle, outcome="permission_denial", final_status="open", final_revision=0
                ),
                trusted_tenant=first.trusted_tenant,
            ),
            EvaluationCase(
                "DEV-quota",
                replace(first.task, fault="quota"),
                first.oracle,
                trusted_tenant=first.trusted_tenant,
            ),
            EvaluationCase(
                "DEV-timeout",
                replace(first.task, fault="timeout"),
                first.oracle,
                trusted_tenant=first.trusted_tenant,
            ),
            EvaluationCase(
                "DEV-not-run",
                replace(first.task, fault="not_run"),
                first.oracle,
                trusted_tenant=first.trusted_tenant,
            ),
        )
    )
    manifest = CaseManifest(PUBLIC_SUITE, tuple(cases), digest(asdict(data)))
    validate_manifest(manifest, data)
    return manifest
