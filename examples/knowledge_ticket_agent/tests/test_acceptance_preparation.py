"""Public substitutes only: no private input, production claim, OAuth or model."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.acceptance_inputs import schema, validate_business
from knowledge_ticket_agent.business_evaluation import (
    CaseManifest,
    EvaluationCase,
    grade_case,
    validate_receipt,
)
from knowledge_ticket_agent.evaluation_preflight import current_revision, source_digests
from knowledge_ticket_agent.evaluation_runtime import CaseInput, execute_public_task
from knowledge_ticket_agent.evaluation_scope import digest
from knowledge_ticket_agent.evidence import EvidenceUnit
from knowledge_ticket_agent.formal_evaluation import (
    M2_VERSION,
    FormalOperator,
    FormalRelease,
    GoldContract,
)
from knowledge_ticket_agent.formal_scope import FORMAL_SUITE, issue, operator_scope
from knowledge_ticket_agent.quality_formal import (
    CONTRACT,
    DENOMINATORS,
    SUITE,
    DecisionGold,
    DocumentaryInput,
    QualityGold,
    QualityInputs,
    RankingGold,
    RankingInput,
    SectionTarget,
    evaluate,
    validate,
)
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.retrieval_strategies import strategy
from knowledge_ticket_agent.schemas import Category, Dataset, Split
from knowledge_ticket_agent.strict import decode, parse_json

from .conftest import DEMO


def m2_substitute() -> tuple[QualityInputs, QualityGold]:
    public = json.loads((DEMO / "verification/m2-revised/cases-v3.0.2.json").read_text())
    data = load_dataset()
    ranking, units = [], []
    for n, row in enumerate(public["ranking"]):
        identifier = f"PUBLIC-R{n}"
        ranking.append(RankingInput(identifier, row["query"], Category(row["category"]), "normal"))
        units.append(
            RankingGold(
                identifier,
                tuple(
                    EvidenceUnit(
                        str(k),
                        (
                            tuple(
                                c.chunk_id
                                for c in Retriever(data).index.chunks
                                if c.doc_ref == ref and c.section_id == section
                            ),
                        ),
                    )
                    for k, (ref, section) in enumerate(row["sections"])
                ),
            )
        )
    tasks: dict[str, list[DocumentaryInput]] = {}
    expected: dict[str, list[DecisionGold]] = {}
    for kind in ("documentary", "negative"):
        tasks[kind], expected[kind] = [], []
        for n, row in enumerate(public[kind]):
            identifier = f"PUBLIC-{kind}-{n}"
            tasks[kind].append(
                DocumentaryInput(
                    identifier,
                    row["query"],
                    row["ticket"],
                    Category(row["category"]),
                    row["intent"],
                    tuple(SectionTarget(*s) for s in row["sections"]),
                    row["snapshot"],
                )
            )
            expected[kind].append(DecisionGold(identifier, row["expected"]))
    return (
        QualityInputs(
            "PUBLIC-SUBSTITUTE-NOT-INDEPENDENT-1",
            data,
            tuple(ranking),
            tuple(tasks["documentary"]),
            tuple(tasks["negative"]),
        ),
        QualityGold(
            "PUBLIC-GOLD-1",
            CONTRACT,
            SUITE,
            strategy("A1").sha256,
            tuple(units),
            tuple(expected["documentary"]),
            tuple(expected["negative"]),
        ),
    )


def private_file(path: Path, value: object) -> str:
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def m2_operator(tmp_path: Path) -> tuple[FormalOperator, FormalRelease]:
    root = tmp_path / "m2-public-rehearsal"
    root.mkdir(mode=0o700)
    inputs, gold = m2_substitute()
    release = FormalRelease(
        current_revision(),
        source_digests(),
        digest(asdict(gold)),
        digest(asdict(inputs.data)),
        private_file(root / "payload.json", asdict(inputs)),
        private_file(root / "gold-contract.json", asdict(gold)),
        gold.version,
        CONTRACT,
        "0" * 64,
        "NOT_APPLICABLE",
        profile="public_rehearsal",
        evaluator=M2_VERSION,
        suite=SUITE,
        denominators=DENOMINATORS,
        dataset_version=inputs.version,
    )
    private_file(
        root / "approval.json",
        {
            "release_identity": release.identity,
            "decision": "AUTHORIZE_LOCKED_M2_ONCE",
            "profile": "rehearsal",
            "m2_dependency": "NOT_APPLICABLE",
        },
    )
    return FormalOperator(root, rehearsal=True, purpose="M2"), release


def test_new_m2_inputs_score_only_inside_rehearsal_claim(tmp_path: Path) -> None:
    operator, release = m2_operator(tmp_path)
    result = operator.execute_m2(release)
    assert result["engineering_contract"] == "PASS"
    assert result["revised_formal_quality"] == "NOT_RUN" and len(result["cases"]) == 28
    assert result["denominators"] == DENOMINATORS
    assert len(list((operator.root / "claim").glob("*.receipt.json"))) == 28
    assert operator.inspect()["state"] == "COMPLETED"
    with pytest.raises(FileExistsError):
        operator.execute_m2(release)
    with pytest.raises(ValueError):
        FormalOperator(tmp_path / "another-output", purpose="M2")


@pytest.mark.parametrize("mutation", ["gold", "version", "input", "denominator"])
def test_m2_mismatch_fails_closed(tmp_path: Path, mutation: str) -> None:
    operator, release = m2_operator(tmp_path)
    if mutation in {"gold", "input"}:
        path = operator.root / ("payload.json" if mutation == "input" else "gold-contract.json")
        path.write_text("{}")
    if mutation == "version":
        release = replace(release, gold_version="WRONG")
    if mutation == "denominator":
        release = replace(release, denominators=(1, 1, 1))
    with pytest.raises(ValueError):
        operator.execute_m2(release)
    assert operator.inspect()["state"] in {"FAILED", "NOT_STARTED"}


def test_m2_interruption_preserves_rows_and_blocks_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import knowledge_ticket_agent.quality_formal as module

    operator, release = m2_operator(tmp_path)

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "resolve_policy", interrupt)
    with pytest.raises(KeyboardInterrupt):
        operator.execute_m2(release)
    assert operator.inspect()["state"] == "FAILED"
    summary = json.loads((operator.root / "claim/failure-summary.json").read_text())
    assert len(summary["cases"]) == 28
    assert sum(c["status"] == "saved_receipt" for c in summary["cases"]) == 12
    with pytest.raises(FileExistsError):
        operator.execute_m2(release)


def business_substitute() -> tuple[Dataset, GoldContract]:
    """Repeated PUBLIC slots exercise interfaces; never a new hidden benchmark."""
    from knowledge_ticket_agent.business_evaluation import public_manifest

    data = load_dataset()
    original = public_manifest(data).cases[0]
    source = next(t for t in data.tickets if t.ticket_id == "KT-D01")
    tickets = {
        f"KT-E{n:02d}": replace(
            source,
            ticket_id=f"KT-E{n:02d}",
            split=Split.EVAL,
            body=f"PUBLIC SUBSTITUTE slot {n}. first purchase credit eligibility policy",
            facts=replace(source.facts, requested_action=None) if n == 24 else source.facts,
        )
        for n in range(1, 25)
    }
    data = replace(
        data,
        tickets=tuple(tickets.get(t.ticket_id, t) for t in data.tickets),
        staff=tuple(
            replace(s, allowed_ticket_ids=tuple(tickets)) if s.actor_id == "SYN-STAFF-02" else s
            for s in data.staff
        ),
    )
    unit = EvidenceUnit(
        "PUBLIC-initial",
        (
            tuple(
                c.chunk_id
                for c in Retriever(data).index.chunks
                if c.doc_ref == "POL-BILL-01@1.0" and c.section_id == "eligibility"
            ),
        ),
    )
    cases = []
    for n, ticket in enumerate(tickets.values(), 1):
        fault = {
            18: "static_deny",
            19: "stale_approval",
            21: "missing_evidence",
            22: "expired_only",
            23: "corrupt_evidence",
        }.get(n, "none")
        mode = {16: "child_write", 17: "path_probe", 20: "readback_order"}.get(n, "normal")
        approval = "approve" if n <= 8 or n in {18, 19} else "refuse"
        outcome = (
            "business_success"
            if n <= 8
            else "permission_denial"
            if n <= 20
            else "tool_error"
            if n == 23
            else "expected_denial"
        )
        oracle = replace(
            original.oracle,
            outcome=outcome,
            category="other" if n == 24 else original.oracle.category,
            decision="unsupported" if n == 24 else original.oracle.decision,
            final_status="resolved" if n <= 8 else "open",
            final_revision=int(n <= 8),
            required_units=original.oracle.required_units if n <= 20 else (),
        )
        cases.append(
            EvaluationCase(
                ticket.ticket_id,
                CaseInput(
                    ticket.ticket_id,
                    ticket.body,
                    approval_fixture=approval,
                    fault=fault,
                    fixture="sealed:" + mode,
                ),
                oracle,
                (unit,) if n <= 18 else (),
                ticket.tenant_id,
            )
        )
    return data, GoldContract(
        "PUBLIC-SUBSTITUTE-GOLD-NEW-2",
        CaseManifest(FORMAL_SUITE, tuple(cases), digest(asdict(data))),
    )


def test_new_public_m5_can_satisfy_all_24_safe_business_contracts(tmp_path: Path) -> None:
    from knowledge_ticket_agent.formal_evaluation import FormalPayload

    from .test_quality_closure import prepare

    operator, _, _ = prepare(tmp_path)
    data, gold = business_substitute()
    validate_business(data, gold)
    release = FormalRelease(
        current_revision(),
        source_digests(),
        gold.manifest.identity,
        digest(asdict(data)),
        "1" * 64,
        "2" * 64,
        gold.version,
        "PUBLIC-M2",
        "3" * 64,
        "PO_EXPERIMENTAL_EXCEPTION",
        profile="public_rehearsal",
        dataset_version="PUBLIC-SUBSTITUTE-DATA-2",
    )
    private_file(
        operator.root / "approval.json",
        {
            "release_identity": release.identity,
            "decision": "AUTHORIZE_LOCKED_M5_ONCE",
            "profile": "rehearsal",
            "m2_dependency": release.m2_dependency,
        },
    )
    report = operator.execute(release, loader=lambda: FormalPayload(data, gold.manifest))
    assert report.passed == 24, [
        (r.case_id, r.errors) for r in report.receipts if r.status != "passed"
    ]
    assert report.rec_macro == 1.0 and dict(report.retrieval_macro)["recall"] >= 0.9
    assert report.relevance_profile == "PUBLIC_SYNTHETIC_FORMAL_COMPATIBILITY_NOT_FORMAL"
    assert all(r.unauthorized_handlers == r.unauthorized_mutations == 0 for r in report.receipts)


@pytest.mark.parametrize(
    "fault,status",
    [
        ("quota", "quota_error"),
        ("timeout", "timeout"),
        ("provider", "infrastructure_error"),
        ("budget", "failed"),
        ("invalid_provenance", "failed"),
    ],
)
def test_errors_are_never_safe_business_success(fault: str, status: str) -> None:
    data, gold = business_substitute()
    case = gold.manifest.cases[0]
    case = replace(case, task=replace(case.task, fault=fault))
    with operator_scope("PUBLIC", gold.manifest.identity, digest(asdict(data))):
        run = execute_public_task(data, case.task, case_scope=issue(data, case.task.ticket_id))
        receipt = grade_case(case, data, run)
    assert receipt.status == status


def test_all_failure_or_infrastructure_gold_rejected_before_execution() -> None:
    data, gold = business_substitute()
    for outcome in ("agent_failure", "budget_exhaustion", "tool_error", "read_only_success"):
        cases = tuple(
            replace(c, oracle=replace(c.oracle, outcome=outcome)) for c in gold.manifest.cases
        )
        with pytest.raises(ValueError):
            validate_business(data, replace(gold, manifest=replace(gold.manifest, cases=cases)))


def test_schema_and_strict_input_roundtrip_without_execution() -> None:
    data, gold = business_substitute()
    inputs, m2gold = m2_substitute()
    for model, value in (
        (Dataset, data),
        (GoldContract, gold),
        (QualityInputs, inputs),
        (QualityGold, m2gold),
    ):
        assert decode(model, parse_json(json.dumps(asdict(value)))) == value
        document = schema(model)
        assert "$defs" in document and document["$ref"].endswith(model.__name__)
    validate(inputs, m2gold)
    assert evaluate(inputs, m2gold)["revised_formal_quality"] == "NOT_RUN"


def test_required_evidence_cannot_be_replaced_by_safe_error_label() -> None:
    data, gold = business_substitute()
    case = gold.manifest.cases[22]
    required = gold.manifest.cases[0].oracle.required_units
    case = replace(case, oracle=replace(case.oracle, required_units=required))
    with operator_scope("PUBLIC", gold.manifest.identity, digest(asdict(data))):
        run = execute_public_task(data, case.task, case_scope=issue(data, case.task.ticket_id))
        receipt = grade_case(case, data, run)
    assert receipt.status == "failed" and receipt.rec == 0.0
    validate_receipt(receipt, case)


@pytest.mark.parametrize("slot", [17, 19, 23, 24])
def test_boundary_mechanisms_are_actual_observations(slot: int) -> None:
    data, gold = business_substitute()
    case = gold.manifest.cases[slot - 1]
    with operator_scope("PUBLIC", gold.manifest.identity, digest(asdict(data))):
        run = execute_public_task(data, case.task, case_scope=issue(data, case.task.ticket_id))
        receipt = grade_case(case, data, run)
    validate_receipt(receipt, case)
    assert receipt.status == "passed" and receipt.integration is not None
    assert receipt.integration.handler_entries == receipt.integration.commits == 0
    assert run.app.initial == run.app.workflow.store.snapshot
    if slot == 17:
        observed = next(r for r in run.app.registry.records if r.call.name == "read_file")
        assert observed.handler_entries == 1 and observed.result.error is not None
        assert observed.result.error.code == "workspace_path_denied"
    if slot == 19:
        assert len(run.app.approval_calls) == 1
        assert receipt.permissions[0][2:] == ("approval_denied", False)
    if slot == 23:
        gets = [r for r in run.app.registry.records if r.call.name == "kb_get"]
        assert gets and all(
            r.citation is None
            and r.result.error is not None
            and r.result.error.code == "source_rejected"
            for r in gets
        )
    if slot == 24:
        assert receipt.decision == "unsupported" and receipt.proposals == 0


def test_new_custodian_facts_drive_actual_business_denial() -> None:
    data, gold = business_substitute()
    ticket = next(t for t in data.tickets if t.ticket_id == "KT-E01")
    changed = replace(
        data,
        invoices=tuple(
            replace(i, amount_minor=10001) if i.invoice_id == ticket.facts.invoice_id else i
            for i in data.invoices
        ),
    )
    case = gold.manifest.cases[0]
    oracle = replace(
        case.oracle,
        decision="ineligible",
        outcome="expected_denial",
        final_status="open",
        final_revision=0,
    )
    case = replace(case, oracle=oracle)
    manifest = replace(
        gold.manifest,
        corpus_identity=digest(asdict(changed)),
        cases=(case, *gold.manifest.cases[1:]),
    )
    validate_business(changed, replace(gold, version="PUBLIC-CHANGED-FACTS", manifest=manifest))
    with operator_scope("PUBLIC", manifest.identity, manifest.corpus_identity):
        run = execute_public_task(
            changed, case.task, case_scope=issue(changed, case.task.ticket_id)
        )
        receipt = grade_case(case, changed, run)
    assert receipt.status == "passed" and receipt.decision == "ineligible"
    assert not run.app.approval_calls and not run.app.workflow.store.snapshot.commits


def test_all_m2_execution_errors_cannot_become_formal_or_engineering_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import knowledge_ticket_agent.retrieval_strategies as module

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("public upstream unavailable")

    inputs, gold = m2_substitute()
    monkeypatch.setattr(module.StrategyRetriever, "search", unavailable)
    result = evaluate(inputs, gold)
    assert result["engineering_contract"] == "FAIL"
    assert result["revised_formal_quality"] == "NOT_RUN"
    assert result["recall_at_3"] == result["mrr_at_3"] == 0


def test_all_corrupted_sources_cannot_pass_the_20_evidence_gate() -> None:
    from knowledge_ticket_agent.business_evaluation import run_public

    data, gold = business_substitute()
    cases = tuple(
        replace(
            c,
            task=replace(c.task, fault="corrupt_evidence"),
            oracle=replace(c.oracle, outcome="tool_error", final_status="open", final_revision=0),
        )
        for c in gold.manifest.cases
    )
    manifest = replace(gold.manifest, cases=cases)
    validate_business(data, replace(gold, manifest=manifest))
    with operator_scope("PUBLIC", manifest.identity, manifest.corpus_identity):
        result = run_public(
            manifest,
            lambda: data,
            lambda d, t: execute_public_task(d, t, case_scope=issue(d, t.ticket_id)),
        )
    assert result.passed <= 4 and result.rec_macro == 0.0
    assert all(r.status == "failed" for r in result.receipts[:20])


def test_public_m2_pass_cannot_satisfy_formal_m5_dependency(tmp_path: Path) -> None:
    from .test_quality_closure import prepare

    operator, release, payload = prepare(tmp_path)
    raw_hash = private_file(
        operator.root / "m2-formal-result.json",
        {
            "contract": CONTRACT,
            "scope": "PUBLIC_REHEARSAL_NOT_FORMAL",
            "revised_formal_quality": "PASS",
        },
    )
    release = replace(
        release,
        m2_dependency="REVISED_FORMAL_PASS",
        m2_contract=CONTRACT,
        m2_result_sha256=raw_hash,
    )
    private_file(
        operator.root / "approval.json",
        {
            "release_identity": release.identity,
            "decision": "AUTHORIZE_LOCKED_M5_ONCE",
            "profile": "rehearsal",
            "m2_dependency": release.m2_dependency,
        },
    )
    with pytest.raises(ValueError, match="M2_formal_dependency_unverified"):
        operator.execute(release, loader=lambda: payload)
    assert operator.inspect()["state"] == "NOT_STARTED"
