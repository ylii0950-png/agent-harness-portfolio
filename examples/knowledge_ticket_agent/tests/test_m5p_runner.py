"""Public runner, prospective artificial scope and persistent gate boundaries."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.agent_grading import grade_run
from knowledge_ticket_agent.business_evaluation import (
    PUBLIC_SUITE,
    empty_receipt,
    grade_case,
    public_manifest,
    report,
    run_public,
    seal,
    validate_receipt,
)
from knowledge_ticket_agent.evaluation_preflight import (
    FORMAL_CASE_IDS,
    FORMAL_DENOMINATORS,
    PublicRunLedger,
    current_revision,
    formal_preflight,
    lock_public,
    run_formal,
)
from knowledge_ticket_agent.evaluation_runtime import (
    CaseInput,
    FeedbackMockProvider,
    execute_public_task,
)
from knowledge_ticket_agent.evaluation_scope import (
    PROBE_VERSION,
    ArtificialCaseScope,
    public_eval_probe,
    validate_scope,
)
from knowledge_ticket_agent.schemas import Dataset

from .m5p_public_oracle import PUBLIC_DEV_EXPECTATIONS


@pytest.fixture(scope="module")
def public_data() -> Dataset:
    return load_dataset()


def test_public_manifest_matches_independent_dev_literals(public_data: Dataset) -> None:
    manifest = public_manifest(public_data)
    assert len(manifest.cases) == 16
    for case, expected in zip(manifest.cases[:12], PUBLIC_DEV_EXPECTATIONS, strict=True):
        assert case.task.ticket_id == expected.ticket_id
        assert case.task.snapshot == expected.snapshot
        assert case.oracle.category == expected.expected_category.value
        assert case.oracle.decision == expected.expected_decision.value
        assert case.oracle.outcome == expected.expected_outcome
        assert (case.oracle.final_status, case.oracle.final_revision) == (
            expected.final_status.value,
            expected.final_revision,
        )


def test_full_public_rehearsal_counts_failures_and_isolates_runs(public_data: Dataset) -> None:
    manifest = public_manifest(public_data)
    result = run_public(manifest, lambda: public_data)
    assert result.case_denominator == 16 and result.passed == 13 and not result.blocker
    assert [r.status for r in result.receipts[12:]] == [
        "passed",
        "quota_error",
        "timeout",
        "not_run",
    ]
    assert result.rec_denominator == 16 and result.rec_macro == 13 / 16
    assert result.retrieval_denominator == 12
    assert len({r.run_id for r in result.receipts if r.run_id is not None}) == 15
    assert all(r.api_usage is None and r.actual_cost is None for r in result.receipts)
    assert result.receipts[-1].integration is None and result.receipts[-1].rec == 0
    for receipt in result.receipts:
        validate_receipt(receipt)


def test_correct_business_outcome_without_matching_oracle_is_not_pass(public_data: Dataset) -> None:
    case = public_manifest(public_data).cases[0]
    run = execute_public_task(public_data, case.task)
    wrong = replace(case, oracle=replace(case.oracle, decision="PO_SCORER_ONLY_NOT_MODEL"))
    assert grade_case(wrong, public_data, run).status == "failed"
    assert "PO_SCORER_ONLY_NOT_MODEL" not in repr(run.provider.calls)
    assert isinstance(run.provider, FeedbackMockProvider)
    assert set(asdict(run.provider.task)) == {"ticket_id", "source_queries"}


def test_retrieval_stream_survives_agent_setup_error(public_data: Dataset) -> None:
    manifest = public_manifest(public_data)
    first = replace(manifest.cases[0], task=replace(manifest.cases[0].task, fault="infrastructure"))
    result = run_public(replace(manifest, cases=(first,) + manifest.cases[1:]), lambda: public_data)
    assert result.receipts[0].status == "infrastructure_error"
    assert result.receipts[0].integration is None and result.receipts[0].retrieval is not None
    assert result.retrieval_denominator == 12 and result.case_denominator == 16


def test_generic_feedback_provider_handles_real_compaction(public_data: Dataset) -> None:
    case = public_manifest(public_data).cases[0]
    run = execute_public_task(public_data, replace(case.task, compaction=True))
    assert grade_run(run).outcome == "business_success"
    assert any(s.compacted_results > 0 for s in run.result.context_stats)
    assert run.result.total_model_calls <= 16 and run.result.total_tool_calls <= 28


def test_artificial_scope_only_and_single_case_store(public_data: Dataset) -> None:
    probe = public_eval_probe(public_data)
    assert probe.scope.version == PROBE_VERSION
    ticket = next(t for t in probe.data.tickets if t.ticket_id == probe.scope.ticket_id)
    assert "NOT A SEALED CASE" in ticket.subject
    task = CaseInput(ticket.ticket_id, ticket.body)
    with pytest.raises(ValueError):
        execute_public_task(probe.data, task)  # Artificial input WITHOUT scope still denied.
    run = execute_public_task(probe.data, task, case_scope=probe.scope)
    assert grade_run(run).outcome == "business_success"
    assert len(run.app.workflow.store.snapshot.tickets) == 1
    assert run.app.workflow.store.read(ticket.ticket_id).revision == 1
    assert validate_scope(probe.data, ticket.ticket_id, probe.scope)
    assert not any(
        "case_scope" in p.name or "Gold" in p.name
        for d in run.provider.tool_catalogs[0]
        for p in d.parameters
    )
    with pytest.raises(FrozenInstanceError):
        probe.scope.ticket_id = "KT-E02"  # type: ignore[misc]


def test_scope_forgery_version_data_target_query_fail_closed(public_data: Dataset) -> None:
    probe = public_eval_probe(public_data)
    with pytest.raises(ValueError, match="scope_issuer_disabled"):
        ArtificialCaseScope("KT-E01", "0" * 64)
    with pytest.raises(ValueError):
        validate_scope(probe.data, "KT-E02", probe.scope)
    with pytest.raises(ValueError):
        validate_scope(
            replace(probe.data, manifest_sha256="0" * 64), probe.scope.ticket_id, probe.scope
        )
    with pytest.raises(ValueError):
        execute_public_task(
            probe.data,
            CaseInput(probe.scope.ticket_id, "arbitrary replacement query"),
            case_scope=probe.scope,
        )


def test_formal_refuses_before_payload_or_path_access() -> None:
    calls = []

    def forbidden() -> object:
        calls.append("payload")
        raise AssertionError("must not load")

    with pytest.raises(PermissionError, match="M5_FORMAL_DISABLED"):
        run_formal(release=cast(Any, object()), payload_loader=forbidden)
    assert not calls and len(FORMAL_CASE_IDS) == 24
    assert dict(FORMAL_DENOMINATORS) == {
        "retrieval_positive": 18,
        "required_evidence": 20,
        "business": 24,
    }
    assert dict(formal_preflight())["formal_24_execution"] == "NOT_RUN"


def test_public_ledger_started_before_loader_and_immutable_completion(
    public_data: Dataset, tmp_path: Path
) -> None:
    manifest = public_manifest(public_data)
    ledger = PublicRunLedger(tmp_path / "control")
    lock = lock_public(manifest, current_revision())

    def load() -> Dataset:
        assert (ledger.root / PUBLIC_SUITE / "STARTED.json").is_file()
        return public_data

    result = ledger.execute(lock, manifest, load)
    claim = ledger.root / PUBLIC_SUITE
    assert result.passed == 13 and ledger.inspect()["state"] == "COMPLETED"
    assert len(tuple((claim / "case-receipts").iterdir())) == 16
    before = (claim / "report.json").read_bytes()
    with pytest.raises(PermissionError, match="already_claimed"):
        ledger.execute(lock, manifest, load)
    assert before == (claim / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        ledger._write(claim / "report.json", {"replacement": "forbidden"})


def test_failure_retains_all_rows_and_cannot_restart(public_data: Dataset, tmp_path: Path) -> None:
    manifest = public_manifest(public_data)
    ledger = PublicRunLedger(tmp_path / "control")

    def interrupted() -> Dataset:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        ledger.execute(lock_public(manifest, current_revision()), manifest, interrupted)
    assert ledger.inspect()["state"] == "FAILED"
    partial = json.loads((ledger.root / PUBLIC_SUITE / "partial-report.json").read_text())
    assert partial["case_denominator"] == 16 and len(partial["receipts"]) == 16
    assert all(r["status"] == "not_run" for r in partial["receipts"])
    with pytest.raises(PermissionError):
        ledger.execute(lock_public(manifest, current_revision()), manifest, lambda: public_data)


def test_concurrent_gate_and_interrupted_claim(public_data: Dataset, tmp_path: Path) -> None:
    manifest = public_manifest(public_data)
    ledger = PublicRunLedger(tmp_path / "control")
    lock = lock_public(manifest, current_revision())
    entered, leave = threading.Event(), threading.Event()

    def held() -> Dataset:
        entered.set()
        assert leave.wait(5)
        raise OSError("public fixture ends before Agent")

    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(ledger.execute, lock, manifest, held)
        assert entered.wait(5)
        try:
            with pytest.raises((BlockingIOError, PermissionError)):
                ledger.execute(lock, manifest, lambda: public_data)
        finally:
            leave.set()
        with pytest.raises(OSError):
            task.result()
    assert ledger.inspect()["retry_allowed"] is False


@pytest.mark.parametrize("field", ("commit", "source_digests", "manifest_identity", "budget"))
def test_version_mismatch_precedes_start(public_data: Dataset, tmp_path: Path, field: str) -> None:
    manifest = public_manifest(public_data)
    lock = lock_public(manifest, current_revision())
    values: dict[str, Any] = {
        "commit": "0" * 40,
        "source_digests": (),
        "manifest_identity": "0" * 64,
        "budget": (1,),
    }
    ledger = PublicRunLedger(tmp_path / "control")
    with pytest.raises(ValueError):
        ledger.execute(replace(lock, **{field: values[field]}), manifest, lambda: public_data)
    assert not ledger.root.exists()


def test_receipt_tampering_and_no_fake_api_zero(public_data: Dataset, tmp_path: Path) -> None:
    case = public_manifest(public_data).cases[0]
    receipt = empty_receipt(case, "not_run", "public_fixture")
    with pytest.raises(ValueError):
        validate_receipt(replace(receipt, status="passed"))
    with pytest.raises(ValueError):
        validate_receipt(replace(receipt, api_usage=cast(Any, 0)))
    # Lexical rejection happens before even an is_symlink/stat of a private-like path.
    with pytest.raises(ValueError, match="private_M2_storage_forbidden"):
        PublicRunLedger(tmp_path / ".git" / "m2f-formal-private")


def test_ancestor_symlink_never_redirects_ledger(public_data: Dataset, tmp_path: Path) -> None:
    fake_private = tmp_path / ".git" / "m2-fake-vault"
    fake_private.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.mkdir()
    ledger = PublicRunLedger(alias / "child")
    alias.rmdir()
    alias.symlink_to(fake_private, target_is_directory=True)
    assert ledger.inspect()["state"] == "CORRUPT_OR_INCOMPLETE"
    manifest = public_manifest(public_data)
    with pytest.raises(ValueError, match="symlink_ancestor"):
        ledger.execute(lock_public(manifest, current_revision()), manifest, lambda: public_data)
    with pytest.raises(ValueError, match="symlink_ancestor"):
        PublicRunLedger(alias / "other")
    assert not (fake_private / "child").exists() and not (fake_private / "other").exists()


@pytest.mark.parametrize(
    "change",
    (
        "missing_rec",
        "wrong_category",
        "wrong_tenant",
        "readback_false",
        "over_budget",
        "nan_metric",
    ),
)
def test_resealed_semantic_contradiction_is_rejected(public_data: Dataset, change: str) -> None:
    manifest = public_manifest(public_data)
    case = manifest.cases[0]
    receipt = grade_case(case, public_data, execute_public_task(public_data, case.task))
    assert receipt.status == "passed"
    if change == "missing_rec":
        wrong = seal(replace(receipt, rec=None))
    elif change == "wrong_category":
        wrong = seal(replace(receipt, category="data_export"))
    elif change == "wrong_tenant":
        wrong = seal(replace(receipt, tenant_id="SYN-TENANT-01"))
    elif change == "readback_false":
        assert receipt.integration is not None
        wrong = seal(
            replace(receipt, integration=replace(receipt.integration, readback_verified=False))
        )
    elif change == "over_budget":
        assert receipt.integration is not None
        wrong = seal(
            replace(receipt, integration=replace(receipt.integration, parent_model_calls=13))
        )
    else:
        with pytest.raises(ValueError):
            seal(replace(receipt, rec=float("nan")))
        return
    with pytest.raises(ValueError):
        validate_receipt(wrong, case)
    rows = (wrong,) + tuple(empty_receipt(c, "not_run", "fixture") for c in manifest.cases[1:])
    with pytest.raises(ValueError):
        report(manifest, rows)


def test_case_insensitive_private_path_alias_is_rejected_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private_M2_storage_forbidden"):
        PublicRunLedger(tmp_path / ".GIT" / "M2-FAKE" / "child")


def test_malformed_executor_still_yields_all_sixteen_rows(public_data: Dataset) -> None:
    manifest = public_manifest(public_data)
    result = run_public(
        manifest, lambda: public_data, executor=cast(Any, lambda data, task: object())
    )
    assert len(result.receipts) == result.case_denominator == 16 and result.blocker
    assert result.receipts[0].status == "infrastructure_error"
    assert all(r.status == "not_run" for r in result.receipts[1:])


def test_inspector_rejects_false_or_falsy_terminal_state(
    public_data: Dataset, tmp_path: Path
) -> None:
    manifest = public_manifest(public_data)
    ledger = PublicRunLedger(tmp_path / "control")

    def failure() -> Dataset:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        ledger.execute(lock_public(manifest, current_revision()), manifest, failure)
    claim = ledger.root / PUBLIC_SUITE
    failed = json.loads((claim / "FAILED.json").read_text())
    assert ledger.inspect()["state"] == "FAILED"
    (claim / "FAILED.json").write_text(json.dumps({**failed, "state": "COMPLETED"}))
    assert ledger.inspect()["state"] == "CORRUPT_OR_INCOMPLETE"
    (claim / "FAILED.json").write_text("{}")
    (claim / "COMPLETED.json").write_text(
        json.dumps(
            {
                "state": "COMPLETED",
                "run_id": failed["run_id"],
                "report_sha256": "0" * 64,
                "blocker": False,
            }
        )
    )
    assert ledger.inspect()["state"] == "CORRUPT_OR_INCOMPLETE"


def test_bool_actual_revision_cannot_masquerade_as_integer_zero(public_data: Dataset) -> None:
    case = public_manifest(public_data).cases[12]
    run = execute_public_task(public_data, case.task)
    assert grade_case(case, public_data, run).status == "passed"
    snapshot = run.app.workflow.store.snapshot
    run.app.workflow.store._snapshot = replace(
        snapshot,
        tickets=tuple(
            replace(t, revision=cast(Any, False)) if t.ticket_id == case.task.ticket_id else t
            for t in snapshot.tickets
        ),
    )
    receipt = grade_case(case, public_data, run)
    assert receipt.status == "failed" and receipt.blocker
    assert "malformed_state_or_observation_type" in receipt.errors


def test_partial_active_case_is_retained_as_unknown_effect(
    public_data: Dataset, tmp_path: Path
) -> None:
    manifest = public_manifest(public_data)
    ledger = PublicRunLedger(tmp_path / "control")
    calls = 0

    def interrupted() -> Dataset:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return public_data

    with pytest.raises(KeyboardInterrupt):
        ledger.execute(lock_public(manifest, current_revision()), manifest, interrupted)
    partial = json.loads((ledger.root / PUBLIC_SUITE / "partial-report.json").read_text())
    assert partial["case_denominator"] == 16 and partial["blocker"]
    assert partial["receipts"][0]["status"] == "infrastructure_error"
    assert partial["receipts"][0]["integration"] is None
    assert all(r["status"] == "not_run" for r in partial["receipts"][1:])
