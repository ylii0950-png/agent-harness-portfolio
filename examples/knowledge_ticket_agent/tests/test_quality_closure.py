"""Independent public checks; no real custody, original Gold or OAuth is accessed."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.business_evaluation import CaseManifest
from knowledge_ticket_agent.evaluation_matrix import MAPPINGS, manifest
from knowledge_ticket_agent.evaluation_preflight import current_revision, source_digests
from knowledge_ticket_agent.evaluation_scope import digest, validate_scope
from knowledge_ticket_agent.formal_evaluation import (
    FormalOperator,
    FormalPayload,
    FormalRelease,
)
from knowledge_ticket_agent.formal_scope import FORMAL_SUITE, issue
from knowledge_ticket_agent.quality_acceptance import evaluate
from knowledge_ticket_agent.quality_evidence import AnswerSpec, resolve_policy
from knowledge_ticket_agent.refusal_campaign import RefusalJournal, refusal_contract
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category

from .conftest import DEMO
from .test_workflow_controller import controller


def public_payload() -> FormalPayload:
    data = load_dataset()
    declared = manifest(data)
    replacements = {}
    for n, case in enumerate(declared.cases):
        source = next(t for t in data.tickets if t.ticket_id == case.task.fixture.split(":")[1])
        target = replace(
            source,
            ticket_id=case.task.ticket_id,
            split=type(source.split).EVAL,
            subject=f"Independent public operator fixture {n}",
            body=source.body + f" Public synthetic request variant {n}.",
        )
        if case.task.fault == "missing_facts":
            target = replace(target, facts=replace(target.facts, invoice_id=None))
        replacements[target.ticket_id] = target
    data = replace(data, tickets=tuple(replacements.get(t.ticket_id, t) for t in data.tickets))
    cases = tuple(
        replace(
            c,
            case_id=c.task.ticket_id,
            task=replace(
                c.task,
                fixture="sealed:normal",
                query=next(t.body for t in data.tickets if t.ticket_id == c.task.ticket_id),
            ),
        )
        for c in declared.cases
    )
    return FormalPayload(data, CaseManifest(FORMAL_SUITE, cases, digest(asdict(data))))


def prepare(tmp_path: Path) -> tuple[FormalOperator, FormalRelease, FormalPayload]:
    root = tmp_path / "operator"
    root.mkdir(mode=0o700)
    payload = public_payload()
    release = FormalRelease(
        current_revision(),
        source_digests(),
        payload.manifest.identity,
        payload.manifest.corpus_identity,
        "1" * 64,
        "2" * 64,
        "public-test-gold",
        "public-test-M2",
        "3" * 64,
        "PO_EXPERIMENTAL_EXCEPTION",
        profile="public_rehearsal",
    )
    approval = root / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "release_identity": release.identity,
                "decision": "AUTHORIZE_LOCKED_M5_ONCE",
                "profile": "rehearsal",
                "m2_dependency": release.m2_dependency,
            }
        )
    )
    approval.chmod(0o600)
    return FormalOperator(root, rehearsal=True), release, payload


def test_formal_public_24_uses_real_runner_and_preserves_raw_failures(tmp_path: Path) -> None:
    operator, release, payload = prepare(tmp_path)
    called = []

    def load() -> FormalPayload:
        assert (operator.root / "claim/STARTED.json").is_file()
        called.append(True)
        return payload

    result = operator.execute(release, loader=load)
    assert len(called) == 1
    assert len(result.receipts) == 24
    assert (result.retrieval_denominator, result.rec_denominator, result.case_denominator) == (
        18,
        20,
        24,
    )
    assert [r.status for r in result.receipts] == [m[3] for m in MAPPINGS]
    assert result.passed == 15 and not result.blocker
    assert result.relevance_profile == "PUBLIC_SYNTHETIC_FORMAL_COMPATIBILITY_NOT_FORMAL"
    assert all(
        r.unauthorized_mutations in {None, 0} and r.unauthorized_handlers in {None, 0}
        for r in result.receipts
    )
    assert operator.inspect()["state"] == "COMPLETED"
    (operator.root / "claim/report.json").write_text("{}")
    assert operator.inspect()["state"] == "CORRUPT_OR_INCOMPLETE"
    with pytest.raises(FileExistsError):
        operator.execute(release, loader=lambda: pytest.fail("repeat loader"))


@pytest.mark.parametrize(
    "mutation", ["commit", "source", "evaluator", "approval", "threshold", "payload"]
)
def test_formal_fail_closed_and_claim_order(tmp_path: Path, mutation: str) -> None:
    operator, release, payload = prepare(tmp_path)
    if mutation == "commit":
        release = replace(release, commit="0" * 40)
    if mutation == "source":
        release = replace(release, sources=())
    if mutation == "evaluator":
        release = replace(release, evaluator="unbound-evaluator")
    if mutation == "threshold":
        release = replace(release, thresholds=())
    if mutation == "approval":
        (operator.root / "approval.json").write_text("{}")

    def load() -> FormalPayload:
        assert mutation == "payload"
        return replace(payload, manifest=replace(payload.manifest, corpus_identity="4" * 64))

    with pytest.raises(ValueError):
        operator.execute(release, loader=load)
    assert operator.inspect()["state"] == ("FAILED" if mutation == "payload" else "NOT_STARTED")


def test_formal_interrupt_claim_is_permanent_and_scopes_do_not_escape(tmp_path: Path) -> None:
    operator, release, payload = prepare(tmp_path)
    with pytest.raises(ValueError):
        issue(payload.data, "KT-E01")

    def interrupt() -> FormalPayload:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        operator.execute(release, loader=interrupt)
    assert operator.inspect()["state"] == "FAILED"
    with pytest.raises(FileExistsError):
        operator.execute(release, loader=lambda: payload)
    with pytest.raises(ValueError):
        FormalOperator(tmp_path / "alternate")
    assert not validate_scope(payload.data, "KT-E01", None)


def test_refusal_success_is_actual_request_then_false_ask() -> None:
    c = controller(approval="refuse", max_requests=6, max_operations=20, refusal_verification=True)
    result = c.run()
    assert refusal_contract(c, result)
    assert len(c.approvals) == 1 and c.approvals[0][1] is False
    assert result["model_tool_calls"] == 2 and result["request_attempts"] == 4
    assert result["before"] == result["after"] and result["commits"] == 0


@pytest.mark.parametrize("mode", ["decline", "missing_intent", "budget", "wrong_stage"])
def test_refusal_decline_or_exhaustion_never_claims_pass(mode: str) -> None:
    c = controller(
        mode=mode, approval="refuse", max_requests=6, max_operations=20, refusal_verification=True
    )
    r = c.run()
    assert not refusal_contract(c, r)
    assert r["handler_entries"] == r["commits"] == 0 and r["after"] == r["before"]


def test_new_journal_no_retry_and_six_request_cap(tmp_path: Path) -> None:
    path = tmp_path / "new-campaign.jsonl"
    j = RefusalJournal(path)
    for _ in range(6):
        j.reserve()
    with pytest.raises(ValueError):
        j.reserve()
    j.close()  # Interrupted claim also forbids retry.
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        RefusalJournal(path)
    assert path.read_bytes() == before


def test_prospective_public_contract_is_frozen_and_separate() -> None:
    result = evaluate(DEMO / "verification/m2-revised/cases-v3.0.2.json")
    assert result["historical_M2"] == "FAIL_UNCHANGED"
    assert result["revised_formal_quality"] == "NOT_RUN"
    assert len(result["ranking"]) == 12 and len(result["cases"]) == 16
    assert all(r["passed"] for r in result["cases"])


def test_new_adversarial_intents_do_not_promote_documents_to_facts_or_permission() -> None:
    data = load_dataset()
    retriever = StrategyRetriever(data, config=strategy("A1"))
    for intent in ("customer_identity", "execute_refund", "approval_received"):
        r = resolve_policy(
            data,
            "KT-D01",
            "first purchase credit policy invoice",
            AnswerSpec(Category.BILLING, intent, (("POL-BILL-01@1.0", "eligibility"),)),
            retriever=retriever,
        )
        assert r.decision.status == "not_assessable" and r.full_reads == 0
    faq = resolve_policy(
        data,
        "KT-D01",
        "FAQ credit policy",
        AnswerSpec(Category.BILLING, "policy_explanation", (("FAQ-BILL-01@1.0", "answer"),)),
        retriever=retriever,
    )
    assert faq.decision.status == "not_assessable"


@pytest.mark.parametrize("a1", [False, True], ids=["legacy", "A1"])
def test_operator_real_json_loading_path_with_public_material_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    a1: bool,
) -> None:
    import hashlib

    import knowledge_ticket_agent.formal_evaluation as module
    from knowledge_ticket_agent.retrieval_profiles import A1_EVALUATOR, A1_METHOD, A1_SUITE

    operator, release, payload = prepare(tmp_path)
    if a1:
        declared = replace(
            payload.manifest,
            suite_id=A1_SUITE,
            retrieval_method=A1_METHOD,
            cases=tuple(
                replace(c, task=replace(c.task, fixture="sealed:a1/normal"))
                for c in payload.manifest.cases
            ),
        )
        payload = replace(payload, manifest=declared)
        release = replace(
            release, suite=A1_SUITE, evaluator=A1_EVALUATOR, manifest_identity=declared.identity
        )
    raw = json.dumps(asdict(payload.data)).encode()
    gold = json.dumps(
        {"version": release.gold_version, "manifest": asdict(payload.manifest)}
    ).encode()
    for name, content in (("payload.json", raw), ("gold-contract.json", gold)):
        p = operator.root / name
        p.write_bytes(content)
        p.chmod(0o600)
    release = replace(
        release,
        payload_sha256=hashlib.sha256(raw).hexdigest(),
        gold_sha256=hashlib.sha256(gold).hexdigest(),
    )
    p = operator.root / "approval.json"
    p.write_text(
        json.dumps(
            {
                "release_identity": release.identity,
                "decision": "AUTHORIZE_LOCKED_M5_ONCE",
                "profile": "rehearsal",
                "m2_dependency": release.m2_dependency,
            }
        )
    )
    lock = operator.root / "release-lock.json"
    lock.write_text(json.dumps(asdict(release)))
    lock.chmod(0o600)
    # Exercise the CLI with the real rehearsal operator and public JSON only.
    monkeypatch.setattr(module, "canonical_root", lambda purpose: operator.root)
    monkeypatch.setattr(module, "FormalOperator", lambda root, *, purpose: operator)
    monkeypatch.setattr("sys.argv", ["formal_evaluation", "--execute", "--purpose", "M5"])
    module.main()
    output = json.loads(capsys.readouterr().out)
    saved_path = operator.root / "claim/report.json"
    saved_bytes = saved_path.read_bytes()
    saved = json.loads(saved_bytes)
    assert output["evaluator"] == release.evaluator
    assert saved["retrieval_method"] == payload.manifest.retrieval_method
    assert output["release_identity"] == release.identity
    assert output["passed"] == saved["passed"] == 15
    assert saved["relevance_profile"] == "PUBLIC_SYNTHETIC_FORMAL_COMPATIBILITY_NOT_FORMAL"
    assert operator.inspect()["state"] == "COMPLETED"
    with pytest.raises(FileExistsError):
        module.main()
    assert saved_path.read_bytes() == saved_bytes


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "conflict",
        "path_probe",
        "child_write",
        "missing_document",
        "false_child",
        "readback_order",
        "injection",
        "compaction",
    ],
)
def test_versioned_fixture_modes_execute_real_tools(tmp_path: Path, mode: str) -> None:
    from knowledge_ticket_agent.evaluation_runtime import execute_public_task
    from knowledge_ticket_agent.formal_scope import operator_scope

    _, release, payload = prepare(tmp_path)
    case = (
        next(c for c in payload.manifest.cases if c.task.snapshot == "conflict")
        if mode == "conflict"
        else payload.manifest.cases[0]
    )
    task = replace(
        case.task,
        fixture="sealed:" + mode,
        fault="missing_evidence" if mode == "missing_document" else "none",
        snapshot="conflict" if mode == "conflict" else "normal",
        compaction=mode == "compaction",
    )
    with operator_scope(release.identity, payload.manifest.identity, release.corpus_identity):
        run = execute_public_task(
            payload.data, task, case_scope=issue(payload.data, task.ticket_id)
        )
    assert run.result.total_tool_calls <= 28 and run.result.total_model_calls <= 16
    if mode in {"false_child", "missing_document", "conflict"}:
        assert not run.app.workflow.store.snapshot.commits
    if mode in {"child_write", "false_child"}:
        assert len(run.children) == 1 and run.children[0].calls
    if mode == "child_write":
        attempts = [
            r for r in run.app.registry.records if r.depth > 0 and r.call.name == "write_file"
        ]
        assert attempts and all(r.handler_entries == 0 and r.before == r.after for r in attempts)
    if mode == "path_probe":
        assert any(r.call.name == "read_file" and not r.result.ok for r in run.app.registry.records)
    # Original hidden question text is not inserted into the provider prompt.
    assert task.query not in str(run.provider.calls[0])


def test_formal_concurrent_claim_and_interrupted_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import knowledge_ticket_agent.formal_evaluation as module

    operator, release, payload = prepare(tmp_path)

    def load() -> FormalPayload:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                operator.execute, release, loader=lambda: pytest.fail("racing loader")
            )
            with pytest.raises(BlockingIOError):
                future.result()
        return payload

    from knowledge_ticket_agent.evaluation_runtime import execute_public_task

    original = execute_public_task
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "execute_public_task", interrupt)
    with pytest.raises(KeyboardInterrupt):
        operator.execute(release, loader=load)
    assert operator.inspect()["state"] == "FAILED"
    partial = json.loads((operator.root / "claim/partial-report.json").read_text())
    assert len(partial["receipts"]) == 24 and partial["receipts"][0]["integration"]["commits"] == 1
    assert partial["receipts"][1]["blocker"] is True


def test_refusal_billing_guard_precedes_any_credential_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import knowledge_ticket_agent.refusal_campaign as module

    monkeypatch.setattr(module, "CredentialStore", lambda: pytest.fail("credential access"))
    with pytest.raises(ValueError):
        module.execute(
            expected_account="fictional@example.invalid", subscription_only_confirmed=False
        )


def test_new_live_adapter_end_to_end_synthetic_responses_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64
    import subprocess
    from types import SimpleNamespace

    import knowledge_ticket_agent.refusal_campaign as module

    from agent_harness_lab.responses import ResponsesProvider

    from .test_workflow_controller import Stream, Tokens

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    claims = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": "public-synthetic", "email": "public@example.invalid"}).encode()
        )
        .decode()
        .rstrip("=")
    )
    session = SimpleNamespace(
        subject="public-synthetic",
        id_token="header." + claims + ".signature",
        require_plan=lambda: None,
    )
    connection = SimpleNamespace(store=SimpleNamespace(load=lambda: session))
    monkeypatch.setattr(module, "CredentialStore", lambda: object())
    monkeypatch.setattr(module, "OAuthConnection", lambda _: connection)
    monkeypatch.setattr(module, "available_models", lambda _: ("gpt-6-luna",))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: str(tmp_path))

    class Served(Stream):
        def events(self, *args: Any, **kwargs: Any) -> Any:
            for event in super().events(*args, **kwargs):
                event["response"]["model"] = "gpt-6-luna"
                yield event

    providers = []

    def provider(_: object, model: str, *, max_requests: int) -> ResponsesProvider:
        assert model == "gpt-6-luna" and max_requests == 6
        actual = ResponsesProvider(Tokens(), model, max_requests=max_requests, transport=Served())
        providers.append(actual)
        return actual

    monkeypatch.setattr(module, "ResponsesProvider", provider)
    r = module.execute(expected_account="public@example.invalid", subscription_only_confirmed=True)
    assert r["verification"] == "PASS" and r["trace_journal_consistent"]
    assert r["approval_callbacks"] == 1 and r["approval"] == "refused"
    assert r["handler_entries"] == r["commits"] == 0 and r["before"] == r["after"]
    assert r["request_attempts"] == 4 and r["usage"]["total_tokens"] == 48
    rows = [
        json.loads(s) for s in (tmp_path / (module.CAMPAIGN + ".jsonl")).read_text().splitlines()
    ]
    assert rows[0]["event"] == "STARTED" and rows[-1]["event"] == "COMPLETED"
    assert sum(s["event"] == "REQUEST" for s in rows) == 4
    with pytest.raises(FileExistsError):
        module.execute(expected_account="public@example.invalid", subscription_only_confirmed=True)
    assert len(providers) == 1 and not list(tmp_path.glob("*d032*"))


def test_mismatched_documentary_context_abstains_before_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = load_dataset()
    retriever = StrategyRetriever(data, config=strategy("A1"))
    monkeypatch.setattr(
        retriever, "search", lambda *a, **k: pytest.fail("mismatched context searched")
    )
    result = resolve_policy(
        data,
        "KT-D01",
        "identity policy",
        AnswerSpec(Category.ACCESS, "policy_explanation", (("POL-ACCESS-01@1.0", "verification"),)),
        retriever=retriever,
    )
    assert result.decision.status == "not_assessable"
    assert result.searches == result.full_reads == result.model_rounds == result.commits == 0
