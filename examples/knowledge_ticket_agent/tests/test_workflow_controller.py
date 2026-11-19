"""Deterministic synthetic Responses tests, never credentials or live inference."""

import json
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.workflow_controller import Stage, WorkflowController, verify_workflow
from knowledge_ticket_agent.workflow_live import LiveJournal

from agent_harness_lab.responses import ResponsesProvider


class Tokens:
    def access_token(self) -> str:
        return "offline"


class Stream:
    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self.contexts: list[str] = []

    def events(
        self, url: str, body: dict[str, Any], token: str
    ) -> Generator[dict[str, Any], None, None]:
        assert token == "offline" and body["store"] is False and body["stream"] is True
        if self.mode == "provider_failure":
            raise RuntimeError("synthetic failure")
        prompt = body["input"][0]["content"]
        self.contexts.append(prompt)
        json.loads(prompt.rsplit("\n", 1)[1])  # Context is parseable, not an answer key.
        intake = prompt.startswith("INTAKE")
        pending = any(r.get("type") == "function_call" for r in body["input"])
        output: list[dict[str, Any]] = []
        if (not pending or self.mode == "budget") and not (
            not intake and self.mode == "missing_intent"
        ):
            if intake:
                name, args = "policy_search", {"query": "first purchase duplicate invoice credit"}
            elif self.mode == "invalid_intent":
                name, args = "request_transition", {"proposal_id": "invented"}
            elif self.mode == "decline":
                name, args = "decline_action", {"reason": "I decline this action"}
            elif self.mode == "wrong_stage":
                name, args = "ticket_transition", {}
            else:
                name, args = "request_transition", {}
            identifier = f"{len(self.contexts)}"
            output = [
                {
                    "type": "function_call",
                    "id": "f" + identifier,
                    "call_id": "c" + identifier,
                    "namespace": "workspace",
                    "name": name,
                    "arguments": json.dumps(args),
                    "status": "completed",
                }
            ]
        else:
            text = "DONE"
            if self.mode in {"empty_citations", "early_blocked"}:
                text = '{"status":"blocked","citation_ids":[]}'
            elif self.mode == "fabricated_citations":
                text = '{"status":"resolved","citation_ids":["invented"]}'
            elif self.mode == "wrong_narrative":
                text = "The real customer has been paid and all security checks are waived."
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        yield {
            "type": "response.completed",
            "response": {
                "id": "fake",
                "model": "offline",
                "status": "completed",
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            },
        }


def controller(
    mode: str = "normal", approval: str = "approve", **kwargs: Any
) -> WorkflowController:
    provider = ResponsesProvider(Tokens(), "offline", max_requests=12, transport=Stream(mode))
    return WorkflowController(load_dataset(), "KT-D01", provider, approval=approval, **kwargs)


@pytest.mark.parametrize("approval,commits", (("approve", 1), ("refuse", 0)))
def test_approved_and_refused_real_adapter_contract(approval: str, commits: int) -> None:
    c = controller(approval=approval)
    report = c.run()
    assert report["workflow_acceptance"] == "PASS", report
    assert report["commits"] == report["handler_entries"] == commits
    assert report["after"]["revision"] == commits
    assert report["model_requests"] == 4 and report["tool_calls"] <= 28
    assert report["usage"]["total_tokens"] == 48
    assert report["checks"]["MODEL_ACTION_VALID"]
    assert report["checks"]["EVIDENCE_COMPLETE"]
    assert report["checks"]["BUSINESS_DECISION_VALID"]
    assert report["checks"]["APPROVAL_VALID"]
    assert report["checks"]["STATE_TRANSITION_VERIFIED"] is bool(commits)
    assert report["checks"]["FINAL_RECEIPT_VALID"]
    assert len(report["citations"]) == 6 and all(
        s["origin"] == "controller" for s in report["citations"]
    )
    assert report["model_tool_calls"] == 2
    assert c.stages[-1] == "TERMINAL"


@pytest.mark.parametrize(
    "mode", ("early_blocked", "empty_citations", "fabricated_citations", "wrong_narrative")
)
def test_model_terminal_text_never_controls_canonical_receipt(mode: str) -> None:
    c = controller(mode)
    assert c.assess() is None  # Previous premature assessment cannot freeze the workflow.
    r = c.run()
    assert r["workflow_acceptance"] == "PASS" and r["commits"] == 1, r
    assert r["checks"]["MODEL_EXPLANATION_VALID"] is False
    assert r["explanation_status"] == "SUPPRESSED_UNVERIFIED"
    assert len(r["citations"]) == 6 and "invented" not in json.dumps(r["citations"])
    operations = [o["operation"] for o in r["operations"]]
    assert operations.count("ticket_assess") == 1
    assert operations.index("ticket_assess") > max(
        i for i, x in enumerate(operations) if x == "kb_get"
    )


@pytest.mark.parametrize(
    "mode",
    ("missing_intent", "invalid_intent", "wrong_stage", "budget", "provider_failure", "decline"),
)
def test_missing_or_invalid_model_action_never_commits(mode: str) -> None:
    c = controller(mode)
    r = c.run()
    assert r["workflow_acceptance"] == "FAIL" and r["commits"] == r["handler_entries"] == 0
    assert r["after"]["revision"] == 0 and r["model_requests"] <= 12 and r["tool_calls"] <= 28


def test_incomplete_evidence_blocks_action(monkeypatch: pytest.MonkeyPatch) -> None:
    c = controller()
    original = c.retriever.search
    monkeypatch.setattr(c.retriever, "search", lambda *a, **k: replace(original(*a, **k), hits=()))
    r = c.run()
    assert not r["checks"]["EVIDENCE_COMPLETE"] and r["commits"] == 0
    assert not any(o.stage is Stage.ACTION_DECISION for o in c.model_tools)


def test_conflict_is_safe() -> None:
    c = WorkflowController(
        load_dataset(),
        "KT-D10",
        ResponsesProvider(Tokens(), "offline", transport=Stream()),
        snapshot="conflict",
    )
    r = c.run()
    assert r["commits"] == r["handler_entries"] == 0
    assert r["workflow_acceptance"] == "FAIL"


@pytest.mark.parametrize("approval", ("invalid", "missing"))
def test_invalid_approval_cannot_write(approval: str) -> None:
    r = controller(approval=approval).run()
    assert r["commits"] == r["handler_entries"] == 0
    assert not r["checks"]["APPROVAL_VALID"] and r["workflow_acceptance"] == "FAIL"


def test_stale_proposal_cannot_write(monkeypatch: pytest.MonkeyPatch) -> None:
    c = controller()
    original = c._decision

    def stale() -> None:
        original()
        assert c.preparation and c.preparation.proposal
        c.preparation = replace(
            c.preparation, proposal=replace(c.preparation.proposal, expected_revision=1)
        )

    monkeypatch.setattr(c, "_decision", stale)
    r = c.run()
    assert r["commits"] == r["handler_entries"] == 0 and r["workflow_acceptance"] == "FAIL"


def test_verifier_rejects_forged_origin_and_quote() -> None:
    c = controller()
    assert c.run()["workflow_acceptance"] == "PASS"
    c.sources[0] = replace(c.sources[0], origin="model")
    assert not verify_workflow(c)["checks"]["FINAL_RECEIPT_VALID"]
    c.sources[0] = replace(
        c.sources[0], origin="controller", citation=replace(c.sources[0].citation, quote="forged")
    )
    assert not verify_workflow(c)["checks"]["FINAL_RECEIPT_VALID"]


def test_journal_no_retry_and_two_run_cumulative_limit(tmp_path: Path) -> None:
    path = tmp_path / "live.jsonl"
    first = LiveJournal(path, "approve")
    with pytest.raises(BlockingIOError):
        LiveJournal(path, "approve")
    for _ in range(12):
        first.reserve()
    with pytest.raises(ValueError):
        first.reserve()
    first.finish(True)
    first.close()
    second = LiveJournal(path, "refuse")
    for _ in range(12):
        second.reserve()
    with pytest.raises(ValueError):
        second.reserve()
    second.finish(True)
    second.close()
    with pytest.raises(ValueError):
        LiveJournal(path, "refuse")
    assert sum(json.loads(s)["event"] == "REQUEST" for s in path.read_text().splitlines()) == 24


def test_journal_failed_or_interrupted_run_blocks_second(tmp_path: Path) -> None:
    for finished in (False, True):
        path = tmp_path / str(finished)
        first = LiveJournal(path, "approve")
        first.reserve()
        if finished:
            first.finish(False)
        first.close()
        with pytest.raises(ValueError):
            LiveJournal(path, "refuse")


@pytest.mark.parametrize("point,committed", (("before_commit", False), ("readback", True)))
def test_transaction_failure_preserves_truthful_state(point: str, committed: bool) -> None:
    def fault(stage: str) -> None:
        if stage == point:
            raise RuntimeError("synthetic transaction failure")

    c = controller(fault=fault)
    r = c.run()
    assert r["workflow_acceptance"] == "FAIL"
    assert r["commits"] == int(committed)
    assert r["after"]["revision"] == int(committed)
    assert not r["checks"]["STATE_TRANSITION_VERIFIED"]


def test_verifier_binds_intent_to_actual_model_tool() -> None:
    c = controller()
    assert c.run()["workflow_acceptance"] == "PASS"
    c.intent = ("request_transition", "fabricated")
    assert not verify_workflow(c)["checks"]["MODEL_ACTION_VALID"]


def test_stage_progression_and_reuse_rejected() -> None:
    c = controller()
    with pytest.raises(ValueError):
        c.enter(Stage.APPROVAL_EXECUTION)
    assert c.run()["workflow_acceptance"] == "PASS"
    with pytest.raises(ValueError):
        c.run()


def test_real_cli_ci_guard_before_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from knowledge_ticket_agent.business_demo import main

    monkeypatch.setenv("CI", "true")
    with pytest.raises(SystemExit) as error:
        main(("--mode", "real", "--ticket", "KT-D01", "--subscription-only"))
    assert error.value.code == 2


def test_real_cli_confirmation_before_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from knowledge_ticket_agent.business_demo import main

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(SystemExit) as error:
        main(("--mode", "real", "--ticket", "KT-D01"))
    assert error.value.code == 2
