"""One new versioned subscription-only ASK-refusal smoke; D032 stays untouched."""

import base64
import fcntl
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_harness_lab.oauth import CredentialStore, OAuthConnection
from agent_harness_lab.responses import ResponsesProvider, available_models
from agent_harness_lab.tracing import JSONLTrace
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.evaluation_preflight import PublicRunLedger
from knowledge_ticket_agent.strict import require
from knowledge_ticket_agent.workflow_controller import WorkflowController
from knowledge_ticket_agent.workflow_live import LiveJournal

CAMPAIGN = "lqd-live-ask-refusal-1.0.0"


class RefusalJournal(LiveJournal):
    def __init__(self, path: Path) -> None:
        # A fixed canonical filename is selected by the operator. Exclusive create
        # makes a failed/interrupted/finished campaign equally non-restartable.
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(self.fd)
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o600,
                "refusal_journal_permissions",
            )
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.rows: list[dict[str, Any]] = []
            self.run_id = "refusal_" + uuid4().hex
            self.append(
                {
                    "event": "STARTED",
                    "campaign": CAMPAIGN,
                    "run_id": self.run_id,
                    "max_requests": 6,
                    "max_operations": 20,
                }
            )
            parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            os.close(self.fd)
            raise

    def reserve(self) -> None:
        used = sum(r["event"] == "REQUEST" for r in self.rows)
        require(used < 6, "refusal_request_budget")
        self.append({"event": "REQUEST", "run_id": self.run_id, "ordinal": used + 1})


def refusal_contract(c: WorkflowController, result: dict[str, Any]) -> bool:
    return bool(
        result["workflow_acceptance"] == "PASS"
        and result["status"] == "BLOCKED"
        and result["model_intent"] == "request_transition"
        and result["approval"] == "refused"
        and len(c.approvals) == 1
        and c.approvals[0][1] is False
        and result["handler_entries"] == result["commits"] == 0
        and result["before"] == result["after"]
        and all(
            result["checks"][k] is True
            for k in (
                "MODEL_ACTION_VALID",
                "EVIDENCE_COMPLETE",
                "BUSINESS_DECISION_VALID",
                "APPROVAL_VALID",
                "FINAL_RECEIPT_VALID",
                "STATE_UNCHANGED_VERIFIED",
            )
        )
        and result["request_attempts"] <= 6
        and result["tool_calls"] <= 20
    )


def execute(*, expected_account: str, subscription_only_confirmed: bool) -> dict[str, Any]:
    require(not os.getenv("CI") and not os.getenv("GITHUB_ACTIONS"), "refusal_live_ci_denied")
    require(subscription_only_confirmed is True and bool(expected_account), "billing_unconfirmed")
    connection = OAuthConnection(CredentialStore())  # Only this project's OAuth store.
    session = connection.store.load()
    require(session is not None, "project_oauth_missing")
    assert session is not None
    session.require_plan()
    # Metadata identity check only. These claims were validated during project OAuth
    # login; no raw token, private OAuth response, or claim payload is logged.
    parts = session.id_token.split(".")
    require(len(parts) == 3, "project_identity_missing")
    claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    require(
        claims.get("sub") == session.subject
        and claims.get("email", "").casefold() == expected_account.casefold(),
        "intended_account_mismatch",
    )
    model = "gpt-6-luna"
    require(model in available_models(connection), "requested_model_unavailable")
    common = Path(
        subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()
    ).resolve()
    journal = RefusalJournal(common / (CAMPAIGN + ".jsonl"))
    passed = False
    try:
        provider = ResponsesProvider(connection, model, max_requests=6)
        with JSONLTrace(common / (CAMPAIGN + ".trace.jsonl")) as trace:
            controller = WorkflowController(
                load_dataset(),
                "KT-D01",
                provider,
                approval="refuse",
                reserve=journal.reserve,
                max_requests=6,
                max_operations=20,
                refusal_verification=True,
                trace=trace,
            )
            result = controller.run()
        trace_rows = [
            json.loads(s) for s in (common / (CAMPAIGN + ".trace.jsonl")).read_text().splitlines()
        ]
        observed_calls = sum(r["event"] == "tool.call" for r in trace_rows)
        observed_results = sum(r["event"] == "tool.result" for r in trace_rows)
        observed_requests = sum(r["event"] == "model.request" for r in trace_rows)
        journal_requests = sum(r["event"] == "REQUEST" for r in journal.rows)
        expected_calls = [
            (hashlib.sha256(o.call.call_id.encode()).hexdigest()[:16], o.run_id, o.call.name)
            for o in controller.model_tools
        ]
        actual_calls = [
            (r["call_ref"], r["run_id"], r["tool_name"])
            for r in trace_rows
            if r["event"] == "tool.call"
        ]
        consistent = (
            expected_calls == actual_calls
            and observed_calls == observed_results == result["model_tool_calls"]
            and observed_requests
            == journal_requests
            == provider.requests
            == result["request_attempts"]
        )
        passed = (
            refusal_contract(controller, result)
            and consistent
            and provider.served_models == {model}
            and result["usage_complete"] is True
        )
        result.update(
            campaign=CAMPAIGN,
            served_models=sorted(provider.served_models),
            approval_callbacks=len(controller.approvals),
            trace_journal_consistent=consistent,
            verification="PASS" if passed else "NOT VERIFIED",
            cost=None,
            billing="project_subscription_scopes_and_operator_confirmed_extra_credits_disabled",
            account_identity_sha256=hashlib.sha256(session.subject.encode()).hexdigest(),
        )
        PublicRunLedger._write(common / (CAMPAIGN + ".receipt.json"), result)
        return result
    finally:
        try:
            journal.finish(passed)
        finally:
            journal.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="One-shot authorized subscription ASK-refusal smoke"
    )
    parser.add_argument("--expected-account", required=True)
    parser.add_argument("--subscription-only-confirmed", action="store_true")
    args = parser.parse_args()
    result = execute(
        expected_account=args.expected_account,
        subscription_only_confirmed=args.subscription_only_confirmed,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "campaign",
                    "verification",
                    "model_intent",
                    "approval_callbacks",
                    "approval",
                    "commits",
                    "handler_entries",
                    "request_attempts",
                    "tool_calls",
                    "usage",
                    "served_models",
                    "trace_journal_consistent",
                    "cost",
                )
            },
            indent=2,
        )
    )
    if result["verification"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
