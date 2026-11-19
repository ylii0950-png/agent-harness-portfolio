"""Fixed two-run D032 local subscription smoke journal, not a formal evaluation system."""

import fcntl
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_harness_lab.oauth import CredentialStore, OAuthConnection
from agent_harness_lab.responses import ResponsesProvider, available_models
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.workflow_controller import WorkflowController


class LiveJournal:
    """Exclusive whole-run lock; durable charges; failures/interruption never grant a retry.

    Same-user local accounting is not tamper-proof isolation. No CLI path override.
    """

    def __init__(self, path: Path, approval: str) -> None:
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(self.fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise ValueError("live_journal_permissions")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if info.st_size > 65536:
                raise ValueError("live_journal_size")
            raw = os.read(self.fd, 65537).decode()
            self.rows: list[dict[str, Any]] = [json.loads(line) for line in raw.splitlines()]
            if not self.rows:
                self.append(
                    {"event": "budget", "campaign": "D032", "max_runs": 2, "max_requests": 24}
                )
            if self.rows[0] != {
                "event": "budget",
                "campaign": "D032",
                "max_runs": 2,
                "max_requests": 24,
            }:
                raise ValueError("live_journal_identity")
            starts = [r for r in self.rows if r["event"] == "STARTED"]
            finishes = [r for r in self.rows if r["event"] in {"COMPLETED", "FAILED"}]
            if (
                len(starts) >= 2
                or len(starts) != len(finishes)
                or (starts and finishes[-1]["event"] != "COMPLETED")
            ):
                raise ValueError("live_authority_consumed_or_interrupted")
            if approval != ("approve" if not starts else "refuse"):
                raise ValueError("live_run_order")
            self.run_id = "live_" + uuid4().hex
            self.append({"event": "STARTED", "run_id": self.run_id, "approval": approval})
        except BaseException:
            os.close(self.fd)
            raise

    def append(self, row: dict[str, Any]) -> None:
        payload = memoryview((json.dumps(row, sort_keys=True) + "\n").encode())
        os.lseek(self.fd, 0, os.SEEK_END)
        while payload:
            size = os.write(self.fd, payload)
            if size <= 0:
                raise OSError("live_journal_write")
            payload = payload[size:]
        os.fsync(self.fd)
        self.rows.append(row)

    def reserve(self) -> None:
        attempts = [r for r in self.rows if r["event"] == "REQUEST"]
        if len(attempts) >= 24 or sum(r["run_id"] == self.run_id for r in attempts) >= 12:
            raise ValueError("live_request_budget")
        self.append({"event": "REQUEST", "run_id": self.run_id, "ordinal": len(attempts) + 1})

    def finish(self, passed: bool) -> None:
        self.append({"event": "COMPLETED" if passed else "FAILED", "run_id": self.run_id})

    def close(self) -> None:
        os.close(self.fd)


def real_cli(
    ticket: str,
    model: str,
    approval: str,
    *,
    structured: bool,
    request_text: str | None = None,
    language: str = "auto",
) -> None:
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        raise ValueError("real_mode_disabled_in_ci")
    if request_text is not None or language != "auto":
        from knowledge_ticket_agent.bilingual import contextualize
        from knowledge_ticket_agent.strict import require

        data = load_dataset()
        text = (
            request_text
            if request_text is not None
            else next(t.body for t in data.tickets if t.ticket_id == ticket)
        )
        _, context = contextualize(data, ticket, text, language)
        require(context.status == "supported", context.status)
    common = Path(
        subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()
    ).resolve()
    journal = LiveJournal(common / "real-workflow-d032-live.jsonl", approval)
    passed = False
    try:
        connection = OAuthConnection(CredentialStore())
        session = connection.store.load()
        if session is None:
            raise ValueError("project_oauth_missing")
        session.require_plan()
        if model not in available_models(connection):
            raise ValueError("model_not_in_selected_account_catalog")
        provider = ResponsesProvider(connection, model, max_requests=12)
        report = WorkflowController(
            load_dataset(),
            ticket,
            provider,
            approval=approval,
            reserve=journal.reserve,
            request_text=request_text,
            language=language,
        ).run()
        report["model"] = model
        report["served_models"] = sorted(provider.served_models)
        report["live_run_id"] = journal.run_id
        report["billing"] = "subscription_only_operator_confirmed_extra_credits_disabled"
        passed = (
            provider.served_models == {model}
            and report["workflow_acceptance"] == "PASS"
            and (
                report["status"] == "COMPLETED"
                if approval == "approve"
                else report["approval"] == "refused"
            )
        )
        if structured:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(
                f"CANONICAL APPLICATION RECEIPT | {ticket} | "
                f"{report['status']} | {report['workflow_acceptance']}"
            )
            if report.get("request", {}).get("language") == "zh-CN":
                print(report["display"]["message"])
            print(f"Model intent: {report['model_intent']} | Approval: {report['approval']}")
            print(
                f"Requests: {report['model_requests']} | Operations: {report['tool_calls']} | "
                f"Usage: {report['usage']} | Cost: unknown"
            )
        if not passed:
            raise SystemExit(1)
    except Exception:
        print(
            json.dumps(
                {
                    "workflow_acceptance": "FAIL",
                    "code": "live_preflight_or_execution_failed",
                    "cost": None,
                }
            )
        )
        raise SystemExit(1) from None
    finally:
        try:
            journal.finish(passed)
        finally:
            journal.close()
