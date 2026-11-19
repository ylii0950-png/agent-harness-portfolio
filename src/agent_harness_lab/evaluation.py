"""Three fixed synthetic tasks, real/mock separation, deterministic artifact grading."""

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.grading import Expected, grade
from agent_harness_lab.integration import ProviderFailure
from agent_harness_lab.oauth import CredentialStore, OAuthConnection
from agent_harness_lab.providers import MockProvider, ModelProvider, ModelResponse
from agent_harness_lab.responses import ResponsesProvider, available_models
from agent_harness_lab.runtime import Agent, RunLimits
from agent_harness_lab.tools import Permission, ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace, TraceSink
from agent_harness_lab.workspace import WorkspaceFiles

Mode = Literal["mock", "real"]


@dataclass(frozen=True, slots=True)
class EvalTask:
    task_id: str
    prompt: str
    fixtures: dict[str, str]
    expected: Expected


def tasks() -> tuple[EvalTask, ...]:
    payload = json.loads(files("agent_harness_lab").joinpath("eval_tasks.json").read_text())
    return tuple(
        EvalTask(
            row["id"],
            row["prompt"],
            row["fixtures"],
            Expected(
                row["answer"],
                row["tool"],
                row.get("error"),
                row.get("file"),
                row.get("file_content"),
                row.get("call_path"),
            ),
        )
        for row in payload
    )


def mock_provider(task: EvalTask) -> MockProvider:
    # Scripts supply decisions only. The real registry/filesystem produce tool results.
    arguments = {
        "read_code": {"path": "project.txt"},
        "write_ready": {"path": "receipt.txt", "content": "ready\n"},
        "deny_escape": {"path": "../outside.txt"},
    }
    return MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("call_1", task.expected.tool, json.dumps(arguments[task.task_id])),
                )
            ),
            task.expected.answer,
        ]
    )


def evaluate(
    output: Path,
    *,
    mode: Mode = "mock",
    model: str | None = None,
    connection: OAuthConnection | None = None,
    max_requests: int = 8,
    mock_factory: Callable[[EvalTask], ModelProvider] = mock_provider,
    task_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    if mode not in ("mock", "real"):
        raise ValueError("unknown evaluation mode")
    if type(max_requests) is not int or not 0 <= max_requests <= 100:
        raise ValueError("request budget must be an integer between 0 and 100")
    selected = set(task_ids) if task_ids is not None else {task.task_id for task in tasks()}
    if not selected or not selected <= {task.task_id for task in tasks()}:
        raise ValueError("task IDs must select from the fixed three-task set")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("evaluation output must not be a symlink")
    # Reserve report before running: never consume usage if it cannot be saved.
    report_fd = os.open(
        output / "report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    rows: list[dict[str, Any]] = []
    used = 0
    served: set[str] = set()
    blocked: str | None = None
    if mode == "real":
        try:
            connection = connection or OAuthConnection(CredentialStore())
            if model is None or model not in available_models(connection):
                blocked = "model_not_in_account_catalog"
        except ProviderFailure as exc:
            blocked = exc.info.kind.value + ":" + exc.info.code
        except (ValueError, OSError):
            blocked = "authorization_storage_unavailable"
    try:
        for task in tasks():
            if task.task_id not in selected:
                rows.append(
                    {
                        "task_id": task.task_id,
                        "status": "not_run",
                        "diagnostic": "not_selected",
                        "usage": None,
                    }
                )
                continue
            if blocked is not None or (mode == "real" and max_requests - used < 2):
                rows.append(
                    {
                        "task_id": task.task_id,
                        "status": "not_run",
                        "diagnostic": blocked or "request_budget_exhausted",
                        "usage": None,
                    }
                )
                continue
            with JSONLTrace(output / f"{task.task_id}.jsonl") as trace:
                provider: ModelProvider
                real: ResponsesProvider | None = None
                if mode == "real":
                    assert connection is not None and model is not None
                    real = ResponsesProvider(
                        connection, model, max_requests=max_requests - used, trace=trace
                    )
                    provider = real
                else:
                    provider = mock_factory(task)
                row = _execute(
                    task,
                    provider,
                    trace,
                    max_model_rounds=min(3, max_requests - used) if real else 3,
                )
                if real is not None:
                    used += real.requests
                    served.update(real.served_models)
                rows.append(row)
                if row.get("failure_kind") is not None:
                    blocked = row["failure_kind"] + ":" + row["diagnostic"]
        counts = {
            status: sum(row["status"] == status for row in rows)
            for status in ("passed", "failed", "not_run")
        }
        completed = counts["passed"] + counts["failed"]
        report: dict[str, Any] = {
            "schema_version": 1,
            "task_set_sha256": hashlib.sha256(
                files("agent_harness_lab").joinpath("eval_tasks.json").read_bytes()
            ).hexdigest(),
            "grader_version": 1,
            "mode": mode,
            "requested_model": model if mode == "real" else None,
            "served_models": sorted(served),
            "requests": used if mode == "real" else None,
            "request_budget": max_requests if mode == "real" else None,
            "counts": counts,
            "success_rate_of_run": counts["passed"] / completed if completed else None,
            "actual_cost": None,
            "cost_note": "not_provided_by_API" if mode == "real" else "not_applicable",
            "tasks": rows,
        }
        with os.fdopen(report_fd, "w", closefd=False) as target:
            json.dump(report, target, indent=2, allow_nan=False)
            target.write("\n")
        return report
    finally:
        os.close(report_fd)


def _execute(
    task: EvalTask, provider: ModelProvider, trace: TraceSink, *, max_model_rounds: int = 3
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="harness-eval-") as directory:
        base = Path(directory)
        root = base / "workspace"
        root.mkdir()
        (base / "outside.txt").write_text("synthetic outside fixture only\n")
        with WorkspaceFiles(root, allow_write=True) as workspace:
            for path, content in task.fixtures.items():
                workspace.write_text(path, content)
            registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
            register_workspace_tools(registry, workspace)
            result = Agent(
                provider, registry, limits=RunLimits(max_model_rounds, 2), trace=trace
            ).run(task.prompt)
            accepted, diagnostic = grade(result, root, task.expected)
        trace.record("eval.result", task_id=task.task_id, result="passed" if accepted else "failed")
        error = result.failure
        return {
            "task_id": task.task_id,
            "status": "passed" if accepted else "failed",
            "diagnostic": error.code if error else diagnostic,
            "failure_kind": error.kind.value if error else None,
            "http_status": error.status if error else None,
            "request_id": error.request_id if error else None,
            "stop_reason": result.stop_reason.value,
            "model_rounds": result.model_rounds,
            "tool_calls": result.tool_calls,
            "usage": [asdict(item) for item in result.usage] or None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-task synthetic mock/real evaluation")
    parser.add_argument("mode", choices=["mock", "real", "models"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument(
        "--task-id", action="append", help="Select fixed task IDs; others are not-run"
    )
    arguments = parser.parse_args()
    if arguments.mode == "models":
        try:
            print(json.dumps(available_models(OAuthConnection(CredentialStore()))))
        except ProviderFailure as exc:
            raise SystemExit(str(exc)) from None
        return
    if arguments.output is None:
        parser.error("--output is required; choose a new directory for each evaluation")
    report = evaluate(
        arguments.output,
        mode=arguments.mode,
        model=arguments.model,
        max_requests=arguments.max_requests,
        task_ids=arguments.task_id,
    )
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "counts": report["counts"],
                "requests": report["requests"],
                "served_models": report["served_models"],
            }
        )
    )


if __name__ == "__main__":
    main()
