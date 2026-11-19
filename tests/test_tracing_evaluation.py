import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.contracts import ToolCall, ToolDefinition
from agent_harness_lab.evaluation import evaluate, tasks
from agent_harness_lab.grading import Expected, grade
from agent_harness_lab.integration import FailureInfo, FailureKind, ProviderFailure, TokenUsage
from agent_harness_lab.providers import ConversationItem, MockProvider, ModelProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.tools import Argument, Permission, ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_trace_records_actual_events_without_prompts_arguments_results_or_secrets(
    tmp_path: Path,
) -> None:
    secret = "sensitive fixture credential text"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    provider = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "sensitive call identity",
                        "write_file",
                        json.dumps({"path": "file", "content": secret}),
                    ),
                ),
                usage=TokenUsage(10, 2, 12),
            ),
            secret,
        ]
    )
    path = tmp_path / "trace.jsonl"
    with JSONLTrace(path) as trace, WorkspaceFiles(workspace_root, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
        register_workspace_tools(registry, workspace)
        result = Agent(provider, registry, trace=trace).run(secret)
    rows = read_rows(path)
    assert secret not in path.read_text() and "sensitive call identity" not in path.read_text()
    assert [row["event"] for row in rows] == [
        "run.started",
        "model.request",
        "model.response",
        "tool.call",
        "tool.result",
        "model.request",
        "model.response",
        "run.stopped",
    ]
    assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[2]["total_tokens"] == 12
    assert "total_tokens" not in rows[6]  # Mock missing usage must not become a fake zero.
    assert rows[-1]["model_rounds"] == 2 and rows[-1]["tool_calls"] == 1
    assert result.usage == (TokenUsage(10, 2, 12),)
    assert path.stat().st_mode & 0o777 == 0o600


def test_trace_quota_error_is_not_a_task_or_authorization_failure(tmp_path: Path) -> None:
    provider = MockProvider(
        [
            ProviderFailure(
                FailureInfo(
                    FailureKind.QUOTA, "subscription_sharing_usage_limit_exceeded", 429, "req_test"
                )
            )
        ]
    )
    path = tmp_path / "trace.jsonl"
    with JSONLTrace(path) as trace:
        result = Agent(provider, ToolRegistry(), trace=trace).run("task")
    assert result.stop_reason == StopReason.PROVIDER_ERROR and result.failure is not None
    assert result.failure.kind == FailureKind.QUOTA
    rows = read_rows(path)
    assert rows[-2]["event"] == "model.error" and rows[-2]["status"] == 429
    assert rows[-1]["stop_reason"] == "provider_error"


def test_trace_limit_is_emitted_without_extra_tool_call(tmp_path: Path) -> None:
    provider = MockProvider([ModelResponse(tool_calls=(ToolCall("one", "read_file", "{}"),))])
    path = tmp_path / "trace.jsonl"
    with JSONLTrace(path) as trace:
        result = Agent(provider, ToolRegistry(), limits=RunLimits(1, 1), trace=trace).run("task")
    assert result.stop_reason == StopReason.MAX_MODEL_ROUNDS
    assert not any(row["event"] == "tool.call" for row in read_rows(path))
    assert read_rows(path)[-1]["stop_reason"] == "max_model_rounds"


def test_trace_file_exclusivity_symlink_and_field_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with JSONLTrace(path) as trace:
        with pytest.raises(ValueError):
            trace.record("model.response", access_token="must not be serialized")
        with pytest.raises(ValueError):
            trace.record("not.an.event")
        trace.record("model.error", error_code="Bearer credential")
    assert read_rows(path)[0]["error_code"] == "redacted"
    with pytest.raises(FileExistsError):
        JSONLTrace(path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "outside")
    with pytest.raises(FileExistsError):
        JSONLTrace(link)
    assert not (tmp_path / "outside").exists()


def test_broken_trace_fails_closed_before_provider(tmp_path: Path) -> None:
    trace = JSONLTrace(tmp_path / "trace.jsonl")
    trace.close()
    provider = MockProvider(["unused"])
    with pytest.raises(OSError):
        Agent(provider, ToolRegistry(), trace=trace).run("task")
    assert provider.calls == ()


class Interrupted:
    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        raise KeyboardInterrupt


def test_user_interruption_stops_without_more_provider_or_tool_work() -> None:
    result = Agent(Interrupted(), ToolRegistry()).run("task")
    assert result.stop_reason == StopReason.INTERRUPTED and result.failure is not None
    assert result.failure.kind == FailureKind.INTERRUPTED and result.tool_calls == 0


def test_tool_interruption_preserves_call_result_pair() -> None:
    def interrupt(arguments: Mapping[str, Argument]) -> str:
        raise KeyboardInterrupt

    registry = ToolRegistry()
    registry.register(
        ToolDefinition("interrupt", "Interrupt."), interrupt, permission=Permission.READ
    )
    provider = MockProvider(
        [ModelResponse(tool_calls=(ToolCall("one", "interrupt", "{}"),)), "unused"]
    )
    result = Agent(provider, registry).run("task")
    assert result.stop_reason == StopReason.INTERRUPTED and result.tool_calls == 1
    assert result.tool_results[0].error is not None and len(provider.calls) == 1


def test_mock_eval_three_fixed_tasks_separate_from_real_statistics(tmp_path: Path) -> None:
    report = evaluate(tmp_path / "mock")
    assert len(tasks()) == 3
    assert report["mode"] == "mock" and report["counts"] == {"passed": 3, "failed": 0, "not_run": 0}
    assert report["requested_model"] is None and report["requests"] is None
    assert report["actual_cost"] is None
    assert all(row["usage"] is None for row in report["tasks"])
    assert len(list((tmp_path / "mock").glob("*.jsonl"))) == 3
    assert json.loads((tmp_path / "mock/report.json").read_text()) == report


def test_grader_rejects_correct_answer_without_actual_tool_execution(tmp_path: Path) -> None:
    report = evaluate(
        tmp_path / "fake", mock_factory=lambda task: MockProvider([task.expected.answer])
    )
    assert report["counts"] == {"passed": 0, "failed": 3, "not_run": 0}
    assert all(row["diagnostic"] == "required_tool_not_called" for row in report["tasks"])


def test_grader_checks_file_bytes_independently_of_final_claim(tmp_path: Path) -> None:
    result = Agent(
        MockProvider([ModelResponse(tool_calls=(ToolCall("one", "write_file", "{}"),)), "SAVED"]),
        ToolRegistry(),
    ).run("task")
    assert (
        grade(
            result,
            tmp_path,
            Expected("SAVED", "write_file", file="receipt.txt", file_content="ready\n"),
        )[0]
        is False
    )
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.WRITE,))
        register_workspace_tools(registry, workspace)
        result = Agent(
            MockProvider(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "one", "write_file", '{"path":"receipt.txt","content":"wrong"}'
                            ),
                        )
                    ),
                    "SAVED",
                ]
            ),
            registry,
        ).run("task")
    assert grade(
        result,
        tmp_path,
        Expected("SAVED", "write_file", file="receipt.txt", file_content="ready\n"),
    ) == (False, "artifact_mismatch")


def test_real_eval_missing_authorization_is_not_run_not_mock_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(connection: object) -> tuple[str, ...]:
        raise ProviderFailure(FailureInfo(FailureKind.AUTHORIZATION, "not_signed_in"))

    monkeypatch.setattr("agent_harness_lab.evaluation.available_models", unavailable)
    from agent_harness_lab.oauth import CredentialStore, OAuthConnection

    connection = OAuthConnection(CredentialStore(tmp_path / "auth"))
    report = evaluate(tmp_path / "real", mode="real", model="test", connection=connection)
    assert report["counts"] == {"passed": 0, "failed": 0, "not_run": 3}
    assert report["success_rate_of_run"] is None and report["requests"] == 0
    assert not list((tmp_path / "real").glob("*.jsonl"))


def test_eval_refuses_overwrite_before_provider_calls(tmp_path: Path) -> None:
    output = tmp_path / "eval"
    evaluate(output)

    def should_not_run(task: object) -> ModelProvider:
        raise AssertionError("must not consume requests")

    with pytest.raises(FileExistsError):
        evaluate(output, mock_factory=should_not_run)


def test_fixture_paths_are_explicit_and_grader_rejects_different_denial_path(
    tmp_path: Path,
) -> None:
    assert [task.expected.call_path for task in tasks()] == [
        "project.txt",
        "receipt.txt",
        "../outside.txt",
    ]
    with WorkspaceFiles(tmp_path) as workspace:
        registry = ToolRegistry()
        register_workspace_tools(registry, workspace)
        result = Agent(
            MockProvider(
                [
                    ModelResponse(
                        tool_calls=(ToolCall("one", "read_file", '{"path":".git/config"}'),)
                    ),
                    "DENIED",
                ]
            ),
            registry,
        ).run("task")
    assert grade(result, tmp_path, tasks()[-1].expected) == (False, "tool_path_mismatch")


def test_selected_task_evaluation_marks_others_not_run(tmp_path: Path) -> None:
    report = evaluate(tmp_path / "selected", task_ids=["deny_escape"])
    assert report["counts"] == {"passed": 1, "failed": 0, "not_run": 2}
    assert report["tasks"][2]["status"] == "passed"
    assert report["tasks"][0]["diagnostic"] == "not_selected"
    assert report["grader_version"] == 1 and len(report["task_set_sha256"]) == 64


def test_invalid_task_selection_rejected_before_provider_or_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate(tmp_path / "bad", task_ids=["not_a_task"])
    assert not (tmp_path / "bad").exists()


def test_bad_usage_metadata_is_model_protocol_error_before_tools() -> None:
    provider = MockProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("one", "unknown", "{}"),), usage=cast(Any, "malformed")
            )
        ]
    )
    result = Agent(provider, ToolRegistry()).run("task")
    assert result.stop_reason == StopReason.MODEL_PROTOCOL_ERROR and result.tool_calls == 0
