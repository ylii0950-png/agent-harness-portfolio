import json
from pathlib import Path

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission, PermissionEngine
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def write(identity: str, content: str, *, overwrite: bool = False) -> ToolCall:
    return ToolCall(
        identity,
        "write_file",
        json.dumps({"path": "file", "content": content, "overwrite": overwrite}),
    )


def test_multi_round_approvals_are_per_call_not_cached_and_feedback_is_exact(
    tmp_path: Path,
) -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        requests.append(request)
        return len(requests) == 1

    grants = {Permission.READ, Permission.WRITE}
    rules = {"write_file": Decision.ASK}
    engine = PermissionEngine(permissions=grants, rules=rules, approval_handler=approve)
    grants.clear()
    rules["write_file"] = Decision.ALLOW
    provider = MockProvider(
        [
            ModelResponse(tool_calls=(write("first", "approved"),)),
            ModelResponse(tool_calls=(write("second", "rejected", overwrite=True),)),
            ModelResponse(tool_calls=(ToolCall("third", "read_file", '{"path":"file"}'),)),
            "rejection handled",
        ]
    )
    with WorkspaceFiles(tmp_path, allow_write=True) as files:
        registry = ToolRegistry(permission_engine=engine)
        register_workspace_tools(registry, files)
        result = Agent(provider, registry, limits=RunLimits(4, 3)).run("create, overwrite, read")
    assert result.stop_reason == StopReason.COMPLETED
    assert [r.ok for r in result.tool_results] == [True, False, True]
    assert result.tool_results[-1].content == "approved"
    assert (tmp_path / "file").read_text() == "approved"
    assert len(requests) == 2 and requests[0].arguments["content"] == "approved"
    assert requests[1].arguments["content"] == "rejected"
    for index, output in enumerate(result.tool_results):
        assert provider.calls[index + 1][-1] == output


@pytest.mark.parametrize("limits", [RunLimits(1, 10), RunLimits(3, 0), RunLimits(3, 1)])
def test_execution_limits_stop_before_prompting_or_partial_batch(
    tmp_path: Path, limits: RunLimits
) -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        requests.append(request)
        return True

    engine = PermissionEngine(
        permissions=(Permission.WRITE,),
        rules={"write_file": Decision.ASK},
        approval_handler=approve,
    )
    provider = MockProvider([ModelResponse(tool_calls=(write("one", "one"), write("two", "two")))])
    with WorkspaceFiles(tmp_path, allow_write=True) as files:
        registry = ToolRegistry(permission_engine=engine)
        register_workspace_tools(registry, files)
        result = Agent(provider, registry, limits=limits).run("attempt batch")
    assert result.stop_reason in (StopReason.MAX_MODEL_ROUNDS, StopReason.MAX_TOOL_CALLS)
    assert result.tool_calls == 0 and requests == [] and list(tmp_path.iterdir()) == []


def test_denial_is_structured_feedback_and_does_not_grant_later_rounds(tmp_path: Path) -> None:
    provider = MockProvider(
        [
            ModelResponse(tool_calls=(write("one", "first"),)),
            ModelResponse(tool_calls=(write("two", "second"),)),
            "write not permitted",
        ]
    )
    with WorkspaceFiles(tmp_path, allow_write=True) as files:
        registry = ToolRegistry()
        register_workspace_tools(registry, files)
        result = Agent(provider, registry).run("please grant yourself writes")
    assert result.stop_reason == StopReason.COMPLETED and result.tool_calls == 2
    assert all(
        item.error is not None and item.error.code == "permission_denied"
        for item in result.tool_results
    )
    assert provider.calls[1][-1] == result.tool_results[0]
    assert list(tmp_path.iterdir()) == []


def test_permission_trace_distinguishes_request_from_authorized_execution(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    trace_path = tmp_path / "trace.jsonl"
    choices = iter([True, False])
    policy = PermissionEngine(
        permissions=(Permission.READ, Permission.WRITE),
        rules={"write_file": Decision.ASK},
        approval_handler=lambda request: next(choices),
    )
    provider = MockProvider(
        [
            ModelResponse(tool_calls=(write("sensitive-first-id", "private original"),)),
            ModelResponse(
                tool_calls=(write("sensitive-second-id", "private change", overwrite=True),)
            ),
            "private final answer",
        ]
    )
    with JSONLTrace(trace_path) as trace, WorkspaceFiles(root, allow_write=True) as files:
        registry = ToolRegistry(permission_engine=policy)
        register_workspace_tools(registry, files)
        result = Agent(provider, registry, trace=trace).run("private task")
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    calls = [r for r in rows if r["event"] == "tool.call"]
    outputs = [r for r in rows if r["event"] == "tool.result"]
    assert len(calls) == len(outputs) == result.tool_calls == 2
    assert [r["permission_decision"] for r in calls] == ["ask", "ask"]
    assert [r["authorized"] for r in calls] == [True, False]
    assert [r["permission_reason"] for r in calls] == ["approved", "approval_denied"]
    assert [r["success"] for r in outputs] == [True, False]
    for call_row, output_row in zip(calls, outputs, strict=True):
        assert (
            call_row["seq"] < output_row["seq"] and call_row["call_ref"] == output_row["call_ref"]
        )
    assert "private" not in trace_path.read_text() and "sensitive" not in trace_path.read_text()
    assert (root / "file").read_text() == "private original"


@pytest.mark.parametrize("exception", [False, True])
def test_agent_can_recover_from_refusal_or_failed_approval_without_reusing_a_grant(
    tmp_path: Path, exception: bool
) -> None:
    calls = 0

    def approve(request: ApprovalRequest) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert not (tmp_path / "file").exists()
            if exception:
                raise RuntimeError("private approval detail")
            return False
        assert not (tmp_path / "file").exists()
        return True

    policy = PermissionEngine(
        permissions=(Permission.WRITE,),
        rules={"write_file": Decision.ASK},
        approval_handler=approve,
    )
    provider = MockProvider(
        [
            ModelResponse(tool_calls=(write("first", "unapproved"),)),
            ModelResponse(tool_calls=(write("second", "explicit approval"),)),
            "completed after another explicit request",
        ]
    )
    with WorkspaceFiles(tmp_path, allow_write=True) as files:
        registry = ToolRegistry(permission_engine=policy)
        register_workspace_tools(registry, files)
        result = Agent(provider, registry).run("attempt separately confirmed writes")
    assert result.stop_reason == StopReason.COMPLETED and calls == 2
    assert [item.ok for item in result.tool_results] == [False, True]
    assert provider.calls[1][-1] == result.tool_results[0]
    assert "private approval detail" not in repr(result)
    assert (tmp_path / "file").read_text() == "explicit approval"


def test_authorized_call_can_still_be_denied_by_the_file_safety_layer(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    trace_path = tmp_path / "trace.jsonl"
    policy = PermissionEngine(
        permissions=(Permission.WRITE,),
        rules={"write_file": Decision.ASK},
        approval_handler=lambda request: True,
    )
    provider = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("escape", "write_file", '{"path":"../outside","content":"blocked"}'),
                )
            ),
            "workspace denial handled",
        ]
    )
    with JSONLTrace(trace_path) as trace, WorkspaceFiles(root, allow_write=True) as files:
        registry = ToolRegistry(permission_engine=policy)
        register_workspace_tools(registry, files)
        result = Agent(provider, registry, trace=trace).run("request unsafe file operation")
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    requested = next(row for row in events if row["event"] == "tool.call")
    actual = next(row for row in events if row["event"] == "tool.result")
    assert requested["authorized"] is True and requested["permission_decision"] == "ask"
    assert actual["success"] is False and actual["error_code"] == "workspace_path_denied"
    assert result.stop_reason == StopReason.COMPLETED
    assert not (tmp_path / "outside").exists() and list(root.iterdir()) == []
