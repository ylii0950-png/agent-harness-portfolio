"""Offline Parent -> Child -> structured result -> Parent receipt demonstration."""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.execution import TotalLimits
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission, PermissionEngine
from agent_harness_lab.providers import ConversationItem, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.subagents import register_subagent_tool
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def _call(identity: str, name: str, **arguments: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(identity, name, json.dumps(arguments)),))


class _ParentMock(MockProvider):
    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        # Prove that Parent inspects the actual Child result before its next action.
        if len(self.calls) == 1:
            output = messages[-1]
            if not isinstance(output, ToolResult) or not output.ok:
                raise RuntimeError("missing child result")
            child = json.loads(output.content)
            if child["status"] != "completed" or child["result"] != "READY":
                raise RuntimeError("unexpected child outcome")
        return super().complete(messages, tools=tools)


def main() -> None:
    approvals: list[str] = []

    def approve(request: ApprovalRequest) -> bool:
        approvals.append(request.call_id)
        return True

    with tempfile.TemporaryDirectory(prefix="harness-subagent-demo-") as temporary:
        root = Path(temporary)
        (root / "input.txt").write_text("READY\n", encoding="utf-8")
        child = MockProvider(
            [
                _call("child_read", "read_file", path="input.txt"),
                "READY",
            ]
        )
        parent = _ParentMock(
            [
                _call(
                    "delegate",
                    "delegate_task",
                    task="Read input.txt; return its status only.",
                    context="The fixture is synthetic.",
                ),
                _call(
                    "receipt", "write_file", path="receipt.txt", content="Child verified READY\n"
                ),
                "Child status received; receipt verified.",
            ]
        )
        registry = ToolRegistry(
            permission_engine=PermissionEngine(
                permissions=(Permission.READ, Permission.WRITE),
                rules={"write_file": Decision.ASK},
                approval_handler=approve,
            )
        )
        trace_path = root / "trace.jsonl"
        with WorkspaceFiles(root, allow_write=True) as workspace, JSONLTrace(trace_path) as trace:
            register_workspace_tools(registry, workspace)
            register_subagent_tool(
                registry, lambda: child, child_tools=("read_file",), child_limits=RunLimits(3, 2)
            )
            result = Agent(
                parent,
                registry,
                limits=RunLimits(4, 3),
                total_limits=TotalLimits(6, 4),
                trace=trace,
            ).run(
                "Delegate fixture inspection, then write a receipt after checking the child result."
            )
            delegated = json.loads(result.tool_results[0].content)
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            valid = (
                result.stop_reason == StopReason.COMPLETED
                and all(r.ok for r in result.tool_results)
                and delegated["status"] == "completed"
                and delegated["stop_reason"] == "completed"
                and (delegated["model_calls"], delegated["tool_calls"]) == (2, 1)
                and (result.model_rounds, result.tool_calls) == (3, 2)
                and (result.total_model_calls, result.total_tool_calls) == (5, 3)
                and workspace.read_text("receipt.txt") == "Child verified READY\n"
                and isinstance(child.calls[1][-1], ToolResult)
                and child.calls[1][-1].content == "READY\n"
                and approvals == ["receipt"]
                and sum(e["event"] == "subagent_started" for e in events) == 1
                and sum(e["event"] == "subagent_completed" for e in events) == 1
                and sum(e["event"] == "run.stopped" for e in events) == 2
            )
        if not valid:
            raise SystemExit("Subagent demo failed its actual child/parent/file/trace checks")
        print("DELEGATE child_status=completed stop_reason=completed")
        print("CHILD read_file success=true; model_calls=2 tool_calls=1")
        print("PARENT structured_result_checked=true; continued=true")
        print("APPROVAL_SOURCE scripted Mock approval; parent receipt write used ASK")
        print("RECEIPT actual_content_verified=true")
        print("TRACE started=1 completed=1; histories omitted")
        print(f"FINAL {result.final_answer}")
        print("PARENT STOP completed (model_rounds=3, tool_calls=2)")
        print("TOTAL model_calls=5 tool_calls=3 depth=1")


if __name__ == "__main__":
    main()
