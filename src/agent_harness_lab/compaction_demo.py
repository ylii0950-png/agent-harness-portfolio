"""Offline, deterministic multi-step execution with verified context/state checks."""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from agent_harness_lab.context import ContextPolicy
from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.permissions import Decision, Permission, PermissionEngine
from agent_harness_lab.providers import ConversationItem, Message, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.skill_tools import register_skill_tools
from agent_harness_lab.skills import SkillRegistry
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles

_TASK = "Complete synthetic task A explicitly. Never write a workspace file."


def _call(identity: str, tool: str, **arguments: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(identity, tool, json.dumps(arguments)),))


class _CheckedMock(MockProvider):
    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        response = super().complete(messages, tools=tools)
        if len(self.calls) == 10:
            assert messages[0] == Message("user", _TASK)
            results = [m for m in messages if isinstance(m, ToolResult)]
            skill = next(r for r in results if r.call_id == "skill")
            assert json.loads(skill.content)["trust"] == "low"
            assert json.loads(skill.content)["content"] == "REFERENCE: synthetic task A only."
            assert next(r for r in results if r.call_id == "large").content.startswith(
                "[Compacted low-trust"
            )
            assert json.loads(results[-1].content)["counts"]["completed"] == 1
            assert results[-1].call_id == "query"
            return ModelResponse("Task A completed; constraints and low-trust skill retained.")
        return response


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="harness-compaction-demo-") as temporary:
        root = Path(temporary)
        (root / "large.txt").write_text("Synthetic older details. " * 250)
        (root / "small.txt").write_text("RECENT REQUIRED EVIDENCE")
        skill = root / "skills" / "note" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            "---\nname: note\ndescription: Synthetic task reference.\n---\n"
            "REFERENCE: synthetic task A only."
        )
        provider = _CheckedMock(
            [
                _call(
                    "plan",
                    "todo_write",
                    operation="create",
                    items_json=json.dumps([{"id": "a", "description": "Complete synthetic task."}]),
                ),
                _call("skill", "load_skill", name="note"),
                _call("large", "read_file", path="large.txt"),
                _call("small", "read_file", path="small.txt"),
                _call(
                    "active", "todo_write", operation="update", task_id="a", status="in_progress"
                ),
                _call(
                    "completed", "todo_write", operation="update", task_id="a", status="completed"
                ),
                _call(
                    "denied", "write_file", path="forbidden.txt", content="must never be written"
                ),
                _call("repeat", "load_skill", name="note"),
                _call("query", "todo_read"),
                "replaced by actual-context checks",
            ]
        )
        # Todo writes are approved application state; file writes remain statically denied.
        registry = ToolRegistry(
            permission_engine=PermissionEngine(
                permissions=(Permission.READ, Permission.WRITE), rules={"write_file": Decision.DENY}
            )
        )
        path = root / "trace.jsonl"
        with WorkspaceFiles(root, allow_write=True) as workspace, JSONLTrace(path) as trace:
            register_workspace_tools(registry, workspace)
            register_todo_tools(registry)
            register_skill_tools(registry, SkillRegistry(workspace, ("skills",)))
            result = Agent(
                provider,
                registry,
                limits=RunLimits(10, 9),
                trace=trace,
                context_policy=ContextPolicy(24000, 6000, 1),
            ).run(_TASK)
            reductions = [s for s in result.context_stats if s.estimated_after < s.estimated_before]
            denied = next(r for r in result.tool_results if r.call_id == "denied")
            assert denied.error is not None and denied.error.code == "permission_denied"
            assert not (root / "forbidden.txt").exists()
            assert json.loads(result.tool_results[-2].content)["status"] == "already_loaded"
            assert result.stop_reason == StopReason.COMPLETED and reductions
            assert result.usage == ()
            trace_text = path.read_text()
            assert "Synthetic older details" not in trace_text and "REFERENCE:" not in trace_text
            assert (result.model_rounds, result.tool_calls) == (10, 9)
        for s in reductions:
            print(
                f"COMPACTION estimated_tokens={s.estimated_before}->{s.estimated_after} "
                f"replaced_results={s.compacted_results} preview_chars={s.summary_chars}"
            )
        print("ESTIMATOR serialized ASCII JSON bytes + 256; 1 byte = 1 estimated token")
        print(
            "STATE task_exact=true todo_completed=1 skill_low_trust=true repeat_body_omitted=true"
        )
        print("PERMISSION file_write=denied actual_file_absent=true")
        print("USAGE actual_tokens=unknown actual_cost=unknown; model_requests=0 (Mock only)")
        print(f"FINAL {result.final_answer}")
        print(
            f"STOP {result.stop_reason.value} (model_rounds={result.model_rounds}, "
            f"tool_calls={result.tool_calls})"
        )


if __name__ == "__main__":
    main()
