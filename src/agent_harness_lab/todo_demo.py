"""A complete Mock TodoWrite lifecycle through our Agent/Registry/permission gate."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission, PermissionEngine
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def _request(identity: str, name: str, **arguments: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(identity, name, json.dumps(arguments)),))


def main() -> None:
    approvals: list[str] = []

    def approve(request: ApprovalRequest) -> bool:
        approvals.append(request.tool_name)
        return True  # Explicit scripted fixture; never the default policy.

    provider = MockProvider(
        [
            _request(
                "create",
                "todo_write",
                operation="create",
                items_json=json.dumps(
                    [
                        {"id": "read_source", "description": "Read synthetic input"},
                        {"id": "write_receipt", "description": "Write synthetic receipt"},
                    ]
                ),
            ),
            _request("list_initial", "todo_read"),
            _request(
                "start_read",
                "todo_write",
                operation="update",
                task_id="read_source",
                status="in_progress",
            ),
            _request("read", "read_file", path="input.txt"),
            _request(
                "finish_read",
                "todo_write",
                operation="update",
                task_id="read_source",
                status="completed",
            ),
            _request(
                "start_write",
                "todo_write",
                operation="update",
                task_id="write_receipt",
                status="in_progress",
            ),
            _request("write", "write_file", path="receipt.txt", content="receipt ready\n"),
            _request(
                "finish_write",
                "todo_write",
                operation="update",
                task_id="write_receipt",
                status="completed",
            ),
            _request("list_final", "todo_read"),
            "Both planned steps completed; receipt verified.",
        ]
    )
    with TemporaryDirectory(prefix="harness-todo-demo-") as temporary:
        base = Path(temporary)
        root = base / "workspace"
        root.mkdir()
        trace_path = base / "trace.jsonl"
        with WorkspaceFiles(root, allow_write=True) as workspace, JSONLTrace(trace_path) as trace:
            workspace.write_text("input.txt", "synthetic input\n")
            policy = PermissionEngine(
                permissions=(Permission.READ, Permission.WRITE),
                rules={"todo_write": Decision.ASK, "write_file": Decision.ASK},
                approval_handler=approve,
            )
            tools = ToolRegistry(permission_engine=policy)
            register_workspace_tools(tools, workspace)
            register_todo_tools(tools)
            result = Agent(provider, tools, limits=RunLimits(10, 9), trace=trace).run(
                "Plan and execute reading input and writing a receipt."
            )
            plan = json.loads(result.tool_results[-1].content)
            valid = (
                result.stop_reason == StopReason.COMPLETED
                and result.tool_calls == 9
                and all(output.ok for output in result.tool_results)
                and plan["counts"] == {"pending": 0, "in_progress": 0, "completed": 2}
                and result.tool_results[3].content == "synthetic input\n"
                and workspace.read_text("receipt.txt") == "receipt ready\n"
                and approvals == ["todo_write"] * 4 + ["write_file", "todo_write"]
            )
        if not valid:
            raise SystemExit("Todo demo failed its actual plan/tool/file checks")
        print("APPROVAL_SOURCE scripted Mock approval; all 6 mutations used ASK")
        for output in result.tool_results:
            if output.name in ("todo_write", "todo_read"):
                payload = json.loads(output.content)
                tasks = ", ".join(task["id"] + "=" + task["status"] for task in payload["tasks"])
                print(f"{output.name} {tasks}")
            else:
                print(f"{output.name} success={str(output.ok).lower()}")
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert sum(row["event"] == "todo.created" for row in events) == 1
        assert sum(row["event"] == "todo.updated" for row in events) == 4
        print("PLAN completed=2/2; actual receipt verified=true")
        print("TRACE created=1 updated=4; descriptions omitted")
        print(f"FINAL {result.final_answer}")
        print(
            f"STOP {result.stop_reason.value} "
            f"(model_rounds={result.model_rounds}, tool_calls={result.tool_calls})"
        )


if __name__ == "__main__":
    main()
