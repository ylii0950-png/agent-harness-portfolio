"""Run with: uv run --no-sync python -m agent_harness_lab.demo"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_harness_lab.contracts import ToolCall, ToolResult
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.tools import Permission, ToolRegistry, register_workspace_tools
from agent_harness_lab.workspace import WorkspaceFiles


def main() -> None:
    expected = "Hello from the Phase 1 mock agent.\n"
    provider = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "call_1",
                        "write_file",
                        json.dumps({"path": "note.txt", "content": expected}),
                    ),
                )
            ),
            ModelResponse(tool_calls=(ToolCall("call_2", "read_file", '{"path":"note.txt"}'),)),
            "Created and read note.txt successfully.",
        ]
    )
    with TemporaryDirectory(prefix="harness-demo-") as directory:
        with WorkspaceFiles(Path(directory), allow_write=True) as files:
            registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
            register_workspace_tools(registry, files)
            result = Agent(
                provider, registry, limits=RunLimits(max_model_rounds=3, max_tool_calls=2)
            ).run("Create note.txt, read it, and report success.")
    if (
        result.stop_reason != StopReason.COMPLETED
        or len(result.tool_results) != 2
        or not all(item.ok for item in result.tool_results)
        or result.tool_results[-1].content != expected
    ):
        raise SystemExit("Mock demo failed its actual tool-result checks")
    for item in result.messages:
        if isinstance(item, ToolCall):
            print(f"CALL {item.call_id} {item.name} {item.arguments}")
        elif isinstance(item, ToolResult):
            print(f"RESULT {item.call_id} {item.output_json()}")
    print(f"FINAL {result.final_answer}")
    print(
        f"STOP {result.stop_reason.value} "
        f"(model_rounds={result.model_rounds}, tool_calls={result.tool_calls})"
    )


if __name__ == "__main__":
    main()
