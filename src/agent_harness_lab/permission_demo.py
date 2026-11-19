"""Deterministic permission demonstration; no network or real human approval."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission, PermissionEngine
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def main() -> None:
    with TemporaryDirectory(prefix="harness-permission-demo-") as temporary:
        base = Path(temporary)
        root = base / "workspace"
        root.mkdir()
        trace_path = base / "permission.jsonl"
        approvals = iter([True, False])

        def approve(request: ApprovalRequest) -> bool:
            # A scripted test double, not a production approval UI.
            return next(approvals)

        with JSONLTrace(trace_path) as trace, WorkspaceFiles(root, allow_write=True) as files:
            files.write_text("input.txt", "synthetic input\n")
            defaults = ToolRegistry()
            register_workspace_tools(defaults, files)
            denied = Agent(
                MockProvider(
                    [
                        ModelResponse(
                            tool_calls=(
                                ToolCall(
                                    "default_write",
                                    "write_file",
                                    '{"path":"unapproved.txt","content":"blocked"}',
                                ),
                            )
                        ),
                        "Default write denied.",
                    ]
                ),
                defaults,
                trace=trace,
            ).run("Attempt a write without a write capability.")
            engine = PermissionEngine(
                permissions=(Permission.READ, Permission.WRITE),
                rules={"write_file": Decision.ASK},
                approval_handler=approve,
            )
            registry = ToolRegistry(permission_engine=engine)
            register_workspace_tools(registry, files)
            provider = MockProvider(
                [
                    ModelResponse(
                        tool_calls=(ToolCall("read", "read_file", '{"path":"input.txt"}'),)
                    ),
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "approved_write",
                                "write_file",
                                '{"path":"receipt.txt","content":"approved\\n"}',
                            ),
                        )
                    ),
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "rejected_overwrite",
                                "write_file",
                                '{"path":"receipt.txt","content":"rejected","overwrite":true}',
                            ),
                        )
                    ),
                    "Approved content preserved; overwrite denied.",
                ]
            )
            result = Agent(provider, registry, limits=RunLimits(4, 3), trace=trace).run(
                "Read input, request a write, then request overwriting it."
            )
            valid = (
                denied.stop_reason == StopReason.COMPLETED
                and denied.tool_results[0].error is not None
                and denied.tool_results[0].error.code == "permission_denied"
                and not (root / "unapproved.txt").exists()
                and result.stop_reason == StopReason.COMPLETED
                and [output.ok for output in result.tool_results] == [True, True, False]
                and files.read_text("receipt.txt") == "approved\n"
            )
        if not valid:
            raise SystemExit("Permission demo failed its actual result and file checks")
        print("APPROVAL_SOURCE scripted Mock decisions: true, false; no interactive input")
        for row in map(json.loads, trace_path.read_text().splitlines()):
            if row["event"] == "tool.call":
                print(
                    f"REQUEST {row['tool_name']} policy={row['permission_decision']} "
                    f"authorized={str(row['authorized']).lower()} reason={row['permission_reason']}"
                )
        print("DEFAULT_WRITE denied; file_created=false")
        print("OVERWRITE rejected; approved_file_unchanged=true")
        print(f"FINAL {result.final_answer}")
        print(f"STOP {result.stop_reason.value}")


if __name__ == "__main__":
    main()
