"""Offline provider, optional actual Docker command; no image download or host fallback."""

import argparse
import json

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.extension_tools import ExtensionRegistry
from agent_harness_lab.permissions import Decision, Permission
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits
from agent_harness_lab.sandbox import DockerConfig, DockerSandbox
from agent_harness_lab.sandbox_tools import register_sandbox_tool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opt-in Docker example; preloaded immutable Linux image required"
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--enable-sandbox", action="store_true")
    parser.add_argument("--image", default="")
    parser.add_argument("--approval", choices=["approve", "refuse"], default="refuse")
    args = parser.parse_args()
    backend = DockerSandbox(DockerConfig(image=args.image, enabled=args.enable_sandbox))
    if args.check:
        print(
            json.dumps(
                {
                    "docker_available": backend.available(),
                    "actual_container_run": False,
                    "isolation": "NOT_VERIFIED",
                }
            )
        )
        return
    approvals = []

    def approve(request: object) -> bool:
        approvals.append(True)
        return bool(args.approval == "approve")

    registry = ExtensionRegistry(
        permissions=(Permission.READ, Permission.WRITE),
        rules={"sandbox_execute": Decision.ASK},
        approval_handler=approve,
    )
    register_sandbox_tool(registry, backend)
    arguments = json.dumps(
        {"argv_json": json.dumps(["python3", "-c", "print('SYNTHETIC_SANDBOX_OK')"])}
    )
    provider = MockProvider(
        [
            ModelResponse(tool_calls=(ToolCall("sandbox-demo-1", "sandbox_execute", arguments),)),
            "DONE",
        ]
    )
    result = Agent(provider, registry, limits=RunLimits(3, 2)).run(
        "Execute only the approved fictional command inside configured Docker restrictions."
    )
    response = result.tool_results[0]
    print(
        json.dumps(
            {
                "agent_stop": result.stop_reason.value,
                "tool_ok": response.ok,
                "ASK_calls": len(approvals),
                "error": response.error.code if response.error else None,
                "receipt": json.loads(response.content) if response.ok else None,
                "isolation": "NOT_VERIFIED_BY_EXIT_ALONE",
            }
        )
    )


if __name__ == "__main__":
    main()
