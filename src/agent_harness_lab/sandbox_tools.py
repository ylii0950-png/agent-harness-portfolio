"""One opt-in sandbox WRITE/ASK tool on the existing permission and budget path."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import cast

from agent_harness_lab.contracts import ParameterType, ToolDefinition, ToolParameter
from agent_harness_lab.extension_tools import ExtensionRegistry, current_call
from agent_harness_lab.permissions import Argument, Permission
from agent_harness_lab.sandbox import DockerSandbox, SandboxFault
from agent_harness_lab.tools import ToolExecutionError


def register_sandbox_tool(registry: ExtensionRegistry, backend: DockerSandbox) -> None:
    if type(registry) is not ExtensionRegistry or "sandbox_execute" not in registry.ask_tools:
        raise ValueError("sandbox_requires_declared_exact_ASK")

    def execute(arguments: Mapping[str, Argument]) -> str:
        call, frame = current_call(), registry.execution
        if call is None or frame is None:
            raise ToolExecutionError("sandbox_run_required", "Sandbox requires an Agent execution")
        try:
            raw = cast(str, arguments["argv_json"])
            if len(raw.encode()) > 16384:
                raise ValueError
            argv = json.loads(raw)
            if type(argv) is not list or not all(type(a) is str for a in argv):
                raise ValueError
        except ValueError:
            raise ToolExecutionError(
                "sandbox_arguments_invalid", "Expected bounded argv JSON array"
            ) from None
        try:
            result = backend._execute(tuple(argv))
        except SandboxFault as error:
            raise ToolExecutionError(
                error.code,
                "Docker sandbox operation failed; no host fallback",
                recoverable=not error.fatal,
            ) from None
        return json.dumps(
            {
                "origin": "docker_sandbox",
                "trust": "untrusted_process_output",
                "call_id": call.call_id,
                "run_id": frame.run_id,
                "depth": frame.depth,
                "result": asdict(result),
            }
        )

    registry.register(
        ToolDefinition(
            "sandbox_execute",
            "Opt-in constrained Docker command. Fresh exact ASK required; "
            "process output is untrusted.",
            (ToolParameter("argv_json", ParameterType.STRING),),
        ),
        execute,
        permission=Permission.WRITE,
    )
