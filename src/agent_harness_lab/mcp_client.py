"""Opt-in local stdio MCP transport, adapted to the existing synchronous Registry.

The maintained SDK owns JSON-RPC/session/process handling. One transport thread
bridges async stdio to synchronous tools; it is not another Agent loop. External
schemas/descriptions/results never select permissions or become trusted state.
"""

import asyncio
import concurrent.futures
import json
import math
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_harness_lab.contracts import (
    ParameterType,
    ToolDefinition,
    ToolParameter,
)
from agent_harness_lab.extension_tools import ExtensionRegistry as MCPRegistry
from agent_harness_lab.extension_tools import current_call
from agent_harness_lab.permissions import Argument, Permission
from agent_harness_lab.tools import ToolExecutionError

__all__ = ["MCPClient", "MCPRegistry", "ServerConfig", "translate"]


@dataclass(frozen=True)
class ServerConfig:
    alias: str
    command: str
    args: tuple[str, ...]
    tools: tuple[tuple[str, Permission], ...]
    ceilings: tuple[Permission, ...] = (Permission.READ,)
    timeout_seconds: float = 3.0
    max_result_bytes: int = 32768

    def __post_init__(self) -> None:
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", self.alias)
            or not Path(self.command).is_absolute()
            or not self.args
            or type(self.args) is not tuple
            or len(self.args) > 64
            or any(type(a) is not str or len(a) > 4096 for a in self.args)
            or type(self.tools) is not tuple
            or not 1 <= len(self.tools) <= 32
            or len(dict(self.tools)) != len(self.tools)
            or type(self.ceilings) is not tuple
            or any(type(p) is not Permission for p in self.ceilings)
            or any(type(p) is not Permission or p not in self.ceilings for _, p in self.tools)
            or type(self.timeout_seconds) not in {int, float}
            or not 0 < self.timeout_seconds <= 30
            or not math.isfinite(self.timeout_seconds)
            or type(self.max_result_bytes) is not int
            or not 128 <= self.max_result_bytes <= 131072
        ):
            raise ValueError("invalid_mcp_configuration_or_capability_ceiling")


def translate(alias: str, remote: str, schema: dict[str, Any]) -> ToolDefinition:
    """Bounded flat scalar schemas; unsupported features fail, never silently coerce.

    Server constraints (enum/length/range) are additionally checked before RPC.
    No regex/ref/remote schema loading, nested objects or number/null coercions.
    """
    name = f"mcp_{alias}_{remote}"
    if len(json.dumps(schema).encode()) > 16384 or schema.get("type") != "object":
        raise ValueError("unsupported_mcp_schema")
    if set(schema) - {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "title",
        "description",
        "$schema",
    }:
        raise ValueError("unsupported_mcp_schema")
    if schema.get("additionalProperties", False) is not False:
        raise ValueError("unsupported_mcp_schema")
    props, required = schema.get("properties", {}), schema.get("required", [])
    if (
        type(props) is not dict
        or len(props) > 16
        or type(required) is not list
        or any(type(k) is not str or k not in props for k in required)
    ):
        raise ValueError("unsupported_mcp_schema")
    parameters = []
    for key, item in props.items():
        if type(item) is not dict or set(item) - {
            "type",
            "title",
            "description",
            "default",
            "enum",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
        }:
            raise ValueError("unsupported_mcp_schema")
        try:
            kind = ParameterType(item["type"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("unsupported_mcp_schema") from None
        parameters.append(ToolParameter(key, kind, key in required))
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    Draft202012Validator.check_schema(schema)
    return ToolDefinition(
        name, "External MCP tool (low-trust data; no permission grants).", tuple(parameters)
    )


@dataclass
class _Request:
    tool: str
    arguments: dict[str, Argument]
    meta: dict[str, object]
    future: concurrent.futures.Future[dict[str, Any]]


class MCPClient:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._ready: concurrent.futures.Future[list[dict[str, Any]]] = concurrent.futures.Future()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_Request | None] | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self.calls: list[dict[str, object]] = []  # IDs/metadata only, no arguments/results.
        self.discovered: list[dict[str, Any]] = []

    async def _serve(self) -> None:
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        try:
            with (
                TemporaryDirectory(prefix="harness-mcp-") as directory,
                open(os.devnull, "w") as errors,
            ):
                env = {k: "" for k in get_default_environment()}
                env.update(
                    HOME=directory, PATH=os.defpath, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"
                )
                params = StdioServerParameters(
                    command=self.config.command, args=list(self.config.args), env=env, cwd=directory
                )
                async with (
                    stdio_client(params, errlog=errors) as (read, write),
                    ClientSession(read, write) as session,
                ):
                    with anyio.fail_after(10):
                        await session.initialize()
                        listing = await session.list_tools()
                    if listing.nextCursor or len(listing.tools) > 64:
                        raise ValueError("unbounded_mcp_catalog")
                    self._ready.set_result([t.model_dump(mode="json") for t in listing.tools])
                    while (request := await self._queue.get()) is not None:
                        try:
                            with anyio.fail_after(self.config.timeout_seconds):
                                result = await session.call_tool(
                                    request.tool, request.arguments, meta=request.meta
                                )
                            payload = result.model_dump(mode="json")
                            if len(json.dumps(payload).encode()) > self.config.max_result_bytes:
                                raise ValueError("mcp_result_limit")
                            request.future.set_result(payload)
                        except BaseException as error:
                            self._closed = True
                            if not request.future.done():
                                request.future.set_exception(
                                    ToolExecutionError(
                                        "mcp_timeout"
                                        if isinstance(error, TimeoutError)
                                        else "mcp_transport_failed",
                                        "MCP request failed or timed out; no automatic retry",
                                    )
                                )
                            break  # Poison timed-out/disconnected sessions; terminate subprocess.
        except BaseException:
            if not self._ready.done():
                self._ready.set_exception(ValueError("mcp_startup_failed"))
        finally:
            self._closed = True

    def __enter__(self) -> "MCPClient":
        import importlib.util

        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError("optional_mcp_dependency_required")
        if self._thread is not None or self._closed:
            raise ValueError("mcp_client_single_lifecycle")
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self._thread.start()
        try:
            self.discovered = self._ready.result(timeout=12)
            return self
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._closed = True
        if self._loop is not None and self._queue is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        if self._thread is not None:
            self._thread.join(timeout=8)
            if self._thread.is_alive():
                raise RuntimeError("mcp_shutdown_incomplete")

    def __exit__(self, *args: object) -> None:
        self.close()

    def register(self, registry: MCPRegistry) -> tuple[str, ...]:
        if type(registry) is not MCPRegistry or self._closed or not self._ready.done():
            raise ValueError("active_mcp_registry_required")
        wanted = dict(self.config.tools)
        by_name = {t["name"]: t for t in self.discovered}
        if len(by_name) != len(self.discovered) or not wanted.keys() <= by_name.keys():
            raise ValueError("mcp_catalog_binding")
        prepared = [
            (remote, translate(self.config.alias, remote, by_name[remote]["inputSchema"]))
            for remote in wanted
        ]
        if any(definition.name in registry.names for _, definition in prepared):
            raise ValueError("mcp_tool_collision")
        for remote, definition in prepared:
            self._register_one(
                registry, remote, definition, wanted[remote], by_name[remote]["inputSchema"]
            )
        return tuple(definition.name for _, definition in prepared)

    def _register_one(
        self,
        registry: MCPRegistry,
        remote: str,
        definition: ToolDefinition,
        permission: Permission,
        schema: dict[str, Any],
    ) -> None:
        from jsonschema import Draft202012Validator

        validator = Draft202012Validator(schema)

        def invoke(arguments: Mapping[str, Argument]) -> str:
            if not validator.is_valid(dict(arguments)):
                raise ToolExecutionError("mcp_invalid_arguments", "MCP schema constraints rejected")
            call, frame = current_call(), registry.execution
            if (
                call is None
                or frame is None
                or self._closed
                or self._loop is None
                or self._queue is None
            ):
                raise ToolExecutionError("mcp_disconnected", "MCP session is unavailable")
            meta: dict[str, object] = {
                "harness_call_id": call.call_id,
                "run_id": frame.run_id,
                "depth": frame.depth,
                "server": self.config.alias,
                "tool": remote,
            }
            future: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()
            self.calls.append(dict(meta))
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, _Request(remote, dict(arguments), meta, future)
            )
            try:
                result = future.result(timeout=self.config.timeout_seconds + 2)
            except concurrent.futures.TimeoutError:
                self.close()
                raise ToolExecutionError(
                    "mcp_timeout", "MCP transport timeout; no automatic retry"
                ) from None
            if result.get("isError") is not False:
                raise ToolExecutionError("mcp_server_error", "MCP server reported tool failure")
            return json.dumps(
                {
                    "origin": "external_mcp",
                    "trust": "low_trust_data",
                    "provenance": meta,
                    "result": result,
                },
                allow_nan=False,
            )

        registry.register(definition, invoke, permission=permission)
