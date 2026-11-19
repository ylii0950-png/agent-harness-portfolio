"""Actual local protocol round trips plus schema, permission and lifecycle failures."""

import json
import sys
from dataclasses import replace

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.execution import TotalLimits
from agent_harness_lab.extension_tools import ExtensionRegistry
from agent_harness_lab.mcp_client import MCPClient, ServerConfig, translate
from agent_harness_lab.permissions import Decision, Permission
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits


def config(**kwargs: object) -> ServerConfig:
    base = ServerConfig(
        "fiction",
        sys.executable,
        ("-m", "agent_harness_lab.mcp_example_server"),
        (("lookup", Permission.READ), ("fail", Permission.READ), ("slow", Permission.READ)),
    )
    return replace(base, **kwargs)  # type: ignore[arg-type]


def run(registry: ExtensionRegistry, name: str, args: str = "{}", *, tools: int = 2) -> object:
    return Agent(
        MockProvider([ModelResponse(tool_calls=(ToolCall("actual-id", name, args),)), "DONE"]),
        registry,
        limits=RunLimits(3, tools),
        total_limits=TotalLimits(3, tools, 1),
    ).run("Public synthetic MCP task")


def test_real_stdio_agent_roundtrip_identity_and_server_failure() -> None:
    with MCPClient(config()) as client:
        registry = ExtensionRegistry()
        client.register(registry)
        result = run(registry, "mcp_fiction_lookup", '{"code":"PUBLIC"}')
        from agent_harness_lab.runtime import AgentResult

        assert isinstance(result, AgentResult)
        row = json.loads(result.tool_results[0].content)
        assert row["provenance"]["harness_call_id"] == result.tool_results[0].call_id == "actual-id"
        assert row["provenance"] == client.calls[-1]
        assert row["result"]["structuredContent"]["meta"]["harness_call_id"] == "actual-id"
        assert row["trust"] == "low_trust_data" and row["origin"] == "external_mcp"
        failed = run(registry, "mcp_fiction_fail")
        assert isinstance(failed, AgentResult)
        assert failed.tool_results[0].error is not None
        assert failed.tool_results[0].error.code == "mcp_server_error"
    assert client._thread is not None and not client._thread.is_alive()
    disconnected = run(registry, "mcp_fiction_lookup", '{"code":"PUBLIC"}')
    assert isinstance(disconnected, AgentResult)
    assert disconnected.tool_results[0].error is not None
    assert disconnected.tool_results[0].error.code == "mcp_disconnected"


def test_permission_and_budget_denials_precede_protocol_dispatch() -> None:
    from agent_harness_lab.runtime import AgentResult

    with MCPClient(config()) as client:
        denied = ExtensionRegistry(rules={"mcp_fiction_lookup": Decision.DENY})
        client.register(denied)
        result = run(denied, "mcp_fiction_lookup", '{"code":"PUBLIC"}')
        assert isinstance(result, AgentResult)
        assert result.tool_results[0].error is not None
        assert result.tool_results[0].error.code == "permission_denied"
        assert client.calls == []
        allowed = ExtensionRegistry()
        client.register(allowed)
        result = run(allowed, "mcp_fiction_lookup", '{"code":"PUBLIC"}', tools=0)
        assert isinstance(result, AgentResult)
        assert result.stop_reason.value == "max_tool_calls" and client.calls == []
        wrong = run(allowed, "mcp_fiction_lookup", '{"code":true}')
        assert isinstance(wrong, AgentResult)
        assert wrong.tool_results[0].error is not None
        assert wrong.tool_results[0].error.code == "invalid_arguments" and client.calls == []


def test_timeout_poisoned_session_has_no_retry() -> None:
    from agent_harness_lab.runtime import AgentResult

    with MCPClient(config(timeout_seconds=0.1)) as client:
        registry = ExtensionRegistry()
        client.register(registry)
        result = run(registry, "mcp_fiction_slow", '{"seconds":2}')
        assert isinstance(result, AgentResult)
        assert result.tool_results[0].error is not None
        assert result.tool_results[0].error.code in {"mcp_timeout", "mcp_transport_failed"}
        assert len(client.calls) == 1 and client._closed


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array"},
        {"type": "object", "properties": {"x": {"type": "number"}}},
        {"type": "object", "properties": {"x": {"type": "object"}}},
        {"type": "object", "properties": {"x": {"$ref": "https://invalid.example/schema"}}},
        {"type": "object", "properties": {"x": {"type": "string", "pattern": "(a+)+$"}}},
        {"type": "object", "properties": {}, "required": ["missing"]},
        {"type": "object", "properties": {}, "additionalProperties": True},
    ],
)
def test_schema_rejection_is_explicit_without_starting_server(schema: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        translate("fiction", "lookup", schema)


def test_ceiling_startup_failure_and_atomic_registration() -> None:
    with pytest.raises(ValueError):
        config(tools=(("lookup", Permission.WRITE),))
    with pytest.raises(ValueError, match="mcp_startup_failed"):
        with MCPClient(config(command="/nonexistent/harness-python")):
            pytest.fail("must not connect")
    with MCPClient(config()) as client:
        registry = ExtensionRegistry()
        client.discovered[0]["inputSchema"] = {"type": "array"}
        with pytest.raises(ValueError):
            client.register(registry)
        assert registry.names == ()


def test_child_call_keeps_origin_and_shared_tool_ceiling() -> None:
    from agent_harness_lab.subagents import register_subagent_tool

    with MCPClient(config()) as client:
        registry = ExtensionRegistry()
        client.register(registry)

        def factory() -> MockProvider:
            return MockProvider(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall("child-mcp-id", "mcp_fiction_lookup", '{"code":"CHILD"}'),
                        )
                    ),
                    "CHILD-DONE",
                ]
            )

        register_subagent_tool(
            registry, factory, child_tools=("mcp_fiction_lookup",), child_limits=RunLimits(3, 2)
        )
        for limit in (1, 2):
            before = len(client.calls)
            provider = MockProvider(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "parent-delegate",
                                "delegate_task",
                                '{"task":"Read fictional MCP item"}',
                            ),
                        )
                    ),
                    "DONE",
                ]
            )
            result = Agent(
                provider, registry, limits=RunLimits(3, 2), total_limits=TotalLimits(5, limit, 1)
            ).run("Parent public context")
            assert result.total_tool_calls <= limit
            assert len(client.calls) - before == (limit - 1)
        assert (
            client.calls[-1]["depth"] == 1 and client.calls[-1]["harness_call_id"] == "child-mcp-id"
        )


def test_actual_server_exit_reports_transport_failure_without_retry() -> None:
    from agent_harness_lab.runtime import AgentResult

    code = (
        "from mcp.server.fastmcp import FastMCP; import os\n"
        "s=FastMCP('Synthetic crash')\n@s.tool()\n"
        "def crash()->str: os._exit(3)\ns.run(transport='stdio')"
    )
    with MCPClient(config(args=("-c", code), tools=(("crash", Permission.READ),))) as client:
        registry = ExtensionRegistry()
        client.register(registry)
        result = run(registry, "mcp_fiction_crash")
        assert isinstance(result, AgentResult)
        assert result.tool_results[0].error is not None
        assert result.tool_results[0].error.code == "mcp_transport_failed"
        assert len(client.calls) == 1 and client._closed
