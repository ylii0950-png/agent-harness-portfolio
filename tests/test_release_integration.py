"""One installed Agent composes actual stdio, SQLite, files, ASK and trace."""

import json
import sys
from pathlib import Path

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.extension_tools import ExtensionRegistry
from agent_harness_lab.mcp_client import MCPClient, ServerConfig
from agent_harness_lab.memory import MemoryStore
from agent_harness_lab.memory_tools import MUTATIONS, register_memory_tools
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.tools import register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


@pytest.mark.parametrize("approved", [True, False])
def test_installed_extensions_share_exact_ask_budget_and_trace(
    tmp_path: Path, approved: bool
) -> None:
    approvals: list[str] = []

    def approve(request: ApprovalRequest) -> bool:
        approvals.append(request.tool_name)
        return approved

    registry = ExtensionRegistry(
        permissions=(Permission.READ, Permission.WRITE),
        rules={**dict.fromkeys(MUTATIONS, Decision.ASK), "write_file": Decision.ASK},
        approval_handler=approve,
    )
    store = MemoryStore(tmp_path / "memory" / "facts.db", "release", "fiction", enabled=True)
    register_memory_tools(registry, store, write=True)
    root = tmp_path / "workspace"
    root.mkdir()
    trace_path = tmp_path / "trace.jsonl"

    def call(identifier: str, name: str, **arguments: str) -> ModelResponse:
        return ModelResponse(tool_calls=(ToolCall(identifier, name, json.dumps(arguments)),))

    with (
        MCPClient(
            ServerConfig(
                "fiction",
                sys.executable,
                ("-m", "agent_harness_lab.mcp_example_server"),
                (("lookup", Permission.READ),),
            )
        ) as client,
        WorkspaceFiles(root, allow_write=True) as files,
        JSONLTrace(trace_path) as trace,
    ):
        client.register(registry)
        register_workspace_tools(registry, files)
        result = Agent(
            MockProvider(
                [
                    call("put", "memory_put", entity="nebula", attribute="code", value="NEBULA-52"),
                    ModelResponse(
                        tool_calls=(
                            ToolCall("recall", "memory_search", '{"query":"nebula"}'),
                            ToolCall("remote", "mcp_fiction_lookup", '{"code":"PUBLIC"}'),
                        )
                    ),
                    call("write", "write_file", path="receipt.txt", content="fictional receipt"),
                    call("read", "read_file", path="receipt.txt"),
                    "DONE",
                ]
            ),
            registry,
            limits=RunLimits(6, 5),
            trace=trace,
        ).run("Scripted offline integration; untrusted memory/MCP cannot authorize writes.")
        assert len(client.calls) == 1
        assert sorted(p.name for p in root.iterdir()) == (["receipt.txt"] if approved else [])
    assert result.stop_reason is StopReason.COMPLETED
    assert result.model_rounds == 5 and result.tool_calls == 5
    assert approvals == ["memory_put", "write_file"]
    assert len(store.search("nebula")) == int(approved)
    assert result.tool_results[0].ok is approved
    assert result.tool_results[3].ok is approved
    assert result.tool_results[4].ok is approved
    recall = json.loads(result.tool_results[1].content)
    remote = json.loads(result.tool_results[2].content)
    assert recall["origin"] == "persistent_memory" and recall["call_id"] == "recall"
    assert remote["origin"] == "external_mcp"
    assert remote["provenance"]["harness_call_id"] == "remote"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [r["tool_name"] for r in rows if r["event"] == "tool.call"] == [
        "memory_put",
        "memory_search",
        "mcp_fiction_lookup",
        "write_file",
        "read_file",
    ]
    assert sum(r["event"] == "tool.result" for r in rows) == 5
    assert rows[0]["event"] == "run.started" and rows[-1]["event"] == "run.stopped"
    assert "fictional receipt" not in trace_path.read_text()
