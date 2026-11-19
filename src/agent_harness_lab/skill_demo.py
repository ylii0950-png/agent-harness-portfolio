"""Deterministic catalog -> lazy load -> deduplicate -> verified mock receipt."""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.permissions import Decision, Permission, PermissionEngine
from agent_harness_lab.providers import ConversationItem, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits, StopReason
from agent_harness_lab.skill_tools import register_skill_tools
from agent_harness_lab.skills import SkillRegistry
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def _call(identity: str, tool: str, **arguments: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(identity, tool, json.dumps(arguments)),))


class _ObservedFiles(WorkspaceFiles):
    def __init__(self, root: Path) -> None:
        super().__init__(root, allow_write=True)
        self.body_reads: list[str] = []

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        if path.endswith("/SKILL.md"):
            self.body_reads.append(path)
        return super().read_text(path, max_bytes=max_bytes)


class _SkillMock(MockProvider):
    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        response = super().complete(messages, tools=tools)
        if len(self.calls) == 4:
            # Use the actual loaded text as data; never evaluate commands/code in it.
            loaded = next(
                json.loads(item.content)
                for item in messages
                if isinstance(item, ToolResult)
                and item.name == "load_skill"
                and json.loads(item.content)["status"] == "loaded"
            )
            if loaded["trust"] != "low":
                raise ValueError("missing trust label")
            line = str(loaded["content"]).split("OUTPUT: ", 1)[1].strip()
            return _call("write", "write_file", path="receipt.txt", content=line + "\n")
        return response


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="harness-skill-demo-") as temporary:
        root = Path(temporary)
        for name, description, body in (
            (
                "receipt",
                "Format a synthetic receipt.",
                "Use this synthetic receipt format.\nOUTPUT: SKILL READY",
            ),
            ("unrelated", "Unrelated synthetic note.", "UNSELECTED PRIVATE BODY"),
        ):
            path = root / "skills" / name / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
        approvals: list[str] = []

        def approve(request: object) -> bool:
            approvals.append("receipt")
            return True

        registry = ToolRegistry(
            permission_engine=PermissionEngine(
                permissions=(Permission.READ, Permission.WRITE),
                rules={"write_file": Decision.ASK},
                approval_handler=approve,
            )
        )
        provider = _SkillMock(
            [
                _call("catalog", "list_skills"),
                _call("selected", "load_skill", name="receipt"),
                _call("repeat", "load_skill", name="receipt"),
                "placeholder replaced by a deterministic response derived from actual loaded text",
                "Receipt created using the requested skill.",
            ]
        )
        trace_path = root / "trace.jsonl"
        workspace = _ObservedFiles(root)
        with workspace, JSONLTrace(trace_path) as trace:
            register_workspace_tools(registry, workspace)
            register_skill_tools(registry, SkillRegistry(workspace, ("skills",)))
            result = Agent(provider, registry, limits=RunLimits(5, 4), trace=trace).run(
                "Select the receipt skill and create a synthetic receipt."
            )
            catalog = json.loads(result.tool_results[0].content)
            selected = json.loads(result.tool_results[1].content)
            repeated = json.loads(result.tool_results[2].content)
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            valid = (
                result.stop_reason == StopReason.COMPLETED
                and all(item.ok for item in result.tool_results)
                and [x["name"] for x in catalog["skills"]] == ["receipt", "unrelated"]
                and selected["status"] == "loaded"
                and selected["trust"] == "low"
                and repeated["status"] == "already_loaded"
                and "content" not in repeated
                and workspace.body_reads == ["skills/receipt/SKILL.md"]
                and "UNSELECTED PRIVATE BODY" not in repr(provider.calls)
                and workspace.read_text("receipt.txt") == "SKILL READY\n"
                and approvals == ["receipt"]
                and (result.model_rounds, result.tool_calls) == (5, 4)
                and sum(r["event"] == "skill_discovered" for r in rows) == 1
                and sum(r["event"] == "skill_loaded" for r in rows) == 2
                and "OUTPUT" not in trace_path.read_text()
            )
        if not valid:
            raise SystemExit("Skill demo failed its actual metadata/body/state/file/trace checks")
        print("CATALOG receipt: Format a synthetic receipt.; unrelated: Unrelated synthetic note.")
        print("LOAD receipt status=loaded trust=low; actual_body_reads=1")
        print("REPEAT receipt status=already_loaded; body_omitted=true")
        print("UNSELECTED unrelated loaded=false; body_absent_from_context=true")
        print("APPROVAL_SOURCE scripted Mock approval; receipt write used ASK")
        print("RECEIPT actual_content=SKILL READY; verified=true")
        print(f"FINAL {result.final_answer}")
        print("STOP completed (model_rounds=5, tool_calls=4)")


if __name__ == "__main__":
    main()
