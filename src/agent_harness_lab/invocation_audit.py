"""Trusted, ephemeral eval evidence around the unchanged Registry permission entry."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_harness_lab.contracts import ToolCall, ToolDefinition, ToolResult
from agent_harness_lab.permissions import Argument, Permission, PermissionEngine
from agent_harness_lab.tools import ToolExecutionError, ToolHandler, ToolRegistry
from agent_harness_lab.tracing import TraceSink
from agent_harness_lab.workspace import WorkspaceFiles


@dataclass(frozen=True, repr=False)
class InvocationEvidence:
    call: ToolCall
    result: ToolResult
    run_id: str
    round_number: int | None
    attempt: int | None
    handler_calls: int
    files_unchanged: bool
    outside_unchanged: bool

    def summary(self, index: int) -> dict[str, object]:
        error = self.result.error.code if self.result.error else None
        stage = (
            ("handler_succeeded" if self.result.ok else "handler_failed")
            if self.handler_calls
            else (
                "schema_rejected"
                if error == "invalid_arguments"
                else "permission_rejected"
                if error == "permission_denied"
                else "dispatch_rejected"
            )
        )
        return {
            "operation_index": index,
            "stage": stage,
            "handler_calls": self.handler_calls,
            "files_unchanged": self.files_unchanged,
            "outside_unchanged": self.outside_unchanged,
        }


class AuditedRegistry(ToolRegistry):
    """Instrumentation only: delegate authorization/execution to the existing registry."""

    def __init__(
        self,
        engine: PermissionEngine,
        workspace: WorkspaceFiles,
        root: Path,
        outside: Path,
        health_check: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(permission_engine=engine)
        self.workspace, self.root, self.outside = workspace, root, outside
        self.evidence: list[InvocationEvidence] = []
        self._active: list[list[int]] = []
        self.healthy = True
        self._health_check = health_check or (lambda: True)

    def register(
        self, definition: ToolDefinition, handler: ToolHandler, *, permission: Permission
    ) -> None:
        def counted(arguments: Mapping[str, Argument]) -> str:
            if not self._active:
                raise RuntimeError("missing trusted invocation scope")
            self._active[-1][0] += 1
            return handler(arguments)

        super().register(definition, counted, permission=permission)

    def _snapshot(self) -> dict[str, object]:
        data: dict[str, object] = {}
        for path in self.root.rglob("*"):
            name = str(path.relative_to(self.root))
            if path.is_symlink():
                info = path.lstat()
                data[name] = ("symlink", info.st_ino, info.st_mode)
            elif path.is_file():
                data[name] = self.workspace.read_text(name)
            elif path.is_dir():
                data[name] = ("directory",)
        return data

    def invoke(
        self,
        call: ToolCall,
        *,
        trace: TraceSink | None = None,
        round_number: int | None = None,
        tool_attempt: int | None = None,
    ) -> ToolResult:
        if not self.healthy or not self._health_check():
            self.healthy = False
            raise ToolExecutionError(
                "eval_audit_unavailable", "eval audit unavailable", recoverable=False
            )
        active = [0]
        self._active.append(active)
        try:
            before, outside = self._snapshot(), self.outside.read_bytes()
            result = super().invoke(
                call, trace=trace, round_number=round_number, tool_attempt=tool_attempt
            )
            unchanged = before == self._snapshot()
            outside_unchanged = outside == self.outside.read_bytes()
            self.evidence.append(
                InvocationEvidence(
                    call,
                    result,
                    self.execution.run_id if self.execution else "",
                    round_number,
                    tool_attempt,
                    active[0],
                    unchanged,
                    outside_unchanged,
                )
            )
            if (
                not outside_unchanged
                or (active[0] == 0 and not unchanged)
                or not self._health_check()
            ):
                self.healthy = False
            return result
        except BaseException:
            self.healthy = False
            raise
        finally:
            self._active.pop()
