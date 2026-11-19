"""Shared optional-tool instrumentation; all authorization stays in Core."""

import contextvars
from collections.abc import Collection, Mapping

from agent_harness_lab.contracts import ToolCall, ToolResult
from agent_harness_lab.permissions import ApprovalHandler, Decision, Permission, PermissionEngine
from agent_harness_lab.tools import ToolRegistry
from agent_harness_lab.tracing import TraceSink

_CURRENT: contextvars.ContextVar[ToolCall | None] = contextvars.ContextVar(
    "extension_call", default=None
)


def current_call() -> ToolCall | None:
    return _CURRENT.get()


class ExtensionRegistry(ToolRegistry):
    """Actual call identity and immutable declared ASK routes, no alternate execution."""

    def __init__(
        self,
        *,
        permissions: Collection[Permission] = (Permission.READ,),
        rules: Mapping[str, Decision] | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        copied = dict(rules or {})
        self.ask_tools = frozenset(k for k, v in copied.items() if v is Decision.ASK)
        super().__init__(
            permission_engine=PermissionEngine(
                permissions=permissions, rules=copied, approval_handler=approval_handler
            )
        )

    def invoke(
        self,
        call: ToolCall,
        *,
        trace: TraceSink | None = None,
        round_number: int | None = None,
        tool_attempt: int | None = None,
    ) -> ToolResult:
        token = _CURRENT.set(call)
        try:
            return super().invoke(
                call, trace=trace, round_number=round_number, tool_attempt=tool_attempt
            )
        finally:
            _CURRENT.reset(token)
