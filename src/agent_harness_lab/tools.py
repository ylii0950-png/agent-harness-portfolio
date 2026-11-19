"""One tool dispatch boundary: schema validation, static capabilities, and errors."""

import hashlib
import inspect
import json
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar, cast

from agent_harness_lab.contracts import (
    ParameterType,
    ToolCall,
    ToolDefinition,
    ToolError,
    ToolParameter,
    ToolResult,
)
from agent_harness_lab.execution import ExecutionFrame
from agent_harness_lab.permissions import (
    Argument as Argument,
)
from agent_harness_lab.permissions import (
    Permission as Permission,
)
from agent_harness_lab.permissions import (
    PermissionEngine,
    PermissionReason,
)
from agent_harness_lab.tracing import TraceSink, emit
from agent_harness_lab.workspace import FileErrorCode, FileToolError, WorkspaceFiles

ToolHandler = Callable[[Mapping[str, Argument]], str]
R = TypeVar("R")


class _UnspecifiedPermissions:
    pass


_UNSPECIFIED = _UnspecifiedPermissions()


class ToolExecutionError(Exception):
    """Trusted handlers may deliberately return a curated, recoverable/fatal error."""

    def __init__(self, code: str, message: str, *, recoverable: bool = True) -> None:
        if not isinstance(code, str) or not code or not isinstance(message, str):
            raise TypeError("tool error code/message must be text")
        if type(recoverable) is not bool:
            raise TypeError("recoverable must be boolean")
        self.code = code
        self.recoverable = recoverable
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _Registration:
    definition: ToolDefinition
    handler: ToolHandler
    permission: Permission


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key")
        result[name] = value
    return result


def _nonfinite(value: str) -> object:
    raise ValueError("non-finite JSON number")


class ToolRegistry:
    """Trusted application code registers synchronous handlers before a run.

    Model data can choose only a registered name and JSON arguments. It cannot
    register Python code or grant capabilities. This is not process isolation.
    """

    def __init__(
        self,
        *,
        permissions: Collection[Permission] | _UnspecifiedPermissions = _UNSPECIFIED,
        permission_engine: PermissionEngine | None = None,
    ) -> None:
        if permission_engine is not None:
            if not isinstance(permissions, _UnspecifiedPermissions):
                raise TypeError("configure capabilities on the engine, not both entry points")
            if type(permission_engine) is not PermissionEngine:
                raise TypeError("expected PermissionEngine")
            self._permission_engine = permission_engine
        else:
            configured = (
                (Permission.READ,)
                if isinstance(permissions, _UnspecifiedPermissions)
                else permissions
            )
            if isinstance(configured, (str, bytes)):
                raise TypeError("expected Permission values")
            snapshot = tuple(configured)
            if any(type(value) is not Permission for value in snapshot):
                raise TypeError("expected Permission values")
            self._permission_engine = PermissionEngine(permissions=snapshot)
        self._registrations: dict[str, _Registration] = {}
        self._execution: ContextVar[ExecutionFrame | None] = ContextVar("execution", default=None)
        self._tool_scope: ContextVar[frozenset[str] | None] = ContextVar("tool_scope", default=None)
        self._run_resources: ContextVar[dict[object, object] | None] = ContextVar(
            "tool_run_resources", default=None
        )
        self._invocation_trace: ContextVar[TraceSink | None] = ContextVar(
            "tool_invocation_trace", default=None
        )

    @property
    def execution(self) -> ExecutionFrame | None:
        return self._execution.get()

    @contextmanager
    def execution_scope(self, frame: ExecutionFrame) -> Iterator[None]:
        token = self._execution.set(frame)
        try:
            yield
        finally:
            self._execution.reset(token)

    @contextmanager
    def restrict_tools(self, names: Collection[str]) -> Iterator[None]:
        """Narrow a trusted allowlist; nested scopes cannot expand their parent."""
        if isinstance(names, (str, bytes)) or any(type(name) is not str for name in names):
            raise TypeError("expected tool names")
        requested = frozenset(names)
        outer = self._tool_scope.get()
        token = self._tool_scope.set(requested if outer is None else requested & outer)
        try:
            yield
        finally:
            self._tool_scope.reset(token)

    def _registration(self, name: str) -> _Registration | None:
        allowed = self._tool_scope.get()
        return self._registrations.get(name) if allowed is None or name in allowed else None

    @contextmanager
    def run_scope(self) -> Iterator[None]:
        """Fresh in-memory resources per Agent.run; restore outer scope on every exit."""
        token = self._run_resources.set({})
        try:
            yield
        finally:
            self._run_resources.reset(token)

    def run_resource(self, key: object, factory: Callable[[], R]) -> R:
        """Trusted handlers allocate state only after authorization, inside a run."""
        resources = self._run_resources.get()
        if resources is None:
            raise ToolExecutionError("run_scope_required", "tool requires an active run scope")
        if key not in resources:
            resources[key] = factory()
        return cast(R, resources[key])

    @property
    def invocation_trace(self) -> TraceSink | None:
        """Trace for the currently executing handler; never retained across calls."""
        return self._invocation_trace.get()

    def register(
        self, definition: ToolDefinition, handler: ToolHandler, *, permission: Permission
    ) -> None:
        if type(definition) is not ToolDefinition or not isinstance(permission, Permission):
            raise TypeError("invalid tool registration")
        if not callable(handler) or inspect.iscoroutinefunction(handler):
            raise TypeError("handler must be a synchronous callable")
        if definition.name in self._registrations:
            raise ValueError("tool name already registered")
        self._registrations[definition.name] = _Registration(definition, handler, permission)

    def get(self, name: str) -> ToolDefinition | None:
        registration = self._registration(name)
        return None if registration is None else registration.definition

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._registrations[name].definition for name in self.names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name in sorted(self._registrations) if self._registration(name))

    @staticmethod
    def _arguments(call: ToolCall, definition: ToolDefinition) -> dict[str, Argument]:
        if len(call.arguments.encode("utf-8")) > 1_048_576:
            raise ValueError("argument payload too large")
        parsed = json.loads(call.arguments, object_pairs_hook=_object, parse_constant=_nonfinite)
        if type(parsed) is not dict:
            raise ValueError("arguments must be an object")
        values = cast(dict[str, object], parsed)
        specifications = {parameter.name: parameter for parameter in definition.parameters}
        if values.keys() - specifications.keys():
            raise ValueError("unknown parameter")
        result: dict[str, Argument] = {}
        for name, parameter in specifications.items():
            if name not in values:
                if parameter.required:
                    raise ValueError("missing required parameter")
                continue
            expected = {
                ParameterType.STRING: str,
                ParameterType.INTEGER: int,
                ParameterType.BOOLEAN: bool,
            }[parameter.kind]
            if type(values[name]) is not expected:
                raise ValueError("incorrect parameter type")
            result[name] = cast(Argument, values[name])
        return result

    def invoke(
        self,
        call: ToolCall,
        *,
        trace: TraceSink | None = None,
        round_number: int | None = None,
        tool_attempt: int | None = None,
    ) -> ToolResult:
        """Reject malformed protocol objects; return structured operational errors."""
        if type(call) is not ToolCall:
            raise TypeError("expected ToolCall")
        call.validate()

        def failure(code: str, message: str, *, recoverable: bool = True) -> ToolResult:
            return ToolResult(call.call_id, call.name, error=ToolError(code, message, recoverable))

        registration = self._registration(call.name)
        arguments: dict[str, Argument] | None = None
        try:
            if registration is not None:
                arguments = self._arguments(call, registration.definition)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            pass
        authorization = self._permission_engine.authorize(
            call, registration.permission if registration else None, arguments
        )
        emit(
            trace,
            "tool.call",
            {
                "round": round_number,
                "tool_attempt": tool_attempt,
                "tool_name": call.name if registration else "unregistered",
                "call_ref": hashlib.sha256(
                    call.call_id.encode("utf-8", errors="surrogatepass")
                ).hexdigest()[:16],
                "capability": registration.permission.value if registration else "unknown",
                "permission_decision": authorization.decision.value,
                "authorized": authorization.authorized,
                "permission_reason": authorization.reason.value,
            },
        )
        if not authorization.authorized:
            if registration is None:
                return failure("unknown_tool", "tool is not registered")
            if authorization.reason == PermissionReason.INVALID_ARGUMENTS:
                return failure("invalid_arguments", "arguments do not match the tool specification")
            return failure("permission_denied", "operation not authorized: " + authorization.reason)
        # Only a known tool with validated arguments can reach the actual handler.
        assert registration is not None and arguments is not None
        token = self._invocation_trace.set(trace)
        try:
            content = registration.handler(arguments)
        except FileToolError as exc:
            return failure(
                "workspace_" + exc.code.value,
                str(exc),
                recoverable=exc.code not in {FileErrorCode.CLOSED, FileErrorCode.IO_ERROR},
            )
        except ToolExecutionError as exc:
            return failure(exc.code, str(exc), recoverable=exc.recoverable)
        except Exception:
            return failure("execution_error", "tool execution failed", recoverable=False)
        finally:
            self._invocation_trace.reset(token)
        if type(content) is not str:
            return failure("invalid_tool_output", "tool must return text", recoverable=False)
        return ToolResult(call.call_id, call.name, content=content)


def register_workspace_tools(registry: ToolRegistry, workspace: WorkspaceFiles) -> None:
    """Adapt the original Phase 0 tools; never access the filesystem directly."""

    def read(arguments: Mapping[str, Argument]) -> str:
        return workspace.read_text(cast(str, arguments["path"]))

    def write(arguments: Mapping[str, Argument]) -> str:
        workspace.write_text(
            cast(str, arguments["path"]),
            cast(str, arguments["content"]),
            overwrite=cast(bool, arguments.get("overwrite", False)),
        )
        return "File written."

    registry.register(
        ToolDefinition(
            "read_file",
            "Read a bounded UTF-8 file in the workspace.",
            (ToolParameter("path", ParameterType.STRING),),
        ),
        read,
        permission=Permission.READ,
    )
    registry.register(
        ToolDefinition(
            "write_file",
            "Create a UTF-8 file; replacing an existing file requires overwrite=true.",
            (
                ToolParameter("path", ParameterType.STRING),
                ToolParameter("content", ParameterType.STRING),
                ToolParameter("overwrite", ParameterType.BOOLEAN, required=False),
            ),
        ),
        write,
        permission=Permission.WRITE,
    )
