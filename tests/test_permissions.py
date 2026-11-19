import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest

from agent_harness_lab.contracts import ParameterType, ToolCall, ToolDefinition, ToolParameter
from agent_harness_lab.permissions import (
    ApprovalRequest,
    Decision,
    Permission,
    PermissionEngine,
    PermissionReason,
)
from agent_harness_lab.tools import Argument, ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles

T = TypeVar("T")
R = TypeVar("R")


def record(items: list[T], value: T, reply: R) -> R:
    items.append(value)
    return reply


def call(name: str = "write_file", identity: str = "one", **arguments: Argument) -> ToolCall:
    return ToolCall(identity, name, json.dumps(arguments))


def engine(approval: Any = None, *, decision: Decision = Decision.ASK) -> PermissionEngine:
    return PermissionEngine(
        permissions=(Permission.READ, Permission.WRITE),
        rules={"write_file": decision},
        approval_handler=approval,
    )


def test_default_read_allowed_and_default_write_denied_before_handler(tmp_path: Path) -> None:
    (tmp_path / "existing").write_text("original")
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry()
        register_workspace_tools(registry, workspace)
        assert registry.invoke(call("read_file", path="existing")).content == "original"
        denied = registry.invoke(call(path="new", content="must not exist"))
        assert denied.error is not None and denied.error.code == "permission_denied"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing"]
    assert (tmp_path / "existing").read_text() == "original"


def test_legacy_explicit_capabilities_remain_preauthorized(tmp_path: Path) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
        register_workspace_tools(registry, workspace)
        assert registry.invoke(call(path="file", content="explicit opt-in")).ok
    assert (tmp_path / "file").read_text() == "explicit opt-in"


@pytest.mark.parametrize("approved", [True, False])
def test_ask_requires_explicit_approval_and_no_effect_on_rejection(
    tmp_path: Path, approved: bool
) -> None:
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        assert list(tmp_path.iterdir()) == []  # No handler effect before approval.
        requests.append(request)
        return approved

    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry(permission_engine=engine(approve))
        register_workspace_tools(registry, workspace)
        result = registry.invoke(call(path="file", content="expected"))
    assert result.ok is approved
    assert (tmp_path / "file").exists() is approved
    assert len(requests) == 1 and requests[0].call_id == "one"
    assert requests[0].tool_name == "write_file" and requests[0].permission == Permission.WRITE
    assert requests[0].arguments == {"path": "file", "content": "expected"}
    if not approved:
        assert result.error is not None and result.error.code == "permission_denied"
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reply", [None, "yes", "allow", 1, 0, [], {}, object()])
def test_truthy_or_malformed_approval_is_never_consent(tmp_path: Path, reply: Any) -> None:
    effects: list[str] = []
    registry = ToolRegistry(permission_engine=engine(lambda request: reply))
    registry.register(
        ToolDefinition("write_file", "A trusted write operation."),
        lambda arguments: record(effects, "ran", "ok"),
        permission=Permission.WRITE,
    )
    result = registry.invoke(call())
    assert result.error is not None and "approval_invalid" in result.error.message
    assert effects == []


def test_missing_approval_handler_denies_even_with_write_capability() -> None:
    outcome = engine().authorize(call(), Permission.WRITE, {})
    assert outcome.decision == Decision.ASK and not outcome.authorized
    assert outcome.reason == PermissionReason.APPROVAL_UNAVAILABLE


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("private credential"),
        ValueError("private path"),
        EOFError(),
        KeyboardInterrupt(),
        SystemExit(),
    ],
)
def test_approval_failure_is_sanitized_and_never_calls_handler(error: BaseException) -> None:
    effects: list[str] = []

    def approve(request: ApprovalRequest) -> bool:
        raise error

    registry = ToolRegistry(permission_engine=engine(approve))
    registry.register(
        ToolDefinition("write_file", "Write."),
        lambda arguments: record(effects, "ran", "ok"),
        permission=Permission.WRITE,
    )
    result = registry.invoke(call())
    assert result.error is not None and result.error.code == "permission_denied"
    assert result.error.recoverable and effects == []
    assert "private" not in result.output_json()


@pytest.mark.parametrize("decision", [Decision.ALLOW, Decision.ASK])
def test_rule_and_approval_cannot_expand_capability_ceiling(decision: Decision) -> None:
    requests: list[ApprovalRequest] = []
    policy = PermissionEngine(
        rules={"write_file": decision},
        approval_handler=lambda request: record(requests, request, True),
    )
    result = policy.authorize(call(), Permission.WRITE, {})
    assert not result.authorized and result.reason == PermissionReason.CAPABILITY_DENIED
    assert requests == []


def test_deny_rule_does_not_consult_approval_or_handler() -> None:
    seen: list[str] = []
    registry = ToolRegistry(
        permission_engine=engine(
            lambda request: record(seen, "approval", True), decision=Decision.DENY
        )
    )
    registry.register(
        ToolDefinition("write_file", "Write."),
        lambda arguments: record(seen, "handler", "ok"),
        permission=Permission.WRITE,
    )
    result = registry.invoke(call())
    assert result.error is not None and "policy_denied" in result.error.message
    assert seen == []


@pytest.mark.parametrize("name", ["bash", "write_file", "unknown"])
def test_unknown_tool_cannot_be_enabled_by_an_allow_rule(name: str) -> None:
    seen: list[ApprovalRequest] = []
    policy = PermissionEngine(
        permissions=(Permission.READ, Permission.WRITE),
        rules={name: Decision.ALLOW},
        approval_handler=lambda request: record(seen, request, True),
    )
    result = ToolRegistry(permission_engine=policy).invoke(call(name))
    assert result.error is not None and result.error.code == "unknown_tool"
    assert seen == []


@pytest.mark.parametrize(
    "configuration",
    [
        {"permissions": "write"},
        {"permissions": ["read"]},
        {"permissions": [True]},
        {"permissions": None},
        {"rules": {"read_file": "allow"}},
        {"rules": {"read_file": True}},
        {"rules": {"*": Decision.ALLOW}},
        {"rules": {1: Decision.ALLOW}},
        {"rules": []},
        {"approval_handler": True},
    ],
)
def test_invalid_configuration_denies_all_even_normally_allowed_reads(
    configuration: dict[str, Any],
) -> None:
    policy = PermissionEngine(**configuration)
    outcome = policy.authorize(call("read_file"), Permission.READ, {})
    assert not outcome.authorized and outcome.reason == PermissionReason.INVALID_POLICY


@pytest.mark.parametrize("operation", ["read", "delete", 1, True, object()])
def test_unknown_operation_category_fails_closed(operation: Any) -> None:
    outcome = PermissionEngine().authorize(call("read_file"), operation, {})
    assert not outcome.authorized and outcome.reason == PermissionReason.INVALID_REQUEST


def test_policy_and_capability_configuration_are_immutable_snapshots() -> None:
    grants = {Permission.READ, Permission.WRITE}
    rules = {"write_file": Decision.DENY}
    policy = PermissionEngine(permissions=grants, rules=rules)
    grants.clear()
    rules["write_file"] = Decision.ALLOW
    assert policy.authorize(call("read_file"), Permission.READ, {}).authorized
    assert not policy.authorize(call(), Permission.WRITE, {}).authorized
    with pytest.raises(FrozenInstanceError):
        cast(Any, policy)._valid = True


def test_approval_request_is_immutable_and_cannot_change_executed_arguments() -> None:
    seen: list[Mapping[str, Argument]] = []

    def approve(request: ApprovalRequest) -> bool:
        with pytest.raises(TypeError):
            request.arguments["text"] = "modified"  # type: ignore[index]
        with pytest.raises(FrozenInstanceError):
            cast(Any, request).tool_name = "other"
        assert "sensitive" not in repr(request)
        return True

    registry = ToolRegistry(permission_engine=engine(approve))
    registry.register(
        ToolDefinition("write_file", "Write.", (ToolParameter("text", ParameterType.STRING),)),
        lambda arguments: record(seen, dict(arguments), "ok"),
        permission=Permission.WRITE,
    )
    assert registry.invoke(call(text="sensitive exact text")).ok
    assert seen == [{"text": "sensitive exact text"}]


@pytest.mark.parametrize("arguments", ['{"path":1}', '{"path":"x","grant":true}', "{", "[]"])
def test_invalid_arguments_do_not_prompt_or_run_handler(arguments: str) -> None:
    seen: list[str] = []
    registry = ToolRegistry(permission_engine=engine(lambda request: record(seen, "asked", True)))
    registry.register(
        ToolDefinition("write_file", "Write.", (ToolParameter("path", ParameterType.STRING),)),
        lambda arguments: record(seen, "ran", "ok"),
        permission=Permission.WRITE,
    )
    result = registry.invoke(ToolCall("one", "write_file", arguments))
    assert result.error is not None and result.error.code == "invalid_arguments"
    assert seen == []


def test_conflicting_or_invalid_engine_configuration_is_rejected_at_registry_boundary() -> None:
    with pytest.raises(TypeError):
        ToolRegistry(permissions=(Permission.WRITE,), permission_engine=engine())
    with pytest.raises(TypeError):
        ToolRegistry(permission_engine=AnyEngine())  # type: ignore[arg-type]


class AnyEngine:
    def authorize(self, *args: Any) -> Any:
        raise AssertionError("an arbitrary engine must not be used")


def test_async_approvers_are_not_started_or_implicitly_awaited() -> None:
    async def approve(request: ApprovalRequest) -> bool:
        raise AssertionError("must not run")

    policy = engine(approve)
    assert not policy.authorize(call(), Permission.WRITE, {}).authorized
    wrapped = engine(lambda request: approve(request))
    result = wrapped.authorize(call(), Permission.WRITE, {})
    assert not result.authorized and result.reason == PermissionReason.APPROVAL_INVALID


def test_async_callable_approval_object_is_rejected_without_starting_it() -> None:
    class AsyncApprover:
        async def __call__(self, request: ApprovalRequest) -> bool:
            raise AssertionError("must not run")

    policy = engine(AsyncApprover())
    result = policy.authorize(call(), Permission.WRITE, {})
    assert not result.authorized and result.reason == PermissionReason.INVALID_POLICY


@pytest.mark.parametrize(
    "path", ["../outside", "/tmp/outside", "alias", "linked/outside", ".git/config"]
)
def test_explicit_approval_never_overrides_workspace_containment(tmp_path: Path, path: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (root / "alias").symlink_to(outside)
    (root / "linked").symlink_to(tmp_path, target_is_directory=True)
    with WorkspaceFiles(root, allow_write=True) as workspace:
        registry = ToolRegistry(permission_engine=engine(lambda request: True))
        register_workspace_tools(registry, workspace)
        result = registry.invoke(call(path=path, content="must not be written"))
    assert result.error is not None and result.error.code == "workspace_path_denied"
    assert outside.read_text() == "untouched"
    assert sorted(p.name for p in root.iterdir()) == ["alias", "linked"]


def test_explicit_approval_cannot_override_workspace_write_opt_in(tmp_path: Path) -> None:
    with WorkspaceFiles(tmp_path) as workspace:
        registry = ToolRegistry(permission_engine=engine(lambda request: True))
        register_workspace_tools(registry, workspace)
        result = registry.invoke(call(path="file", content="blocked"))
    assert result.error is not None and result.error.code == "workspace_permission_denied"
    assert list(tmp_path.iterdir()) == []


def test_direct_dispatch_trace_records_permission_without_sensitive_payload(tmp_path: Path) -> None:
    secret = "sensitive-private-payload"
    trace_path = tmp_path / "trace.jsonl"
    effects: list[str] = []
    registry = ToolRegistry(permission_engine=engine(lambda request: False))
    registry.register(
        ToolDefinition("write_file", "Write.", (ToolParameter("text", ParameterType.STRING),)),
        lambda arguments: record(effects, "ran", "ok"),
        permission=Permission.WRITE,
    )
    with JSONLTrace(trace_path) as trace:
        result = registry.invoke(call(identity=secret, text=secret), trace=trace)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert result.error is not None and effects == []
    assert rows[0]["event"] == "tool.call" and rows[0]["permission_decision"] == "ask"
    assert rows[0]["authorized"] is False and rows[0]["permission_reason"] == "approval_denied"
    assert secret not in trace_path.read_text()
    assert trace_path.stat().st_mode & 0o777 == 0o600


def test_broken_authorization_trace_prevents_handler_effects(tmp_path: Path) -> None:
    effects: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition("read", "Read."),
        lambda arguments: record(effects, "ran", "ok"),
        permission=Permission.READ,
    )
    trace = JSONLTrace(tmp_path / "trace.jsonl")
    trace.close()
    with pytest.raises(OSError):
        registry.invoke(call("read"), trace=trace)
    assert effects == []
