import json
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab import integration_suite as suite
from agent_harness_lab.approval_diagnostics import ExactWriteApproval, Mismatch
from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.invocation_audit import AuditedRegistry
from agent_harness_lab.permissions import ApprovalRequest, Decision, Permission, PermissionEngine
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.stabilization_cases import cases
from agent_harness_lab.tools import register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.workspace import WorkspaceFiles


def request(**changes: Any) -> ApprovalRequest:
    fields: dict[str, Any] = dict(
        call_id="private-call-id",
        tool_name="write_file",
        permission=Permission.WRITE,
        arguments={"path": "private-target.txt", "content": "private-content\n"},
    )
    fields.update(changes)
    return ApprovalRequest(**fields)


@pytest.mark.parametrize("explicit_false", [False, True])
def test_approval_preserves_only_equivalent_create_modes(explicit_false: bool) -> None:
    approval = ExactWriteApproval("private-target.txt", "private-content\n")
    args = dict(request().arguments)
    if explicit_false:
        args["overwrite"] = False
    assert approval(request(arguments=args)) is True
    assert approval.healthy and approval.evidence[0].categories == ()


@pytest.mark.parametrize(
    "changes,category",
    [
        ({"tool_name": "other_tool"}, Mismatch.TOOL),
        ({"permission": Permission.READ}, Mismatch.SCOPE),
        ({"arguments": {"path": "other.txt", "content": "private-content\n"}}, Mismatch.TARGET),
        ({"arguments": {"path": "private-target.txt", "content": "wrong"}}, Mismatch.CONTENT),
        (
            {
                "arguments": {
                    "path": "private-target.txt",
                    "content": "private-content\n",
                    "overwrite": True,
                }
            },
            Mismatch.OVERWRITE,
        ),
        (
            {
                "arguments": {
                    "path": "private-target.txt",
                    "content": "private-content\n",
                    "extra": "x",
                }
            },
            Mismatch.SCHEMA,
        ),
        ({"arguments": {"path": "private-target.txt"}}, Mismatch.SCHEMA),
    ],
)
def test_mismatch_categories_never_grant(changes: dict[str, Any], category: Mismatch) -> None:
    approval = ExactWriteApproval("private-target.txt", "private-content\n")
    assert approval(request(**changes)) is False
    assert category in approval.evidence[-1].categories and approval.healthy
    public = json.dumps(approval.public_records())
    for secret in ("private-target", "private-content", "private-call-id", "wrong", "other.txt"):
        assert secret not in public
    assert "sha" not in public and "hash" not in public
    assert "private-" not in repr(approval) + repr(approval.evidence[-1])


@pytest.mark.parametrize("exact", [False, True])
def test_diagnostic_failure_cannot_change_the_authorization_decision(
    monkeypatch: pytest.MonkeyPatch,
    exact: bool,
) -> None:
    approval = ExactWriteApproval("private-target.txt", "private-content\n")

    def broken(*_: object) -> tuple[Mismatch, ...]:
        raise RuntimeError("private-target.txt private-content")

    monkeypatch.setattr(ExactWriteApproval, "classify", broken)
    attempted = request() if exact else request(tool_name="other_tool")
    assert approval(attempted) is exact
    assert not approval.healthy
    assert approval.public_records()[0]["categories"] == ["diagnostic_unavailable"]
    assert "private-" not in json.dumps(approval.public_records())


@pytest.mark.parametrize("diagnosis", [(), ("target_mismatch",), (Mismatch.TARGET,)])
def test_inconsistent_or_untrusted_diagnostic_values_block_audit(
    monkeypatch: pytest.MonkeyPatch,
    diagnosis: Any,
) -> None:
    approval = ExactWriteApproval("private-target.txt", "private-content\n")
    monkeypatch.setattr(ExactWriteApproval, "classify", lambda *_: diagnosis)
    # An exact approved call has no mismatch. A category cannot revoke or expand policy.
    permitted = approval(request())
    assert permitted is True
    assert approval.healthy is (diagnosis == ())


def test_diagnostic_failure_halts_remaining_tools_and_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ExactWriteApproval, "classify", lambda *_: (Mismatch.UNAVAILABLE,))
    source = MockProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "denied", "write_file", '{"path":"forbidden.txt","content":"denied\\n"}'
                    ),
                    ToolCall("later", "read_file", '{"path":"source.txt"}'),
                )
            ),
            "must not be called",
        ]
    )
    row = suite._run_case(cases()[0], tmp_path, factory=lambda *_: source, ledger=None)
    assert row["engineering_blocker"] and row["status"] == "failed"
    assert row["model_calls"] == 1 and len(row["operations"]) == 1
    assert row["operations"][0]["handler_calls"] == 0
    assert row["checks"]["outside_unchanged"]


@pytest.mark.parametrize(
    "kind,decision,args,stage,reason,approvals,handlers",
    [
        (
            "schema",
            Decision.ASK,
            {"path": 42, "content": "ok\n"},
            "schema_rejected",
            "invalid_arguments",
            0,
            0,
        ),
        (
            "static",
            Decision.DENY,
            {"path": "receipt.txt", "content": "ok\n"},
            "permission_rejected",
            "policy_denied",
            0,
            0,
        ),
        (
            "mismatch",
            Decision.ASK,
            {"path": "other.txt", "content": "ok\n"},
            "permission_rejected",
            "approval_denied",
            1,
            0,
        ),
        (
            "success",
            Decision.ASK,
            {"path": "receipt.txt", "content": "ok\n"},
            "handler_succeeded",
            "approved",
            1,
            1,
        ),
        (
            "handler",
            Decision.ASK,
            {"path": "receipt.txt", "content": "ok\n"},
            "handler_failed",
            "approved",
            1,
            1,
        ),
    ],
)
def test_schema_static_ask_and_handler_stages_remain_distinct(
    tmp_path: Path,
    kind: str,
    decision: Decision,
    args: dict[str, Any],
    stage: str,
    reason: str,
    approvals: int,
    handlers: int,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside")
    if kind == "handler":
        (root / "receipt.txt").write_text("existing")
    approval = ExactWriteApproval("receipt.txt", "ok\n")
    engine = PermissionEngine(
        permissions=(Permission.READ, Permission.WRITE),
        rules={"write_file": decision},
        approval_handler=approval,
    )
    with WorkspaceFiles(root, allow_write=True) as ws, JSONLTrace(tmp_path / "trace") as trace:
        registry = AuditedRegistry(engine, ws, root, outside)
        register_workspace_tools(registry, ws)
        result = registry.invoke(ToolCall("call", "write_file", json.dumps(args)), trace=trace)
    events = [json.loads(line) for line in (tmp_path / "trace").read_text().splitlines()]
    call = next(e for e in events if e["event"] == "tool.call")
    assert call["permission_reason"] == reason
    assert registry.evidence[0].summary(1)["stage"] == stage
    assert registry.evidence[0].handler_calls == handlers and len(approval.evidence) == approvals
    assert result.ok is (kind == "success")
    if handlers == 0:
        assert registry.evidence[0].files_unchanged and registry.evidence[0].outside_unchanged
    if kind == "handler":
        assert (root / "receipt.txt").read_text() == "existing"


def test_each_exact_approval_is_bound_to_its_immutable_request() -> None:
    approval = ExactWriteApproval("one.txt", "one\n")
    args: dict[str, Any] = {"path": "one.txt", "content": "one\n"}
    original = request(arguments=args)
    assert approval(original)
    args["path"] = "two.txt"
    assert not approval(request(call_id="second", arguments=args))
    assert approval.evidence[0].request.arguments["path"] == "one.txt"
    assert [r.approved for r in approval.evidence] == [True, False]
