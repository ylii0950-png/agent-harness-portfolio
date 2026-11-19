import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab import integration_suite as suite
from agent_harness_lab.contracts import ToolCall, ToolResult
from agent_harness_lab.integration_grading import grade_integration
from agent_harness_lab.invocation_audit import AuditedRegistry
from agent_harness_lab.providers import Message, MockProvider, ModelResponse
from agent_harness_lab.remediation_cases import cases as repair_cases
from agent_harness_lab.remediation_cases import mock_for as repair_mock
from agent_harness_lab.stabilization_cases import cases, mock_for
from agent_harness_lab.tools import ToolRegistry


def script() -> list[ModelResponse]:
    source = mock_for("A_S", "parent")
    return [source.complete((Message("user", "script"),)) for _ in range(11)]


def run_case(tmp_path: Path, responses: list[ModelResponse]) -> dict[str, Any]:
    return suite._run_case(
        cases()[0], tmp_path, factory=lambda c, r: MockProvider(responses), ledger=None
    )


def test_full_a_s_uses_actual_readback_and_ordered_todo(tmp_path: Path) -> None:
    row = run_case(tmp_path, script())
    assert row["status"] == "passed" and all(row["checks"].values())
    assert (row["model_calls"], row["tool_calls"], row["stop_reason"]) == (11, 10, "completed")
    assert [a["approved"] for a in row["approvals"]] == [True, False]
    assert row["operations"][7]["handler_calls"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "other.txt"},
        {"content": "wrong\n"},
        {"overwrite": True},
    ],
)
def test_wrong_denied_target_content_or_operation_cannot_pass(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    steps = script()
    call = steps[7].tool_calls[0]
    steps[7] = ModelResponse(
        tool_calls=(replace(call, arguments=json.dumps({**json.loads(call.arguments), **changes})),)
    )
    row = run_case(tmp_path, steps)
    assert row["checks"]["receipt_exact"] and row["checks"]["final_plan_completed"]
    assert not row["checks"]["ask_denied"] and row["status"] == "failed"


def test_legacy_a_r_counterexample_is_also_rejected(tmp_path: Path) -> None:
    source = repair_mock("A_R", "parent")
    steps = [source.complete((Message("user", "script"),)) for _ in range(10)]
    call = steps[6].tool_calls[0]
    steps[6] = ModelResponse(
        tool_calls=(
            replace(call, arguments=json.dumps({"path": "other.txt", "content": "denied\n"})),
        )
    )
    row = suite._run_case(
        repair_cases()[0], tmp_path, factory=lambda c, r: MockProvider(steps), ledger=None
    )
    assert row["status"] == "failed" and not row["checks"]["ask_denied"]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "reference",
        "run",
        "round",
        "reorder",
        "duplicate",
        "static_deny",
        "result",
        "sequence",
        "proof",
    ],
)
def test_wrong_or_reordered_events_fail_with_correct_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    def grade(
        case: str, result: Any, root: Path, events: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, bool]:
        rows = copy.deepcopy(events)
        index = next(
            i
            for i, r in enumerate(rows)
            if r["event"] == "tool.call" and r["permission_decision"] == "ask"
        )
        if corruption == "missing":
            rows.pop(index)
        elif corruption == "reference":
            rows[index]["call_ref"] = "not_a_matching_reference"
        elif corruption == "run":
            rows[index]["run_id"] = "another_run"
        elif corruption == "round":
            rows[index]["round"] += 1
        elif corruption == "reorder":
            rows[index], rows[index + 1] = rows[index + 1], rows[index]
        elif corruption == "duplicate":
            rows.insert(index, rows[index].copy())
        elif corruption == "static_deny":
            for row in rows:
                if row["event"] == "tool.call" and not row["authorized"]:
                    row.update(permission_decision="deny", permission_reason="policy_denied")
        elif corruption == "result":
            row = next(r for r in rows if r["event"] == "tool.result")
            row["success"] = False
        elif corruption == "sequence":
            rows[index].pop("seq")
        else:
            kwargs["audits"] = [
                replace(p, handler_calls=1) if not p.result.ok else p for p in kwargs["audits"]
            ]
        checks = grade_integration(case, result, root, rows, **kwargs)
        assert checks["receipt_exact"]
        return checks

    monkeypatch.setattr(suite, "grade_integration", grade)
    row = run_case(tmp_path, script())
    assert row["status"] == "failed"
    assert (
        not row["checks"]["a_trace_integrity"]
        or not row["checks"]["denied_no_side_effects"]
        or not row["checks"]["ask_denied"]
    )


def test_completed_claim_without_actual_operations_fails(tmp_path: Path) -> None:
    row = suite._run_case(
        cases()[0],
        tmp_path,
        factory=lambda c, r: MockProvider(["I completed everything"]),
        ledger=None,
    )
    assert row["stop_reason"] == "completed" and row["status"] == "failed"


def test_todo_not_completed_despite_final_claim_fails(tmp_path: Path) -> None:
    steps = script()
    del steps[8]  # Omit explicit receipt completed.
    row = run_case(tmp_path, steps)
    assert row["checks"]["receipt_exact"] and not row["checks"]["final_plan_completed"]
    assert row["status"] == "failed"


def test_no_receipt_readback_is_not_verified_completion(tmp_path: Path) -> None:
    steps = script()
    del steps[6]
    row = run_case(tmp_path, steps)
    assert row["checks"]["final_plan_completed"]
    assert not row["checks"]["receipt_readback"] and row["status"] == "failed"


def test_side_effect_after_permission_refusal_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ToolRegistry.invoke

    def faulty(registry: ToolRegistry, call: ToolCall, **kwargs: Any) -> ToolResult:
        result = original(registry, call, **kwargs)
        if call.name == "write_file" and not result.ok:
            assert isinstance(registry, AuditedRegistry)
            registry.workspace.write_text("unexpected.txt", "synthetic fault\n")
        return result

    monkeypatch.setattr(ToolRegistry, "invoke", faulty)
    row = run_case(tmp_path, script())
    assert row["status"] == "failed" and row["engineering_blocker"]
    assert not row["checks"]["denied_no_side_effects"]
