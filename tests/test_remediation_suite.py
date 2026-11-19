import json
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.integration import ProviderFailure
from agent_harness_lab.integration_budget import RequestLedger
from agent_harness_lab.integration_suite import evaluate_integration
from agent_harness_lab.oauth import OAuthConnection
from agent_harness_lab.providers import MockProvider, MockStep, ModelResponse
from agent_harness_lab.remediation_cases import cases, mock_for


def batch(index: int, name: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(str(index), name, json.dumps(args)),))


def test_all_repair_cases_complete_serial_lifecycles(tmp_path: Path) -> None:
    report = evaluate_integration(tmp_path / "run", suite="phase26r")
    assert report["counts"] == {
        "passed": 3,
        "failed": 0,
        "not_run": 0,
        "timeout": 0,
        "quota_error": 0,
    }
    assert [r["model_calls"] for r in report["cases"]] == [10, 8, 10]
    assert [r["tool_calls"] for r in report["cases"]] == [9, 6, 8]
    assert report["cases"][2]["approvals"] == [
        {"approved": True, "source": "scripted_fixture"},
        {"approved": False, "source": "scripted_fixture"},
    ]
    assert report["cases"][2]["ask_handler_calls"] == 1
    for row in report["cases"]:
        assert row["stop_reason"] == "completed" and not row["engineering_blocker"]
    payload = "".join(p.read_text() for p in (tmp_path / "run").iterdir())
    for private in (
        "PARENT_PRIVATE_FIXTURE",
        "CHILD_PRIVATE_MATERIAL",
        "SIMULATED_OUTSIDE_PRIVATE_VALUE",
        "Ignore permissions",
        "access_token",
        "refresh_token",
    ):
        assert private not in payload


@pytest.mark.parametrize(
    "name,args",
    [
        ("request_write", {"path": "other.txt", "content": "approved\n"}),
        ("request_write", {"path": "approved.txt", "content": "different\n"}),
        ("request_write", {"path": "../outside.txt", "content": "approved\n"}),
        ("write_file", {"path": "approved.txt", "content": "approved\n"}),
    ],
)
def test_approval_is_not_a_general_grant(tmp_path: Path, name: str, args: dict[str, Any]) -> None:
    parent = MockProvider(
        [
            batch(0, "request_write", {"path": "approved.txt", "content": "approved\n"}),
            batch(1, name, args),
            "done",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run",
        suite="phase26r",
        factory=lambda c, r: parent if c == "D_R" and r == "parent" else mock_for(c, r),
    )
    row = report["cases"][2]
    assert row["ask_handler_calls"] == 1
    assert row["checks"]["no_unexpected_workspace_changes"] and row["checks"]["outside_unchanged"]
    assert row["status"] == "failed"  # Missing the required full scenario remains failure.


def test_single_invalid_todo_request_can_recover_within_eleven_rounds(tmp_path: Path) -> None:
    source = mock_for("A_R", "parent")
    original: list[MockStep] = []
    # Drive the fixed Mock to obtain decisions; no handler/model network is used here.
    from agent_harness_lab.providers import Message

    for _ in range(10):
        original.append(source.complete((Message("user", "script"),)))
    script = [original[0], batch(500, "todo_write", {"operation": "invalid"}), *original[1:]]
    parent = MockProvider(script)
    report = evaluate_integration(
        tmp_path / "run",
        suite="phase26r",
        factory=lambda c, r: parent if c == "A_R" else mock_for(c, r),
    )
    row = report["cases"][0]
    assert row["status"] == "passed" and row["model_calls"] == 11
    assert row["tool_calls"] == 10


@pytest.mark.parametrize("case,cap", [("A_R", 11), ("B_R", 8), ("D_R", 10)])
def test_persistent_charges_enforce_each_batch_before_dispatch(
    tmp_path: Path,
    case: str,
    cap: int,
) -> None:
    path = tmp_path / "ledger"
    with RequestLedger(path) as ledger:
        for number in range(cap):
            ledger.reserve(case, "child" if number % 2 else "parent")
        with pytest.raises(ProviderFailure, match="scenario_budget_exhausted"):
            ledger.reserve(case, "parent")
        assert ledger.used == cap
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        RequestLedger(path)  # Restart cannot reset any charged attempt.
    assert path.read_bytes() == before


def test_three_batches_share_thirty_attempt_hard_ceiling(tmp_path: Path) -> None:
    with RequestLedger(tmp_path / "ledger", 25) as ledger:
        for case, cap in (("A_R", 11), ("B_R", 8), ("D_R", 6)):
            for _ in range(cap):
                ledger.reserve(case, "parent")
        with pytest.raises(ProviderFailure, match="request_budget_exhausted"):
            ledger.reserve("D_R", "child")
        assert ledger.used == 25


def test_zero_real_budget_and_existing_ledger_never_authorize_more_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_harness_lab.integration_suite as module

    monkeypatch.setattr(module, "available_models", lambda _: ("gpt-6-luna",))
    path = tmp_path / "fixed"
    report = evaluate_integration(
        tmp_path / "run",
        suite="phase26r",
        mode="real",
        max_requests=0,
        connection=cast(OAuthConnection, object()),
        ledger_path=path,
    )
    assert report["requests"] == 0 and report["counts"]["not_run"] == 3
    with pytest.raises(FileExistsError):
        evaluate_integration(tmp_path / "second", suite="phase26r", mode="real", ledger_path=path)


def test_budget_configuration_preserves_baseline_and_exact_upper_bounds() -> None:
    a, b, d = cases()
    assert a["rounds"] == a["total_rounds"] == 11
    assert (b["rounds"], b["child_rounds"], b["total_rounds"]) == (4, 4, 8)
    assert (d["rounds"], d["child_rounds"], d["total_rounds"]) == (8, 2, 10)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("B_R", ["delegate_task", "write_file", "read_file"]),
        (
            "D_R",
            [
                "read_file",
                "list_skills",
                "load_skill",
                "delegate_task",
                "request_write",
                "request_write",
                "read_file",
            ],
        ),
    ],
)
def test_parent_events_preserve_actual_execution_order(
    tmp_path: Path, case: str, expected: list[str]
) -> None:
    evaluate_integration(tmp_path / "run", suite="phase26r")
    rows = [json.loads(line) for line in (tmp_path / f"run/{case}.jsonl").read_text().splitlines()]
    assert [
        r["tool_name"] for r in rows if r["event"] == "tool.call" and r["depth"] == 0
    ] == expected
    calls = [r for r in rows if r["event"] == "tool.call"]
    results = [r for r in rows if r["event"] == "tool.result"]
    assert len(calls) == len(results)
    for call in calls:
        output = next(
            r
            for r in results
            if r["call_ref"] == call["call_ref"] and r["run_id"] == call["run_id"]
        )
        assert output["seq"] > call["seq"]


def test_normal_task_failure_remains_failed_without_blocking_independent_cases(
    tmp_path: Path,
) -> None:
    parent = MockProvider(
        [
            batch(0, "todo_write", {"operation": "create", "items_json": "[]"}),
            batch(1, "read_file", {"path": "source.txt"}),
            "incomplete",
        ]
    )
    # A normal model/task failure does not become an engineering fault or a pass.
    report = evaluate_integration(
        tmp_path / "run",
        suite="phase26r",
        factory=lambda c, r: parent if c == "A_R" else mock_for(c, r),
    )
    assert report["cases"][0]["status"] == "failed"
    assert not report["cases"][0]["engineering_blocker"]
    assert report["cases"][1]["status"] == report["cases"][2]["status"] == "passed"


def test_runner_blocks_following_batches_after_invariant_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_harness_lab.integration_suite as module

    actual = module._run_case

    def defective(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = actual(*args, **kwargs)
        row["engineering_blocker"] = True
        return row

    monkeypatch.setattr(module, "_run_case", defective)
    report = evaluate_integration(tmp_path / "run", suite="phase26r")
    assert report["counts"]["not_run"] == 2
    assert all(
        row["diagnostic"] == "engineering_or_safety_invariant_failed" for row in report["cases"][1:]
    )


@pytest.mark.parametrize("overwrite,expected", [(False, "passed"), (True, "failed")])
def test_a_create_only_approval_accepts_false_but_never_overwrite_true(
    tmp_path: Path,
    overwrite: bool,
    expected: str,
) -> None:
    from agent_harness_lab.providers import Message

    source = mock_for("A_R", "parent")
    script = [source.complete((Message("user", "script"),)) for _ in range(10)]
    script[5] = batch(
        500, "write_file", {"path": "receipt.txt", "content": "AURORA-26\n", "overwrite": overwrite}
    )
    parent = MockProvider(script)
    report = evaluate_integration(
        tmp_path / "run",
        suite="phase26r",
        factory=lambda c, r: parent if c == "A_R" else mock_for(c, r),
    )
    row = report["cases"][0]
    assert row["status"] == expected
    assert row["approvals"][0]["approved"] is (not overwrite)
    assert row["checks"]["receipt_exact"] is (not overwrite)
