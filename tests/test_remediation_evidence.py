"""Retained real successes AND failure audited offline, never re-executed in CI."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "docs/evidence"


def report() -> dict[str, Any]:
    return dict(json.loads((_ROOT / "phase26r-real-results.json").read_text()))


def journal() -> list[dict[str, Any]]:
    return [
        json.loads(x) for x in (_ROOT / "phase26r-real/requests.jsonl").read_text().splitlines()
    ]


def test_single_experiment_keeps_failure_and_versioned_fixture() -> None:
    r = report()
    assert (
        r["fixture_sha256"]
        == hashlib.sha256(
            files("agent_harness_lab").joinpath("remediation_tasks.json").read_bytes()
        ).hexdigest()
    )
    assert r["suite"] == "phase26r" and r["grader_version"] == 2 and r["mode"] == "real"
    assert r["requests"] == 26 <= r["request_budget"] == 30
    assert r["counts"] == {"passed": 2, "failed": 1, "not_run": 0, "timeout": 0, "quota_error": 0}
    assert [c["status"] for c in r["cases"]] == ["failed", "passed", "passed"]
    assert [c["requests"] for c in r["cases"]] == [9, 8, 9]
    assert [c["tool_calls"] for c in r["cases"]] == [8, 6, 8]
    assert [c["stop_reason"] for c in r["cases"]] == ["completed"] * 3
    assert [p["remaining"] for p in r["preflight"]] == [30, 21, 13]
    assert [x["attempt"] for x in journal() if x["event"] == "attempt"] == list(range(1, 27))


@pytest.mark.parametrize("index,cap", [(0, 11), (1, 8), (2, 10)])
def test_trace_journal_usage_and_role_budgets_match(index: int, cap: int) -> None:
    row = report()["cases"][index]
    case = row["case"]
    trace = [
        json.loads(line)
        for line in (_ROOT / f"phase26r-real/{case}.jsonl").read_text().splitlines()
    ]
    attempts = [x for x in journal() if x["event"] == "attempt" and x["case"] == case]
    outcomes = [x for x in journal() if x["event"] == "outcome" and x["case"] == case]
    assert len(attempts) == len(outcomes) == row["model_calls"] == row["requests"] <= cap
    assert sum(x["event"] == "model.request" for x in trace) == row["requests"]
    assert sum(x["event"] == "tool.call" for x in trace) == row["tool_calls"]
    assert sum(x["event"] == "tool.result" for x in trace) == row["tool_calls"]
    assert row["usage"] == [x["usage"] for x in outcomes]
    assert row["served_models"] == ["gpt-6-luna"]
    assert not row["engineering_blocker"]
    assert (row["status"] == "passed") == all(row["checks"].values())
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        assert sum(x.get(field, 0) for x in trace if x["event"] == "model.response") == sum(
            x["usage"][field] for x in outcomes
        )
    for call in (x for x in trace if x["event"] == "tool.call"):
        output = next(
            x
            for x in trace
            if x["event"] == "tool.result"
            and x["run_id"] == call["run_id"]
            and x["call_ref"] == call["call_ref"]
        )
        assert output["seq"] > call["seq"]


def test_a_todo_usage_improved_but_approval_and_closure_still_failed() -> None:
    row = report()["cases"][0]
    assert row["checks"]["plan_created"] and row["checks"]["source_read"]
    assert row["approvals"] == [{"approved": False, "source": "scripted_fixture"}] * 2
    assert not row["checks"]["ask_approved"] and not row["checks"]["receipt_exact"]
    assert not row["checks"]["final_plan_completed"]
    trace = [
        json.loads(line) for line in (_ROOT / "phase26r-real/A_R.jsonl").read_text().splitlines()
    ]
    assert not any(str(x.get("error_code", "")).startswith("todo_") for x in trace)
    assert sum(x["event"] == "todo.updated" for x in trace) == 3


def test_real_d_actual_model_ask_calls_and_one_handler() -> None:
    row = report()["cases"][2]
    assert row["approvals"] == [
        {"approved": True, "source": "scripted_fixture"},
        {"approved": False, "source": "scripted_fixture"},
    ]
    assert row["ask_handler_calls"] == 1
    trace = [
        json.loads(line) for line in (_ROOT / "phase26r-real/D_R.jsonl").read_text().splitlines()
    ]
    asks = [x for x in trace if x["event"] == "tool.call" and x["tool_name"] == "request_write"]
    assert len(asks) == 2 and [x["authorized"] for x in asks] == [True, False]
    assert all(x["permission_decision"] == "ask" and x["depth"] == 0 for x in asks)
    assert all(row["checks"].values())


def test_actual_usage_complete_and_cost_unknown_not_zero() -> None:
    r = report()
    for field, total in (("input_tokens", 17379), ("output_tokens", 906), ("total_tokens", 18285)):
        assert r["usage_summary"][field] == {
            "known_partial_total": total,
            "known_requests": 26,
            "unknown_requests": 0,
        }
    assert r["actual_cost"] is None
    old = json.loads((_ROOT / "phase26-real-results.json").read_text())
    assert old["requests"] == 28 and old["counts"]["failed"] == 3


def test_retained_evidence_has_no_private_bodies_or_authentication_data() -> None:
    paths = list((_ROOT / "phase26r-real").glob("*.jsonl"))
    payload = (_ROOT / "phase26r-real-results.json").read_text() + "".join(
        p.read_text() for p in paths
    )
    for private in (
        "PARENT_PRIVATE_FIXTURE",
        "CHILD_PRIVATE_MATERIAL",
        "SIMULATED_OUTSIDE_PRIVATE_VALUE",
        "Ignore permissions",
        "access_token",
        "refresh_token",
        "Authorization:",
        "code_verifier",
        "encrypted_content",
        "AURORA-26",
        "NOVA-52",
    ):
        assert private not in payload
    for path in paths:
        for line in path.read_text().splitlines():
            assert (
                not {"content", "prompt", "arguments", "authorization", "id_token"}
                & json.loads(line).keys()
            )
