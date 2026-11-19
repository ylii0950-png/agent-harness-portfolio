"""Audit immutable sanitized experiment evidence offline; never rerun the live task."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

EVIDENCE = Path(__file__).resolve().parents[1] / "docs/evidence"


def retained() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    report = json.loads((EVIDENCE / "phase26s-fix-real-results.json").read_text())
    root = EVIDENCE / "phase26s-fix-real"
    trace = [json.loads(s) for s in (root / "A_S.jsonl").read_text().splitlines()]
    journal = [json.loads(s) for s in (root / "requests.jsonl").read_text().splitlines()]
    return report, trace, journal


def test_single_experiment_charges_match_actual_model_trace_and_journal() -> None:
    report, trace, journal = retained()
    assert report["request_budget"] == 11 and report["requests"] == 10
    attempts = [r for r in journal if r["event"] == "attempt"]
    outcomes = [r for r in journal if r["event"] == "outcome"]
    assert len(attempts) == len(outcomes) == 10
    assert [r["attempt"] for r in attempts] == list(range(1, 11))
    assert all(r["case"] == "A_S" and r["role"] == "parent" for r in attempts)
    assert all(r["completed"] is True for r in outcomes)
    assert sum(r["event"] == "model.request" for r in trace) == 10
    assert sum(r["event"] == "model.response" for r in trace) == 10
    assert trace[-1]["stop_reason"] == "completed" and trace[-1]["total_model_calls"] == 10
    assert report["cases"][0]["served_models"] == ["gpt-6-luna"]


def test_runtime_grading_and_refusal_side_effect_evidence_remain_distinct() -> None:
    report, trace, _ = retained()
    case = report["cases"][0]
    assert case["status"] == "passed" and all(case["checks"].values())
    assert not case["engineering_blocker"] and case["provider_failures"] == []
    calls = [r for r in trace if r["event"] == "tool.call"]
    results = [r for r in trace if r["event"] == "tool.result"]
    assert len(calls) == len(results) == len(case["operations"]) == 10
    assert [r["tool_attempt"] for r in calls] == list(range(1, 11))
    ask = [
        (c, r, o)
        for c, r, o in zip(calls, results, case["operations"], strict=True)
        if c["permission_decision"] == "ask"
    ]
    assert len(ask) == 2 and [c["authorized"] for c, _, _ in ask] == [True, False]
    allowed, denied = ask
    assert allowed[0]["permission_reason"] == "approved"
    assert allowed[1]["success"] and allowed[2]["handler_calls"] == 1
    assert denied[0]["permission_reason"] == "approval_denied"
    assert denied[1]["error_code"] == "permission_denied"
    assert denied[2]["handler_calls"] == 0
    assert denied[2]["files_unchanged"] and denied[2]["outside_unchanged"]
    assert allowed[0]["round"] == denied[0]["round"] == 6
    assert case["approvals"][1]["categories"] == ["target_mismatch", "content_mismatch"]
    for call, result in zip(calls, results, strict=True):
        assert call["call_ref"] == result["call_ref"] and call["run_id"] == result["run_id"]
        assert call["seq"] < result["seq"]


def test_actual_usage_is_api_provided_and_unknown_cost_is_not_zero() -> None:
    report, trace, journal = retained()
    outcomes = [r for r in journal if r["event"] == "outcome"]
    responses = [r for r in trace if r["event"] == "model.response"]
    assert report["actual_cost"] is None
    for name, total in (("input_tokens", 11075), ("output_tokens", 345), ("total_tokens", 11420)):
        assert sum(r["usage"][name] for r in outcomes) == total
        assert sum(r[name] for r in responses) == total
        assert sum(r[name] for r in report["cases"][0]["usage"]) == total
        assert report["usage_summary"][name] == {
            "known_requests": 10,
            "unknown_requests": 0,
            "known_partial_total": total,
        }


def test_retained_configuration_and_separate_mock_counts() -> None:
    report, _, _ = retained()
    assert (
        report["fixture_sha256"]
        == hashlib.sha256(
            files("agent_harness_lab").joinpath("stabilization_tasks.json").read_bytes()
        ).hexdigest()
    )
    assert report["grader_version"] == report["a_grading_revision"] == 3
    assert [r["case"] for r in report["cases"]] == ["A_S"]
    assert report["counts"] == dict(passed=1, failed=0, not_run=0, timeout=0, quota_error=0)
    mock = json.loads((EVIDENCE / "phase26s-fix-mock-results.json").read_text())
    assert mock["mode"] == "mock" and mock["requests"] is None
    assert (
        mock["cases"][0]["model_calls"] == 11
    )  # Scripted serial path, distinct from live batching.


def test_retained_metadata_has_no_private_payloads_or_content_hashes() -> None:
    paths = [
        EVIDENCE / "phase26s-fix-real-results.json",
        *(EVIDENCE / "phase26s-fix-real").iterdir(),
    ]
    payload = "".join(p.read_text() for p in paths)
    for secret in (
        "AURORA-26",
        "receipt.txt",
        "source.txt",
        "forbidden.txt",
        "CODE:",
        "access_token",
        "refresh_token",
        "authorization_code",
        "PARENT_PRIVATE_FIXTURE",
        "SIMULATED_OUTSIDE_PRIVATE_VALUE",
        "/Users/",
        "arguments_sha256",
    ):
        assert secret not in payload
