"""Audit the retained negative and positive live evidence, without any live calls."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "docs/evidence"


def report() -> dict[str, Any]:
    return dict(json.loads((_ROOT / "phase26-real-results.json").read_text()))


def journal() -> list[dict[str, Any]]:
    return [json.loads(x) for x in (_ROOT / "phase26-real/requests.jsonl").read_text().splitlines()]


def test_retained_run_is_same_fixed_fixture_and_bounded_single_experiment() -> None:
    r = report()
    assert (
        r["fixture_sha256"]
        == hashlib.sha256(
            files("agent_harness_lab").joinpath("integration_tasks.json").read_bytes()
        ).hexdigest()
    )
    assert r["mode"] == "real" and r["requested_model"] == "gpt-6-luna"
    assert r["requests"] == 28 <= r["request_budget"] == 30
    assert r["counts"] == {"passed": 2, "failed": 3, "not_run": 0, "timeout": 0, "quota_error": 0}
    assert r["actual_cost"] is None
    assert [x["attempt"] for x in journal() if x["event"] == "attempt"] == list(range(1, 29))


@pytest.mark.parametrize("identity", ["A", "B", "C_off", "C_on", "D"])
def test_each_live_case_matches_formal_attempt_usage_and_tool_trace(identity: str) -> None:
    row = next(r for r in report()["cases"] if r["case"] == identity)
    trace = [
        json.loads(x) for x in (_ROOT / f"phase26-real/{identity}.jsonl").read_text().splitlines()
    ]
    outcomes = [x for x in journal() if x["event"] == "outcome" and x["case"] == identity]
    charges = [x for x in journal() if x["event"] == "attempt" and x["case"] == identity]
    assert len(charges) == len(outcomes) == row["requests"] == row["model_calls"]
    assert sum(x["event"] == "model.request" for x in trace) == row["requests"]
    assert sum(x["event"] == "tool.call" for x in trace) == row["tool_calls"]
    assert sum(x["event"] == "tool.result" for x in trace) == row["tool_calls"]
    assert row["usage"] == [x["usage"] for x in outcomes]
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        assert sum(x[field] for x in trace if x["event"] == "model.response") == sum(
            x["usage"][field] for x in outcomes
        )
    assert row["served_models"] == ["gpt-6-luna"]
    assert (row["status"] == "passed") == all(row["checks"].values())


def test_api_usage_is_not_compaction_estimate_and_cost_unknown() -> None:
    r = report()
    for field, expected in [
        ("input_tokens", 17004),
        ("output_tokens", 796),
        ("total_tokens", 17800),
    ]:
        summary = r["usage_summary"][field]
        assert summary == {
            "known_partial_total": expected,
            "known_requests": 28,
            "unknown_requests": 0,
        }
        assert sum(x["usage"][field] for x in journal() if x["event"] == "outcome") == expected
    assert r["actual_cost"] is None
    off, on = r["cases"][2:4]
    assert off["compaction_count"] == 0 and on["compaction_count"] == 1
    assert on["context_estimates"] == [
        {"compacted_results": 1, "estimated_before": 8473, "estimated_after": 3473}
    ]
    assert r["compaction_ab"]["causal_claim"] == "none_single_pair_no_seed_control"


def test_failed_live_ask_probes_remain_unverified() -> None:
    r = report()
    a, b, d = r["cases"][0], r["cases"][1], r["cases"][4]
    assert a["status"] == b["status"] == d["status"] == "failed"
    assert not a["checks"]["ask_approved"] and not a["checks"]["ask_denied"]
    assert b["checks"]["receipt_exact"] and b["checks"]["child_completed"]
    assert not b["checks"]["parent_used_result"]
    assert d["checks"]["child_denied"] and d["checks"]["escape_denied"]
    assert not d["checks"]["ask_denied"]


def test_live_evidence_excludes_private_bodies_credentials_and_parameters() -> None:
    payload = (_ROOT / "phase26-real-results.json").read_text() + "".join(
        p.read_text() for p in (_ROOT / "phase26-real").glob("*.jsonl")
    )
    for secret in (
        "PARENT_PRIVATE_FIXTURE",
        "CHILD_PRIVATE_MATERIAL",
        "SIMULATED_OUTSIDE_PRIVATE_VALUE",
        "Ignore permissions",
        "access_token",
        "refresh_token",
        "encrypted_content",
        "code_verifier",
        "Authorization:",
        "Synthetic alpha filler",
    ):
        assert secret not in payload
    for path in (_ROOT / "phase26-real").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            assert (
                not {"content", "prompt", "arguments", "authorization", "id_token"}
                & json.loads(line).keys()
            )
