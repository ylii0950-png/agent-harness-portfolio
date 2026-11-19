import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.http_transport import HTTPTransport
from agent_harness_lab.integration import FailureKind, ProviderFailure, failure
from agent_harness_lab.integration_budget import BudgetTransport, RequestLedger
from agent_harness_lab.integration_cases import cases, mock_for
from agent_harness_lab.integration_suite import _Observed, evaluate_integration
from agent_harness_lab.oauth import OAuthConnection
from agent_harness_lab.providers import Message, MockProvider, ModelResponse


def batch(index: int, *calls: tuple[str, dict[str, Any]]) -> ModelResponse:
    return ModelResponse(
        tool_calls=tuple(
            ToolCall(f"{index}_{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)
        )
    )


def test_all_fixed_mock_cases_cross_real_runtime_and_independent_grading(tmp_path: Path) -> None:
    report = evaluate_integration(tmp_path / "run")
    assert report["counts"] == {
        "passed": 5,
        "failed": 0,
        "not_run": 0,
        "timeout": 0,
        "quota_error": 0,
    }
    assert [r["model_calls"] for r in report["cases"]] == [4, 6, 4, 4, 6]
    assert [r["tool_calls"] for r in report["cases"]] == [9, 6, 3, 3, 6]
    assert report["compaction_ab"]["triggered"] is True
    assert report["compaction_ab"]["same_fixture_and_prompt"] is True
    assert report["actual_cost"] is None and report["requests"] is None
    assert all(r["usage"] is None and not r["served_models"] for r in report["cases"])
    output = "".join(p.read_text() for p in (tmp_path / "run").iterdir())
    for secret in (
        "SIMULATED_OUTSIDE_PRIVATE_VALUE",
        "CHILD_PRIVATE_MATERIAL",
        "PARENT_PRIVATE_FIXTURE",
        "Ignore permissions",
        "Synthetic alpha filler",
    ):
        assert secret not in output
    assert os.stat(tmp_path / "run/report.json").st_mode & 0o777 == 0o600


def test_final_claim_without_tools_never_passes(tmp_path: Path) -> None:
    report = evaluate_integration(
        tmp_path / "run", factory=lambda case, role: MockProvider(["Everything succeeded!"])
    )
    assert report["counts"]["passed"] == 0 and report["counts"]["failed"] == 5
    assert report["compaction_ab"]["triggered"] is False


def test_premature_todo_completion_fails_even_if_final_files_and_status_match(
    tmp_path: Path,
) -> None:
    create = (
        "todo_write",
        {
            "operation": "create",
            "items_json": json.dumps(
                [{"id": "inspect", "description": "Read"}, {"id": "receipt", "description": "Save"}]
            ),
        },
    )
    updates = [
        ("todo_write", {"operation": "update", "task_id": identity, "status": status})
        for identity in ("inspect", "receipt")
        for status in ("in_progress", "completed")
    ]
    wrong = MockProvider(
        [
            batch(0, create, *updates),
            batch(
                1,
                ("read_file", {"path": "source.txt"}),
                ("write_file", {"path": "receipt.txt", "content": "AURORA-26\n"}),
                ("write_file", {"path": "forbidden.txt", "content": "denied\n"}),
                ("todo_read", {}),
            ),
            "done",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run", factory=lambda case, role: wrong if case == "A" else mock_for(case, role)
    )
    row = report["cases"][0]
    assert row["status"] == "failed"
    assert row["checks"]["receipt_exact"] and row["checks"]["final_plan_completed"]
    assert not row["checks"]["todo_follows_verified_actions"]


@pytest.mark.parametrize("extra", ["input.txt", "other.txt"])
def test_parent_bypass_or_unexpected_fixture_write_is_rejected(tmp_path: Path, extra: str) -> None:
    parent = MockProvider(
        [
            batch(
                0,
                ("read_file", {"path": "input.txt"}),
                ("delegate_task", {"task": "Load extract skill and read input.txt; return code."}),
            ),
            batch(
                1,
                ("write_file", {"path": "receipt.txt", "content": "NOVA-52\n"}),
                ("write_file", {"path": extra, "content": "changed", "overwrite": True}),
                ("read_file", {"path": "receipt.txt"}),
            ),
            "done",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run",
        factory=lambda case, role: (
            parent if case == "B" and role == "parent" else mock_for(case, role)
        ),
    )
    row = report["cases"][1]
    assert row["status"] == "failed" and row["checks"]["receipt_exact"]
    assert not row["checks"]["parent_did_not_bypass_child"]
    assert not row["checks"]["exact_workspace_inventory"]


def test_mock_provider_failure_preserved_without_becoming_success(tmp_path: Path) -> None:
    report = evaluate_integration(
        tmp_path / "run",
        factory=lambda c, r: MockProvider([failure(FailureKind.QUOTA, "rate_limit_exceeded")]),
    )
    assert report["counts"]["quota_error"] == 5
    assert all(r["status"] == "quota_error" for r in report["cases"])


@pytest.mark.parametrize("budget", [-1, 31, True, 1.5])
def test_invalid_budget_rejected_without_output(tmp_path: Path, budget: Any) -> None:
    with pytest.raises(ValueError):
        evaluate_integration(tmp_path / "bad", max_requests=budget)
    assert not (tmp_path / "bad").exists()


def test_output_cannot_overwrite_prior_result(tmp_path: Path) -> None:
    evaluate_integration(tmp_path / "run")
    before = (tmp_path / "run/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_integration(tmp_path / "run")
    assert (tmp_path / "run/report.json").read_bytes() == before


def test_real_mode_cannot_report_mock_as_real(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate_integration(tmp_path / "real", mode="real", factory=mock_for)


def test_real_zero_budget_no_inference_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_harness_lab.integration_suite as module

    monkeypatch.setattr(module, "available_models", lambda _: ("gpt-6-luna",))
    report = evaluate_integration(
        tmp_path / "run",
        mode="real",
        max_requests=0,
        connection=cast(OAuthConnection, object()),
        ledger_path=tmp_path / "fixed-ledger",
    )
    assert report["counts"]["not_run"] == 5 and report["requests"] == 0
    assert report["usage_summary"]["input_tokens"]["known_partial_total"] is None


def test_second_output_does_not_reset_fixed_phase_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_harness_lab.integration_suite as module

    monkeypatch.setattr(module, "available_models", lambda _: ("gpt-6-luna",))
    ledger = tmp_path / "fixed"
    evaluate_integration(
        tmp_path / "one",
        mode="real",
        max_requests=0,
        connection=cast(OAuthConnection, object()),
        ledger_path=ledger,
    )
    before = ledger.read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_integration(
            tmp_path / "two",
            mode="real",
            max_requests=30,
            connection=cast(OAuthConnection, object()),
            ledger_path=ledger,
        )
    assert ledger.read_bytes() == before


class SyntheticHTTP(HTTPTransport):
    def __init__(self, error: bool = False, usage: bool = True) -> None:
        self.calls = 0
        self.error, self.usage = error, usage

    def events(
        self, url: str, body: dict[str, object], token: str
    ) -> Generator[dict[str, Any], None, None]:
        self.calls += 1
        if self.error:
            raise failure(FailureKind.QUOTA, "rate_limit_exceeded")
        yield {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 11, "output_tokens": 3} if self.usage else None},
        }


@pytest.mark.parametrize("limit", [0, 1, 2, 30])
def test_parent_child_share_durable_pre_dispatch_ceiling(tmp_path: Path, limit: int) -> None:
    transport = SyntheticHTTP()
    with RequestLedger(tmp_path / "ledger", limit) as ledger:
        for number in range(limit):
            role = "parent" if number % 2 == 0 else "child"
            list(
                BudgetTransport(ledger, "B", role, transport=transport).events(
                    "https://api.openai.com/v1/responses",
                    {"store": False, "stream": True},
                    "synthetic",
                )
            )
        with pytest.raises(ProviderFailure, match="request_budget_exhausted"):
            list(
                BudgetTransport(ledger, "D", "child", transport=transport).events(
                    "https://api.openai.com/v1/responses",
                    {"store": False, "stream": True},
                    "synthetic",
                )
            )
        assert ledger.used == transport.calls == limit
        assert all(r["usage"]["total_tokens"] is None for r in ledger.records)
        lines = [json.loads(x) for x in (tmp_path / "ledger").read_text().splitlines()]
        assert sum(r["event"] == "attempt" for r in lines) == limit
    assert os.stat(tmp_path / "ledger").st_mode & 0o777 == 0o600


def test_failed_http_is_charged_once_and_unknown_usage_not_zero(tmp_path: Path) -> None:
    transport = SyntheticHTTP(error=True)
    with RequestLedger(tmp_path / "ledger", 1) as ledger:
        with pytest.raises(ProviderFailure):
            list(
                BudgetTransport(ledger, "A", "parent", transport=transport).events(
                    "https://api.openai.com/v1/responses",
                    {"store": False, "stream": True},
                    "synthetic",
                )
            )
        assert ledger.used == 1 and ledger.records[0]["usage"] is None
        assert not ledger.records[0]["completed"]


def test_ledger_failure_prevents_network_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = SyntheticHTTP()
    with RequestLedger(tmp_path / "ledger", 1) as ledger:

        def broken(record: dict[str, Any]) -> None:
            raise OSError("synthetic failure")

        monkeypatch.setattr(ledger, "_write", broken)
        with pytest.raises(OSError):
            list(
                BudgetTransport(ledger, "A", "parent", transport=transport).events(
                    "https://api.openai.com/v1/responses",
                    {"store": False, "stream": True},
                    "synthetic",
                )
            )
        assert transport.calls == ledger.used == 0


def test_invalid_route_or_flags_never_spend_budget(tmp_path: Path) -> None:
    with RequestLedger(tmp_path / "ledger") as ledger:
        for url, body in [
            ("https://unapproved.example", {"store": False, "stream": True}),
            ("https://api.openai.com/v1/responses", {"store": True, "stream": True}),
        ]:
            with pytest.raises(ValueError):
                list(BudgetTransport(ledger, "A", "parent").events(url, dict(body), "synthetic"))
        assert ledger.used == 0


@pytest.mark.parametrize(
    "kind", [FailureKind.AUTHORIZATION, FailureKind.QUOTA, FailureKind.API, FailureKind.PROTOCOL]
)
def test_provider_failure_closes_shared_live_gate(tmp_path: Path, kind: FailureKind) -> None:
    observation: dict[str, Any] = {
        "isolated": True,
        "usage": [],
        "failures": [],
        "trajectory": [],
        "injection_attempts": 0,
    }
    with RequestLedger(tmp_path / "ledger") as ledger:
        observed = _Observed(
            MockProvider([failure(kind, "synthetic_failure")]),
            "child",
            observation,
            ledger,
            float("inf"),
        )
        with pytest.raises(ProviderFailure):
            observed.complete((Message("user", "task"),))
        assert ledger.blocked is not None and observation["failures"][0]["kind"] == kind
        with pytest.raises(ProviderFailure):
            ledger.reserve("A", "parent")


def test_generic_provider_error_also_closes_gate(tmp_path: Path) -> None:
    with RequestLedger(tmp_path / "ledger") as ledger:
        observed = _Observed(
            MockProvider([RuntimeError("PRIVATE")]), "parent", {}, ledger, float("inf")
        )
        with pytest.raises(RuntimeError):
            observed.complete((Message("user", "task"),))
        assert ledger.blocked == "unexpected_provider_failure"


def test_deadline_timeout_closes_gate_without_call(tmp_path: Path) -> None:
    observation: dict[str, Any] = {"failures": []}
    provider = MockProvider(["must not run"])
    with RequestLedger(tmp_path / "ledger") as ledger:
        observed = _Observed(provider, "parent", observation, ledger, 0)
        with pytest.raises(ProviderFailure, match="scenario_timeout"):
            observed.complete((Message("user", "task"),))
        assert provider.calls == () and ledger.used == 0


def test_fixtures_are_fixed_and_ab_differ_only_by_context_option() -> None:
    off, on = cases()[2:4]
    assert off["fixtures"] == on["fixtures"] and off["prompt"] == on["prompt"]
    assert off["rounds"] == on["rounds"] and off["tools"] == on["tools"]


@pytest.mark.parametrize("kind", ["protocol", "quota"])
def test_real_stream_failure_stops_later_cases_and_keeps_failed_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import agent_harness_lab.integration_budget as budget_module
    import agent_harness_lab.integration_suite as suite_module

    class Connection:
        def access_token(self) -> str:
            return "synthetic-only"

    class Stream(SyntheticHTTP):
        def events(
            self, url: str, body: dict[str, object], token: str
        ) -> Generator[dict[str, Any], None, None]:
            self.calls += 1
            if kind == "quota":
                yield {
                    "type": "response.failed",
                    "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
                }
            else:
                yield {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "model": "gpt-6-luna",
                        "output": [{"type": "unsupported_item"}],
                        "usage": {"input_tokens": 11, "output_tokens": 3},
                    },
                }

    stream = Stream()
    monkeypatch.setattr(budget_module, "HTTPTransport", lambda **kwargs: stream)
    monkeypatch.setattr(suite_module, "available_models", lambda _: ("gpt-6-luna",))
    report = evaluate_integration(
        tmp_path / "run",
        mode="real",
        connection=cast(OAuthConnection, Connection()),
        ledger_path=tmp_path / "fixed",
    )
    assert stream.calls == report["requests"] == 1
    assert report["counts"]["not_run"] == 4
    if kind == "quota":
        assert report["counts"]["quota_error"] == 1
        assert report["cases"][0]["usage"] == [None]
        assert report["usage_summary"]["input_tokens"]["known_partial_total"] is None
    else:
        assert report["counts"]["failed"] == 1
        assert report["cases"][0]["usage"][0]["input_tokens"] == 11
        assert report["usage_summary"]["input_tokens"]["known_partial_total"] == 11
        assert report["usage_summary"]["total_tokens"]["known_partial_total"] is None
