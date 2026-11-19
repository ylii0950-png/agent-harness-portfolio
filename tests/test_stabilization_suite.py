import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness_lab import integration_suite as suite
from agent_harness_lab.approval_diagnostics import ExactWriteApproval
from agent_harness_lab.integration import ProviderFailure
from agent_harness_lab.integration_budget import BudgetTransport, RequestLedger
from agent_harness_lab.oauth import OAuthConnection
from agent_harness_lab.stabilization_cases import cases, mock_for
from agent_harness_lab.verification_journal import stabilization_journal


def repository(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    common = root / ".git"
    (common / "objects").mkdir(parents=True)
    (common / "HEAD").write_text("ref: refs/heads/main\n")
    (common / "config").write_text(
        '[remote "origin"]\nurl = https://github.com/liyongyong455-cell/agent-harness-lab.git\n'
    )
    (root / "pyproject.toml").touch()
    return common


def test_mock_report_and_trace_retain_no_private_content_or_payload_hashes(tmp_path: Path) -> None:
    report = suite.evaluate_integration(tmp_path / "run", suite="phase26s", max_requests=11)
    assert report["counts"] == dict(passed=1, failed=0, not_run=0, timeout=0, quota_error=0)
    assert report["grader_version"] == 3
    assert report["requests"] is None and report["actual_cost"] is None
    case = report["cases"][0]
    assert case["trajectory_sha256"] is None
    assert case["approvals"][1]["categories"] == ["target_mismatch", "content_mismatch"]
    payload = "".join(p.read_text() for p in (tmp_path / "run").iterdir())
    for secret in (
        "AURORA-26",
        "CODE:",
        "forbidden.txt",
        "receipt.txt",
        "source.txt",
        "SIMULATED_OUTSIDE_PRIVATE_VALUE",
        "access_token",
        "refresh_token",
        hashlib.sha256(b"AURORA-26\n").hexdigest(),
    ):
        assert secret not in payload


@pytest.mark.parametrize("limit", [12, 30, -1, True])
def test_a_s_rejects_non_authorized_budget_before_creating_files(
    tmp_path: Path, limit: Any
) -> None:
    with pytest.raises(ValueError):
        suite.evaluate_integration(tmp_path / "run", suite="phase26s", max_requests=limit)
    assert not (tmp_path / "run").exists()


def test_persistent_ceiling_includes_children_failed_attempts_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "ledger"

    class BrokenTransport:
        def events(self, *_: Any) -> Any:
            raise OSError("synthetic offline transport failure")
            yield  # pragma: no cover

    with RequestLedger(path, 11) as ledger:
        transport = BudgetTransport(ledger, "A_S", "child", transport=cast(Any, BrokenTransport()))
        with pytest.raises(OSError):
            list(
                transport.events(
                    "https://api.openai.com/v1/responses",
                    {"store": False, "stream": True},
                    "synthetic",
                )
            )
        for index in range(10):
            ledger.reserve("A_S", "parent" if index % 2 else "child")
        with pytest.raises(ProviderFailure):
            ledger.reserve("A_S", "child")
        assert ledger.used == 11
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len([r for r in rows if r["event"] == "attempt"]) == 11
    assert rows[2]["event"] == "outcome" and rows[2]["completed"] is False
    assert path.stat().st_mode & 0o777 == 0o600
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        RequestLedger(path, 11)
    assert path.read_bytes() == before


def test_a_s_real_mode_cannot_select_a_fresh_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    repository(tmp_path)
    with pytest.raises(ValueError, match="fixed phase ledger"):
        suite.evaluate_integration(
            tmp_path / "run",
            suite="phase26s",
            mode="real",
            max_requests=11,
            ledger_path=tmp_path / "fresh",
        )
    assert not (tmp_path / "run").exists()


def test_zero_budget_cannot_access_credentials_and_restart_cannot_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    repository(tmp_path)

    def prohibited(*_: Any) -> Any:
        raise AssertionError("credentials/catalog must not be accessed")

    monkeypatch.setattr(suite, "OAuthConnection", prohibited)
    monkeypatch.setattr(suite, "available_models", prohibited)
    report = suite.evaluate_integration(
        tmp_path / "run", suite="phase26s", mode="real", max_requests=0
    )
    assert report["requests"] == 0 and report["counts"]["not_run"] == 1
    with pytest.raises(FileExistsError):
        suite.evaluate_integration(
            tmp_path / "second", suite="phase26s", mode="real", max_requests=11
        )
    assert not (tmp_path / "second").exists()


def test_primary_and_linked_worktree_share_one_durable_phase_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary"
    common = repository(primary)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "pyproject.toml").touch()
    worktree = common / "worktrees" / "linked"
    worktree.mkdir(parents=True)
    (worktree / "commondir").write_text("../..\n")
    (worktree / "gitdir").write_text(str(linked / ".git") + "\n")
    (linked / ".git").write_text("gitdir: " + str(worktree) + "\n")
    fixed = stabilization_journal(primary)
    assert fixed == stabilization_journal(linked)
    fixed.parent.mkdir()
    with RequestLedger(fixed, 11) as ledger:
        for _ in range(11):
            ledger.reserve("A_S", "parent")
    before = fixed.read_bytes()
    for directory in (primary, linked):
        monkeypatch.chdir(directory)
        with pytest.raises(FileExistsError):
            suite.evaluate_integration(
                directory / "new-output", suite="phase26s", mode="real", max_requests=11
            )
        assert not (directory / "new-output").exists()
    assert fixed.read_bytes() == before


@pytest.mark.parametrize("kind", ["markers", "wrong_origin", "bad_backlink"])
def test_untrusted_checkout_identity_cannot_start_a_new_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    if kind == "markers":
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").touch()
    else:
        common = repository(tmp_path)
        if kind == "wrong_origin":
            (common / "config").write_text(
                '[remote "origin"]\nurl = https://example.invalid/other\n'
            )
        else:
            linked = tmp_path / "linked"
            linked.mkdir()
            (linked / "pyproject.toml").touch()
            worktree = common / "worktrees" / "linked"
            worktree.mkdir(parents=True)
            (worktree / "commondir").write_text("../..")
            (worktree / "gitdir").write_text(str(tmp_path / ".git"))
            (linked / ".git").write_text("gitdir: " + str(worktree))
            tmp_path = linked
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        suite.evaluate_integration(tmp_path / "run", suite="phase26s", mode="real", max_requests=11)
    assert not (tmp_path / "run").exists()


def test_fixed_data_answer_and_old_suites_are_unchanged() -> None:
    from agent_harness_lab.remediation_cases import cases as old_cases

    case = cases()[0]
    assert case["fixtures"] == old_cases()[0]["fixtures"]
    assert case["fixtures"] == {"source.txt": "CODE: AURORA-26\n"}
    assert "AURORA-26" not in case["prompt"]
    assert "predicate" not in case["prompt"].lower()
    assert case["rounds"] == case["total_rounds"] == 11
    with pytest.raises(ValueError):
        mock_for("A_S", "child")


@pytest.mark.parametrize("value", ["NIMBUS-19", "TEAL-82", "synthetic-value"])
def test_exact_approval_is_generic_not_a_fixed_answer(value: str) -> None:
    from agent_harness_lab.permissions import ApprovalRequest, Permission

    approval = ExactWriteApproval("synthetic.txt", value + "\n")
    request = ApprovalRequest(
        "opaque", "write_file", Permission.WRITE, {"path": "synthetic.txt", "content": value + "\n"}
    )
    assert approval(request) is True
    wrong = ApprovalRequest(
        "other", "write_file", Permission.WRITE, {"path": "synthetic.txt", "content": "AURORA-26\n"}
    )
    assert approval(wrong) is False


def test_a_s_real_cannot_substitute_scripted_factory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot substitute"):
        suite.evaluate_integration(
            tmp_path / "run",
            mode="real",
            suite="phase26s",
            max_requests=11,
            factory=mock_for,
            connection=cast(OAuthConnection, object()),
        )
