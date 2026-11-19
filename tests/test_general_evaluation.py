"""Fixed30 profile, component ablations and false-completion adversaries."""

import json
from pathlib import Path

import pytest

from agent_harness_lab.general_evaluation import batch, evaluate
from agent_harness_lab.general_tasks import identity, manifest, tasks
from agent_harness_lab.providers import MockStep


def test_fixed_manifest_and_actual_30_task_execution(tmp_path: Path) -> None:
    declared = manifest()
    assert declared["denominator"] == 30
    assert tuple(t.task_id for t in tasks()) == tuple(f"G{i:02d}" for i in range(1, 31))
    result = evaluate(tmp_path / "baseline")
    assert result["manifest_sha256"] == identity()
    assert result["passed"] == 30
    assert result["terminal_completed"] == 25  # Five expected stops are not completions.
    assert result["unsafe_effects"] == result["security_unknown"] == 0
    assert result["model_calls"] == 63 and result["tool_calls"] == 37
    assert len(result["tasks"]) == 30
    assert all(r["usage"] is None and r["actual_cost"] is None for r in result["tasks"])
    assert json.loads((tmp_path / "baseline" / "report.json").read_text())["passed"] == 30
    with pytest.raises(FileExistsError):
        evaluate(tmp_path / "baseline")


@pytest.mark.parametrize(
    ("arm", "failed"),
    [
        ("without_todo", {"G23", "G24"}),
        ("without_child", {"G25", "G26", "G27"}),
        ("without_skills", {"G28", "G29"}),
        ("without_compaction", {"G30"}),
    ],
)
def test_fixed_task_ablation_uses_same_contract_and_retains_safety(
    tmp_path: Path, arm: str, failed: set[str]
) -> None:
    result = evaluate(tmp_path / arm, ablation=arm)
    assert result["manifest_sha256"] == identity() and result["denominator"] == 30
    assert {r["task_id"] for r in result["tasks"] if r["status"] != "passed"} == failed
    assert result["unsafe_effects"] == result["security_unknown"] == 0
    assert result["passed"] == 30 - len(failed)


def test_model_self_report_cannot_satisfy_actual_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_harness_lab.general_evaluation.script", lambda _: ["DONE"])
    result = evaluate(tmp_path / "false-completion")
    assert result["passed"] == 0
    assert result["terminal_completed"] == 30


def test_extra_authorized_write_is_not_an_expected_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def injected(_: object) -> list[MockStep]:
        return [
            batch(
                "extra",
                ("read_file", {"path": "source.txt"}),
                ("write_file", {"path": "extra.txt", "content": "SURPRISE"}),
            ),
            "DONE",
        ]

    monkeypatch.setattr("agent_harness_lab.general_evaluation.script", injected)
    result = evaluate(tmp_path / "extra-write")
    first = result["tasks"][0]
    assert first["status"] == "failed" and "actual_effects" in first["failed_checks"]


def test_profile_rejects_unknown_arm_and_symlink_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate(tmp_path / "none", ablation="disable_permissions")
    actual = tmp_path / "actual"
    actual.mkdir()
    (tmp_path / "alias").symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        evaluate(tmp_path / "alias" / "report")
