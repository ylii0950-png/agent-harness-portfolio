import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from knowledge_ticket_agent.evaluation_integrity import (
    ATTESTATION_SHA256,
    COMPAT,
    LOCK,
    LOCK_SHA256,
    NEW_CI_SHA256,
    OLD_CI_SHA256,
    _read,
    approved_ci_transition,
    verify_frozen_inputs,
)
from knowledge_ticket_agent.strict import FixtureError

from .conftest import REPO


@pytest.fixture
def frozen_tree(tmp_path: Path) -> Path:
    lock = json.loads((REPO / LOCK).read_text())
    attest = json.loads((REPO / COMPAT / "attestation.json").read_text())
    freeze = json.loads(
        (REPO / "examples/knowledge_ticket_agent/verification/m2e/freeze-manifest.json").read_text()
    )
    paths = (
        set(lock["files_sha256"])
        | set(attest["historical_evidence_sha256"])
        | set(freeze["maintenance_tools_sha256"])
        | {COMPAT + "ci-before.yml", COMPAT + "attestation.json"}
    )
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    return tmp_path


def test_precise_compatibility_retains_original_contract(frozen_tree: Path) -> None:
    result = verify_frozen_inputs(frozen_tree)
    assert result["original_inputs"] == 192
    assert result["unchanged_non_ci_inputs"] == 191
    assert result["retained_historical_evidence"] == 14
    assert result["retained_maintenance_tools"] == 3
    assert result["private_gold_loaded"] is False
    assert result["formal_scoring"] == "NOT_AUTHORIZED_NOT_RUN"
    lock = json.loads((frozen_tree / LOCK).read_text())
    assert lock["base_commit"] == "958a124111551a60779d6d8853f97dd49c8f264f"
    assert hashlib.sha256((frozen_tree / LOCK).read_bytes()).hexdigest() == LOCK_SHA256
    assert (
        hashlib.sha256((frozen_tree / COMPAT / "attestation.json").read_bytes()).hexdigest()
        == ATTESTATION_SHA256
    )


def test_old_ci_and_exact_patch_are_bound() -> None:
    old = (REPO / COMPAT / "ci-before.yml").read_bytes()
    new = (REPO / ".github/workflows/ci.yml").read_bytes()
    assert hashlib.sha256(old).hexdigest() == OLD_CI_SHA256
    assert hashlib.sha256(new).hexdigest() == NEW_CI_SHA256
    assert approved_ci_transition(old) == new
    # Every job step, including M2's offline CLI, is preserved byte-for-byte.
    old_steps = old.split(b"    steps:\n")
    new_steps = new.split(b"    steps:\n")
    assert (
        old_steps[1].split(b"\n  demo_checks:\n")[0] == new_steps[1].split(b"\n  demo_checks:\n")[0]
    )
    assert old_steps[2] == new_steps[2]
    assert b"M2 offline retrieval and citation CLI" in new


@pytest.mark.parametrize(
    "relative",
    [
        LOCK,
        COMPAT + "attestation.json",
        COMPAT + "ci-before.yml",
        ".github/workflows/ci.yml",
        "examples/knowledge_ticket_agent/src/knowledge_ticket_agent/retrieval_strategies.py",
        "examples/knowledge_ticket_agent/verification/m2r/selection-lock.json",
        "examples/knowledge_ticket_agent/verification/m2/eval-locked.json",
        "examples/knowledge_ticket_agent/verification/m2e/freeze-manifest.json",
        "examples/knowledge_ticket_agent/verification/m2e/dataset-qa.json",
        "docs/phase3-m2e-verification.md",
        "examples/knowledge_ticket_agent/src/knowledge_ticket_agent/evaluation_custody.py",
        "examples/knowledge_ticket_agent/src/knowledge_ticket_agent/evaluation_maintenance.py",
        "examples/knowledge_ticket_agent/src/knowledge_ticket_agent/evaluation_cli.py",
        "src/agent_harness_lab/permissions.py",
        "tests/test_workspace.py",
    ],
)
def test_any_digest_or_history_change_fails_closed(frozen_tree: Path, relative: str) -> None:
    path = frozen_tree / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(FixtureError):
        verify_frozen_inputs(frozen_tree)


def test_rewriting_both_lock_and_current_file_cannot_self_authorize(frozen_tree: Path) -> None:
    relative = "src/agent_harness_lab/permissions.py"
    (frozen_tree / relative).write_bytes(b"synthetic unauthorized replacement")
    lock = json.loads((frozen_tree / LOCK).read_text())
    lock["files_sha256"][relative] = hashlib.sha256(
        (frozen_tree / relative).read_bytes()
    ).hexdigest()
    (frozen_tree / LOCK).write_text(json.dumps(lock))
    with pytest.raises(FixtureError, match="original_lock_digest"):
        verify_frozen_inputs(frozen_tree)


def test_editing_attestation_cannot_authorize_another_ci(frozen_tree: Path) -> None:
    (frozen_tree / ".github/workflows/ci.yml").write_bytes(b"synthetic permissive workflow")
    path = frozen_tree / COMPAT / "attestation.json"
    a = json.loads(path.read_text())
    a["integrated_ci_sha256"] = hashlib.sha256(
        (frozen_tree / ".github/workflows/ci.yml").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(a))
    with pytest.raises(FixtureError, match="attestation_digest"):
        verify_frozen_inputs(frozen_tree)


def test_original_archive_required(frozen_tree: Path) -> None:
    (frozen_tree / COMPAT / "ci-before.yml").unlink()
    with pytest.raises(FixtureError, match="integrity_read"):
        verify_frozen_inputs(frozen_tree)


@pytest.mark.parametrize("parent", [False, True])
def test_symlinks_never_substitute_evidence(frozen_tree: Path, parent: bool) -> None:
    path = frozen_tree / (".github/workflows" if parent else COMPAT + "ci-before.yml")
    moved = path.with_name("original")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=parent)
    with pytest.raises(FixtureError, match="integrity_read"):
        verify_frozen_inputs(frozen_tree)


def test_ci_contains_no_private_custody_or_network_scoring_steps() -> None:
    raw = (REPO / ".github/workflows/ci.yml").read_text()
    for token in ["m2e-steward-private", "--vault", "key.bin", "upload-artifact", "--score"]:
        assert token not in raw
    assert "persist-credentials: false" in raw
    assert "contents: read" in raw
    assert "pull_request_target" not in raw


def test_public_integrity_cli_and_formal_rejection() -> None:
    command = [sys.executable, "-m", "knowledge_ticket_agent.evaluation_integrity"]
    good = subprocess.run(
        command + ["--repository-root", str(REPO)], capture_output=True, text=True, check=True
    )
    assert json.loads(good.stdout)["private_gold_loaded"] is False
    denied = subprocess.run(command + ["--score"], capture_output=True, text=True)
    assert denied.returncode == 2


def test_bad_root_cli_error_does_not_disclose_path() -> None:
    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_ticket_agent.evaluation_integrity",
            "--repository-root",
            "/synthetic/PRIVATE-SENTINEL",
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "PRIVATE-SENTINEL" not in bad.stdout + bad.stderr


@pytest.mark.parametrize("relative", ["../private", "/private", "a/../b", "a//b", "./a", ""])
def test_invalid_integrity_paths_rejected(tmp_path: Path, relative: str) -> None:
    with pytest.raises(FixtureError, match="integrity_path"):
        _read(tmp_path, relative)


def test_root_symlink_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(FixtureError, match="integrity_symlink"):
        _read(linked, "file")


def test_oversized_integrity_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "large"
    path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(FixtureError, match="integrity_file"):
        _read(tmp_path, "large")
