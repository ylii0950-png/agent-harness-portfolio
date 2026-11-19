import json
import subprocess
import sys
import tarfile
import zipfile

from knowledge_ticket_agent.evaluation_integrity import verify_frozen_inputs

from .conftest import DEMO


def test_public_cli_without_gold() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "knowledge_ticket_agent.evaluation_cli", "--public-check"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["private_gold_loaded"] is False
    assert data["formal_scoring"] == "NOT_RUN"


def test_cli_refuses_formal_scoring_flag() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_ticket_agent.evaluation_cli",
            "--public-check",
            "--score",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_cli_does_not_disclose_bad_private_paths() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_ticket_agent.evaluation_cli",
            "--vault",
            "/synthetic/secret/location",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "secret/location" not in result.stdout
    assert json.loads(result.stdout)["error_code"] == "vault_location"


def test_distributions_have_tools_not_private_material() -> None:
    with zipfile.ZipFile(
        DEMO / "dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl"
    ) as archive:
        names = archive.namelist()
        assert "knowledge_ticket_agent/evaluation_maintenance.py" in names
        assert "knowledge_ticket_agent/evaluation_custody.py" in names
    with tarfile.open(DEMO / "dist/knowledge_ticket_agent_demo-0.1.0.tar.gz") as archive:
        names += archive.getnames()
    assert not any(
        part in n
        for n in names
        for part in (
            ".git/",
            "key.bin",
            ".aesgcm",
            "heldout_gold.json",
            "initial-analysis",
            "verification/",
        )
    )


def test_precreation_retriever_lock_matches_all_existing_inputs() -> None:
    root = DEMO.parents[1]
    result = verify_frozen_inputs(root)
    assert result["original_inputs"] == 192
    assert result["unchanged_non_ci_inputs"] == 191
