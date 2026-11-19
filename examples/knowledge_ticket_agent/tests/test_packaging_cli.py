import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from knowledge_ticket_agent import resource_root
from knowledge_ticket_agent.cli import main

from .conftest import DEMO, REPO


def test_cli_public_validation_never_pretends_gold_was_checked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["lqd-validate"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "valid" and result["business_gold"] == "not_loaded"
    assert result["gold_cases"] is None and result["model_requests"] == 0


def test_cli_explicit_trusted_gold_validation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["lqd-validate", "--gold-root", str(DEMO / "tests/gold/v1")])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["gold_cases"] == 36 and result["retrieval_gold"] == "not_created"


def test_cli_failure_reports_generic_error_without_raw_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    private = tmp_path / "sensitive-unavailable"
    monkeypatch.setattr(sys, "argv", ["lqd-validate", "--root", str(private)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert str(private) not in output and json.loads(output)["status"] == "invalid"


def test_real_wheel_and_sdist_inventory() -> None:
    wheel = DEMO / "dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl"
    archive = DEMO / "dist/knowledge_ticket_agent_demo-0.1.0.tar.gz"
    with zipfile.ZipFile(wheel) as built:
        names = built.namelist()
        assert "knowledge_ticket_agent/py.typed" in names
        assert sum(n.endswith(".md") and "/resources/v1/documents/" in n for n in names) == 20
        assert not any(
            "business_gold.json" in n or "freeze.json" in n or "/tests/" in n for n in names
        )
        manifest = json.loads(built.read("knowledge_ticket_agent/resources/v1/manifest.json"))
        assert manifest["counts"]["eval"] == 24 and manifest["counts"]["dev"] == 12
        installed = resource_root().parents[1]
        for name in names:
            if name.startswith("knowledge_ticket_agent/") and not name.endswith("/"):
                relative = name.removeprefix("knowledge_ticket_agent/")
                assert (installed / relative).read_bytes() == built.read(name), relative
        metadata = built.read("knowledge_ticket_agent_demo-0.1.0.dist-info/METADATA").decode()
        assert "Requires-Dist: agent-harness-lab==0.9.0" in metadata
    with tarfile.open(archive) as built:
        names = built.getnames()
        assert sum(n.endswith(".md") and "/resources/v1/documents/" in n for n in names) == 20
        assert not any("business_gold.json" in n or "/tests/" in n for n in names)
        assert any(n.endswith("/knowledge_ticket_agent/gold.py") for n in names)
    assert "/site-packages/" in str(resource_root())  # Installed artifact, no src path injection.


def test_core_artifact_remains_independent() -> None:
    with zipfile.ZipFile(REPO / "dist/agent_harness_lab-0.9.0-py3-none-any.whl") as built:
        assert not any(
            "knowledge_ticket_agent" in n or n.startswith("examples/") for n in built.namelist()
        )
