import hashlib
import json
import shutil
import socket
from pathlib import Path
from typing import Any

import pytest
from knowledge_ticket_agent import load_dataset, resource_root
from knowledge_ticket_agent.gold import BusinessGold, load_gold
from knowledge_ticket_agent.schemas import Dataset

DEMO = Path(__file__).resolve().parents[1]
REPO = DEMO.parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("M1 must remain offline and credential-free")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    from agent_harness_lab.oauth import CredentialStore

    monkeypatch.setattr(CredentialStore, "__init__", denied)


@pytest.fixture
def dataset() -> Dataset:
    return load_dataset()


@pytest.fixture
def gold(dataset: Dataset) -> BusinessGold:
    return load_gold(DEMO / "tests/gold/v1", dataset, public_root=resource_root())


@pytest.fixture
def copied(tmp_path: Path) -> Path:
    target = tmp_path / "public"
    shutil.copytree(resource_root(), target)
    return target


def rewrite(root: Path, path: str, value: Any) -> None:
    (root / path).write_text(json.dumps(value, indent=2) + "\n")
    refresh(root)


def refresh(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["resources"]:
        path = root / item["path"]
        if path.is_file():
            data = path.read_bytes()
            item.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
