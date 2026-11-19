import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError
from knowledge_ticket_agent.retrieval_dev import DEV_SHA256, DevGold, load_dev, validate_dev
from knowledge_ticket_agent.schemas import Dataset, Split

from .conftest import DEMO


@pytest.fixture
def dev(dataset: Dataset) -> DevGold:
    return load_dev(DEMO / "tests/dev_gold/m2r-v1", dataset)


def test_dev_only_exact_mechanical_copy_of_original(dev: DevGold) -> None:
    original = json.loads((DEMO / "tests/retrieval_gold/v1/retrieval_gold.json").read_text())
    from dataclasses import asdict

    assert json.loads(json.dumps([asdict(c) for c in dev.cases])) == [
        c for c in original["cases"] if c["split"] == "dev"
    ]
    assert len(dev.cases) == 12 and {c.split for c in dev.cases} == {Split.DEV}


@pytest.mark.parametrize("mutation", ["query", "unit", "eval", "missing", "duplicate"])
def test_direct_dev_object_tampering_rejected(
    dataset: Dataset, dev: DevGold, mutation: str
) -> None:
    first = dev.cases[0]
    if mutation == "missing":
        changed = replace(dev, cases=dev.cases[:-1])
    elif mutation == "duplicate":
        changed = replace(dev, cases=(first, *dev.cases[:-1]))
    else:
        case = (
            replace(first, query="different")
            if mutation == "query"
            else replace(first, units=())
            if mutation == "unit"
            else replace(first, split=Split.EVAL, case_id="KT-E01")
        )
        changed = replace(dev, cases=(case, *dev.cases[1:]))
    with pytest.raises(FixtureError):
        validate_dev(changed, dataset)


def test_changed_dev_file_and_freeze_cannot_relabel_release(
    dataset: Dataset, tmp_path: Path
) -> None:
    import hashlib

    root = tmp_path / "dev"
    shutil.copytree(DEMO / "tests/dev_gold/m2r-v1", root)
    raw = json.loads((root / "dev_gold.json").read_text())
    raw["cases"][0]["query"] = "easier query"
    payload = json.dumps(raw).encode()
    (root / "dev_gold.json").write_bytes(payload)
    freeze = json.loads((root / "freeze.json").read_text())
    freeze.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    (root / "freeze.json").write_text(json.dumps(freeze))
    assert freeze["sha256"] != DEV_SHA256
    with pytest.raises(FixtureError):
        load_dev(root, dataset)


def test_dev_loader_does_not_accept_old_eval_gold_directory(dataset: Dataset) -> None:
    with pytest.raises(FixtureError):
        load_dev(DEMO / "tests/retrieval_gold/v1", dataset)


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink", "oversize"])
def test_dev_filesystem_boundary(dataset: Dataset, tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "dev"
    shutil.copytree(DEMO / "tests/dev_gold/m2r-v1", root)
    target = root / "dev_gold.json"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (root / "unexpected.json").write_text("{}")
    elif mutation == "symlink":
        outside = tmp_path / "outside.json"
        target.rename(outside)
        target.symlink_to(outside)
    else:
        target.write_text("x" * (256 * 1024 + 1))
    with pytest.raises(FixtureError):
        load_dev(root, dataset)


def test_dev_cli_has_no_eval_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    from knowledge_ticket_agent.retrieval_dev import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lqd-retrieval-dev",
            "--dev-root",
            str(DEMO / "tests/dev_gold/m2r-v1"),
            "--output",
            str(tmp_path / "not-created.json"),
            "--split",
            "eval",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not (tmp_path / "not-created.json").exists()
