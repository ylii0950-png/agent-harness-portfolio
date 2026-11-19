import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest
from knowledge_ticket_agent.retrieval import SearchResult
from knowledge_ticket_agent.retrieval_dev import load_dev, run_experiment
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever
from knowledge_ticket_agent.schemas import Category, Dataset

from .conftest import DEMO

REPORT = DEMO / "verification/m2r/dev-experiment.json"


def test_fixed_experiment_reproduces_and_only_uses_dev(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = load_dev(DEMO / "tests/dev_gold/m2r-v1", dataset)
    allowed = {c.query for c in dev.cases}
    original = StrategyRetriever.search
    seen: list[str] = []

    def checked(
        self: StrategyRetriever,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
        mode: str = "authority",
    ) -> SearchResult:
        assert query in allowed
        seen.append(query)
        return original(
            self, query, category=category, snapshot=snapshot, top_k=top_k, method=method, mode=mode
        )

    monkeypatch.setattr(StrategyRetriever, "search", checked)
    result = run_experiment(dataset, dev)
    assert json.loads(json.dumps(result)) == json.loads(REPORT.read_text())
    assert len(seen) == 5 * 12 * 2
    assert set(seen) == allowed
    assert result["v1_sealed_new_strategy_runs"] == 0


def test_selection_and_means_independently() -> None:
    report = json.loads(REPORT.read_text())
    candidates = report["candidates"]
    for candidate in candidates:
        authority = candidate["authority"]
        outcomes = authority["outcomes"]
        assert len(outcomes) == authority["denominator"] == 12
        for key in ("hit", "recall", "reciprocal_rank"):
            assert authority["aggregate"][key] == pytest.approx(
                sum(o["metrics"][key] for o in outcomes) / 12, abs=1e-15
            )
        for outcome in outcomes:
            assert outcome["metrics"]["top3"] == [h["chunk_id"] for h in outcome["hits"]]
            assert not outcome["error_code"]
    metrics = lambda c: tuple(  # noqa: E731
        c["authority"]["aggregate"][k] for k in ("recall", "reciprocal_rank", "hit")
    )
    expected = max(candidates, key=metrics)
    assert report["selection"]["recommended"] == expected["config"]["name"] == "A1"
    assert report["held_out_v2"] == "not_created_not_run"
    assert report["model_requests"] == 0
    assert report["api_token_usage"] is report["actual_cost"] is None


def test_old_dev_control_is_reproduced() -> None:
    report = json.loads(REPORT.read_text())
    # Values from the unchanged M2 development report, not new sealed selection.
    assert report["controls"]["bm25"]["aggregate"]["recall"] == 19 / 24
    assert report["controls"]["bm25"]["aggregate"]["reciprocal_rank"] == 17 / 24
    assert report["controls"]["keyword"]["aggregate"]["recall"] == 19 / 24
    assert report["controls"]["keyword"]["aggregate"]["reciprocal_rank"] == 2 / 3


def test_complete_required_anchor_diagnostics_are_not_ranking_recall(dataset: Dataset) -> None:
    dev = load_dev(DEMO / "tests/dev_gold/m2r-v1", dataset)
    report = json.loads(REPORT.read_text())
    supplement = json.loads((DEMO / "verification/m2r/dev-required-evidence.json").read_text())
    methods = {
        **report["controls"],
        **{c["config"]["name"]: c["authority"] for c in report["candidates"]},
    }
    for name, method in methods.items():
        full = 0
        for case, outcome, saved in zip(
            dev.cases, method["outcomes"], supplement["methods"][name]["outcomes"], strict=True
        ):
            retrieved = set(outcome["metrics"]["top3"])
            covered = [
                u.unit_id
                for u in case.required_units
                if any(set(bundle) <= retrieved for bundle in u.alternatives)
            ]
            missing = [u.unit_id for u in case.required_units if u.unit_id not in covered]
            assert saved["metrics"]["covered_units"] == covered
            assert saved["metrics"]["missing_units"] == missing
            full += not missing
        assert supplement["methods"][name]["full_coverage_cases"] == full
    assert supplement["methods"]["A1"]["full_coverage_cases"] == 2


def test_frozen_source_config_and_evidence() -> None:
    from knowledge_ticket_agent.retrieval_strategies import strategy

    lock = json.loads((DEMO / "verification/m2r/selection-lock.json").read_text())
    assert lock["selected_config"] == json.loads(json.dumps(asdict(strategy("A1"))))
    assert lock["selected_config_sha256"] == strategy("A1").sha256
    for relative, expected in lock["files_sha256"].items():
        assert hashlib.sha256((DEMO.parents[1] / relative).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("mode", ["authority", "faq_navigation"])
def test_installed_v2_cli(mode: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_ticket_agent.retrieval_v2_cli",
            "first purchase credit",
            "--strategy",
            "A1",
            "--mode",
            mode,
            "--category",
            "billing_refund",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=DEMO,
    )
    payload = json.loads(result.stdout)
    assert payload["model_requests"] == 0
    assert f":{mode}:" in payload["search"]["method"]
    assert payload["search"]["hits"]


def test_dev_cli_does_not_overwrite_evidence(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text("preserved")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_ticket_agent.retrieval_dev",
            "--dev-root",
            str(DEMO / "tests/dev_gold/m2r-v1"),
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and target.read_text() == "preserved"


def test_both_distributions_exclude_all_gold_and_reports() -> None:
    wheel = DEMO / "dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl"
    sdist = DEMO / "dist/knowledge_ticket_agent_demo-0.1.0.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        entry = next(n for n in names if n.endswith("entry_points.txt"))
        assert "lqd-search-v2" in archive.read(entry).decode()
        assert "lqd-retrieval-dev" in archive.read(entry).decode()
    with tarfile.open(sdist) as archive:
        names += archive.getnames()
    assert not any(
        part in name
        for name in names
        for part in ("dev_gold.json", "business_gold.json", "retrieval_gold.json", "verification/")
    )
