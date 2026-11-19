import json
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError, resource_root
from knowledge_ticket_agent.chunking import build_index
from knowledge_ticket_agent.gold import BusinessGold
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.retrieval_eval import evaluate
from knowledge_ticket_agent.retrieval_gold import load_retrieval_gold
from knowledge_ticket_agent.schemas import Dataset, Split

from .conftest import DEMO


@pytest.mark.parametrize(
    "split,filename,denominator",
    [(Split.DEV, "dev-initial.json", 12), (Split.EVAL, "eval-locked.json", 18)],
)
@pytest.mark.parametrize("method", ["bm25", "keyword"])
def test_locked_reports_reproduce_without_tuning(
    dataset: Dataset, gold: BusinessGold, split: Split, filename: str, denominator: int, method: str
) -> None:
    index = build_index(dataset)
    mapping = load_retrieval_gold(
        DEMO / "tests/retrieval_gold/v1", dataset, gold, index, public_root=resource_root()
    )
    actual = evaluate(Retriever(dataset), mapping, split, method, data=dataset, business=gold)
    saved = json.loads((DEMO / "verification/m2" / filename).read_text())[method]
    assert json.loads(json.dumps(actual)) == saved
    assert actual["denominator"] == denominator
    assert actual["required_evidence"] == {
        "fixed_eval_denominator": 20,
        "mapping_ready": True,
        "runtime_coverage": None,
        "status": "not_run_M4_M5",
    }


def test_published_metrics_have_independent_arithmetic() -> None:
    reports = json.loads((DEMO / "verification/m2/eval-locked.json").read_text())
    for report in reports.values():
        totals = [0.0, 0.0, 0.0, 0.0]
        assert len(report["outcomes"]) == 18
        for case in report["outcomes"]:
            top = list(dict.fromkeys(case["metrics"]["top3"]))[:3]
            retrieved = set(top)
            units = case["units"]
            relevant = {c for u in units for b in u["alternatives"] for c in b}
            complete = sum(any(set(b) <= retrieved for b in u["alternatives"]) for u in units)
            rank = next((i for i, c in enumerate(top, 1) if c in relevant), None)
            totals[0] += bool(retrieved & relevant)
            totals[1] += complete / len(units)
            totals[2] += 0.0 if rank is None else 1 / rank
            totals[3] += len(retrieved & relevant) / len(relevant)
        expected = dict(
            zip(
                ("hit_at_3", "macro_recall_at_3", "mrr_at_3", "auxiliary_raw_chunk_recall_at_3"),
                (n / 18 for n in totals),
                strict=True,
            )
        )
        assert report["aggregate"] == expected
        assert report["quality_gate_met"] == (
            expected["macro_recall_at_3"] >= 0.9 and expected["mrr_at_3"] >= 0.8
        )
        assert report["api_token_usage"] is None and report["actual_cost"] is None


def test_retrieval_errors_are_zero_with_fixed_denominator(
    dataset: Dataset, gold: BusinessGold, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = build_index(dataset)
    mapping = load_retrieval_gold(
        DEMO / "tests/retrieval_gold/v1", dataset, gold, index, public_root=resource_root()
    )
    retriever = Retriever(dataset)

    def fail(*args: object, **kwargs: object) -> None:
        raise FixtureError("synthetic_retrieval_error")

    monkeypatch.setattr(retriever, "search", fail)
    report = evaluate(retriever, mapping, Split.EVAL, "bm25", data=dataset, business=gold)
    assert report["denominator"] == 18 and report["quality_gate_met"] is False
    assert report["aggregate"] == {
        "hit_at_3": 0.0,
        "macro_recall_at_3": 0.0,
        "mrr_at_3": 0.0,
        "auxiliary_raw_chunk_recall_at_3": 0.0,
    }


def test_official_report_output_cannot_overwrite_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from knowledge_ticket_agent.retrieval_eval import main

    target = tmp_path / "report.json"
    target.write_text("original report")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lqd-retrieval-eval",
            "--business-gold-root",
            str(DEMO / "tests/gold/v1"),
            "--gold-root",
            str(DEMO / "tests/retrieval_gold/v1"),
            "--split",
            "dev",
            "--output",
            str(target),
        ],
    )
    with pytest.raises(SystemExit):
        main()
    assert target.read_text() == "original report"
    assert json.loads(capsys.readouterr().out)["error_code"] == "report_or_resource_unavailable"
