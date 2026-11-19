import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.retrieval_cli import main
from knowledge_ticket_agent.schemas import Dataset

from .conftest import DEMO, refresh


def test_cli_real_search_and_source_checks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["lqd-search", "first purchase credit", "--category", "billing_refund"]
    )
    main()
    output = json.loads(capsys.readouterr().out)
    assert output["search"]["hits"] and output["model_requests"] == 0
    assert all(item["valid"] for item in output["citation_checks"])
    assert output["semantic_support"] == "not_established_by_citation_validity"


def test_cli_load_error_distinct_from_no_match(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> None:
        raise FixtureError("resource_read_failed")

    monkeypatch.setattr("knowledge_ticket_agent.retrieval_cli.load_dataset", fail)
    monkeypatch.setattr(sys, "argv", ["lqd-search", "query"])
    with pytest.raises(SystemExit):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "load_error"
    assert result["error_code"] == "resource_read_failed"


def test_excerpts_are_bounded_and_full_get_is_distinct(copied: Path) -> None:
    manifest = json.loads((copied / "manifest.json").read_text())
    file = next(d["path"] for d in manifest["documents"] if d["kind"] == "faq")
    path = copied / file
    path.write_text(path.read_text() + " ".join(["longword"] * 280) + "\n")
    refresh(copied)
    retriever = Retriever(load_dataset(copied))
    result = retriever.search("longword")
    assert result.hits and all(h.truncated and len(h.excerpt) == 600 for h in result.hits)
    for hit in result.hits:
        full = retriever.get_chunk(hit.chunk_id)
        assert len(full.text) > 600 and full.text.startswith(hit.excerpt)
        assert len(full.text.encode()) <= 8192


def test_hidden_mapping_and_reports_are_not_distribution_data() -> None:
    with zipfile.ZipFile(DEMO / "dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl") as built:
        names = built.namelist()
        assert not any(n.endswith("retrieval_gold.json") or "/verification/" in n for n in names)
        assert "knowledge_ticket_agent/retrieval.py" in names
        entry = built.read("knowledge_ticket_agent_demo-0.1.0.dist-info/entry_points.txt").decode()
        assert "lqd-search" in entry and "lqd-retrieval-eval" in entry
    with tarfile.open(DEMO / "dist/knowledge_ticket_agent_demo-0.1.0.tar.gz") as built:
        assert not any(
            n.endswith("retrieval_gold.json") or "/verification/" in n for n in built.getnames()
        )


def test_empty_category_is_not_unsupported_query_success(dataset: Dataset) -> None:
    from knowledge_ticket_agent.schemas import Category

    result = Retriever(dataset).search("purchase", category=Category.OTHER)
    assert result.status == "no_match" and not result.hits
