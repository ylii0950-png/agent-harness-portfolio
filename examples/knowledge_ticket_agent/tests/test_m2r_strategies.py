import math
from dataclasses import replace
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.retrieval_strategies import (
    CANDIDATES,
    FieldDocument,
    StrategyRetriever,
    diversify,
    field_rank,
    strategy,
)
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind


def test_weighted_fields_independent_hand_calculation() -> None:
    docs = (
        FieldDocument("a", "red red", "blue", "red green"),
        FieldDocument("b", "blue", "", "blue green green"),
    )
    title = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / 1.5))
    body = math.log(2) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / 2.5))
    result = field_rank(docs, "red red", weights=(2, 2, 1))
    assert result[0][0] == "a" and result[0][1] == pytest.approx(2 * title + body)
    assert len(result) == 1
    assert result == field_rank(docs, "red", weights=(2, 2, 1))


@pytest.mark.parametrize("query", ["", "unmatched", "!!!"])
def test_empty_and_nonmatching_field_scores(query: str) -> None:
    assert field_rank((FieldDocument("a", "one", "", ""),), query, weights=(1, 1, 2)) == ()
    assert field_rank((), query, weights=(1, 1, 2)) == ()


def test_empty_fields_contribute_zero_and_each_field_weight_is_independent() -> None:
    single = (FieldDocument("a", "red", "", ""),)
    assert field_rank(single, "red", weights=(2, 2, 1))[0][1] == pytest.approx(
        2 * math.log(1 + 1 / 3)
    )
    body = (FieldDocument("a", "", "", "red"),)
    assert field_rank(body, "red", weights=(1, 1, 2))[0][1] == pytest.approx(
        2 * math.log(1 + 1 / 3)
    )


@pytest.mark.parametrize("weights", [(0, 1, 1), (-1, 2, 1), (True, 1, 1), (11, 1, 1)])
def test_invalid_field_weights_rejected(weights: tuple[int, int, int]) -> None:
    with pytest.raises(FixtureError, match="field_weights"):
        field_rank((FieldDocument("a", "x", "", ""),), "x", weights=weights)


def test_diversity_hand_calculation_not_gold_coverage() -> None:
    docs = (
        FieldDocument("a", "red blue", "", ""),
        FieldDocument("b", "red blue", "", ""),
        FieldDocument("c", "green", "", ""),
    )
    result = diversify((("a", 10.0), ("b", 9.0), ("c", 8.0)), docs, strength=0.5, top_k=3)
    assert [r.identifier for r in result] == ["a", "c", "b"]
    assert [r.score for r in result] == pytest.approx([1, 0.8, 0.4])
    assert [r.base_score for r in result] == [10, 8, 9]


def test_diversity_has_no_document_quota_and_never_inserts_zero_match() -> None:
    docs = (
        FieldDocument("same:1", "red", "", ""),
        FieldDocument("same:2", "blue", "", ""),
        FieldDocument("unused", "green", "", ""),
    )
    ranked = (("same:1", 10.0), ("same:2", 9.0))
    result = diversify(ranked, docs, strength=0.5, top_k=3)
    assert [r.identifier for r in result] == ["same:1", "same:2"]
    assert diversify((), docs, strength=0.5, top_k=3) == ()


@pytest.mark.parametrize("strength", [-1.0, 1.01, math.nan, math.inf, True])
def test_bad_diversity_parameters(strength: float) -> None:
    with pytest.raises(FixtureError):
        diversify((), (), strength=strength, top_k=3)


def test_damaged_or_duplicate_scores_fail() -> None:
    docs = (FieldDocument("a", "word", "", ""),)
    for ranked in (
        (("unknown", 1.0),),
        (("a", math.nan),),
        (("a", 0.0),),
        (("a", 1.0), ("a", 2.0)),
    ):
        with pytest.raises(FixtureError):
            diversify(ranked, docs, strength=0.5, top_k=3)
    with pytest.raises(FixtureError):
        field_rank((docs[0], docs[0]), "word", weights=(1, 1, 1))


def test_ties_stable_and_generic_token_id_renaming() -> None:
    for prefix, word in (("random", "copper"), ("changed", "quartz"), ("else", "ＡＰＰＬＥ")):
        docs = (
            FieldDocument(prefix + "z", word, "", ""),
            FieldDocument(prefix + "a", word, "", ""),
        )
        scores = field_rank(docs, word, weights=(2, 2, 1))
        assert [i for i, _ in scores] == [prefix + "a", prefix + "z"]
        diverse = diversify(scores, docs, strength=0.5, top_k=2)
        assert [r.identifier for r in diverse] == [prefix + "a", prefix + "z"]


@pytest.mark.parametrize("config", CANDIDATES, ids=lambda c: c.name)
def test_modes_keep_all_authority_types_and_faq_is_not_authority(
    dataset: Dataset, config: object
) -> None:
    from knowledge_ticket_agent.retrieval_strategies import StrategyConfig

    assert isinstance(config, StrategyConfig)
    retriever = StrategyRetriever(dataset, config=config)
    for category in (Category.BILLING, Category.ACCESS, Category.EXPORT):
        for mode in ("authority", "faq_navigation"):
            result = retriever.search("policy request guidance", category=category, mode=mode)
            assert config.sha256 in result.method and config.version in result.method
            for hit in result.hits:
                chunk = retriever.get_chunk(hit.chunk_id, category=category)
                assert (chunk.kind == DocKind.FAQ) == (mode == "faq_navigation")
                verify_citation(
                    cite(retriever, hit.chunk_id, category=category),
                    dataset,
                    retriever,
                    category=category,
                    require_authority=mode == "authority",
                )
                if mode == "faq_navigation":
                    with pytest.raises(FixtureError, match="faq_not_authority"):
                        verify_citation(
                            cite(retriever, hit.chunk_id, category=category),
                            dataset,
                            retriever,
                            category=category,
                        )
    original = Retriever(dataset).selection(category=Category.BILLING)
    assert retriever.selection(category=Category.BILLING) == original
    assert "EX-BILL-TEAM@1.0" in original.exception_refs


@pytest.mark.parametrize("name", [c.name for c in CANDIDATES])
def test_conflicts_remain_explicit_and_exceptions_not_deleted(dataset: Dataset, name: str) -> None:
    retriever = StrategyRetriever(dataset, config=strategy(name))
    result = retriever.search("retention", category=Category.EXPORT, snapshot="conflict")
    assert result.status == "policy_conflict" and len(result.catalog.conflicts[0]) == 2
    assert {h.doc_ref for h in result.hits} & set(result.catalog.conflicts[0]) == set(
        result.catalog.conflicts[0]
    )
    assert not retriever.search("retention", category=Category.EXPORT).catalog.conflicts
    assert retriever.search("Team", category=Category.BILLING).catalog.exception_refs


@pytest.mark.parametrize("name", [c.name for c in CANDIDATES])
def test_inherited_reference_path_version_and_query_guards(dataset: Dataset, name: str) -> None:
    retriever = StrategyRetriever(dataset, config=strategy(name))
    for identifier in ("../outside", "/tmp/private", "file:///secret"):
        with pytest.raises(FixtureError):
            retriever.get_chunk(identifier)
    old = next(c for c in retriever.index.chunks if "@0.9" in c.doc_ref)
    with pytest.raises(FixtureError):
        retriever.get_chunk(old.chunk_id)
    valid = next(c for c in retriever.index.chunks if c.kind == DocKind.POLICY)
    with pytest.raises(FixtureError):
        verify_citation(
            replace(cite(retriever, valid.chunk_id), quote="forged"), dataset, retriever
        )
    for kwargs in ({"mode": "all"}, {"top_k": 0}, {"top_k": True}, {"method": "other"}):
        from typing import Any, cast

        with pytest.raises(FixtureError):
            retriever.search("word", **cast(Any, kwargs))
    with pytest.raises(FixtureError):
        retriever.search("x" * 257)


def test_v1_default_not_silently_replaced_and_configs_readonly(dataset: Dataset) -> None:
    assert Retriever(dataset).search("purchase").method == "bm25"
    assert Retriever(dataset).search("purchase", method="keyword").method == "keyword"
    retriever = StrategyRetriever(dataset, config=strategy("A2"))
    with pytest.raises(AttributeError):
        retriever.__setattr__("config", strategy("A1"))
    with pytest.raises(FixtureError):
        StrategyRetriever(dataset, config=replace(strategy("A2"), weights=(9, 9, 9)))
    with pytest.raises(FixtureError):
        strategy("not-registered")


def test_new_ranker_never_loads_gold(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args: object, **kwargs: object) -> None:
        raise AssertionError("no Gold in runtime")

    for module, function in (
        ("gold", "load_gold"),
        ("retrieval_gold", "load_retrieval_gold"),
        ("retrieval_dev", "load_dev"),
    ):
        monkeypatch.setattr(f"knowledge_ticket_agent.{module}.{function}", deny)
    for config in CANDIDATES:
        assert StrategyRetriever(dataset, config=config).search("purchase").hits


def test_symlink_dataset_still_rejected(copied: Path, tmp_path: Path) -> None:
    path = copied / "documents/POL-BILL-01-1.0.md"
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_text("private")
    path.symlink_to(outside)
    with pytest.raises(FixtureError):
        StrategyRetriever(load_dataset(copied), config=strategy("A1"))
