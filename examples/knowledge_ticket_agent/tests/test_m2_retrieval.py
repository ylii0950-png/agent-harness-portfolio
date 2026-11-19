import math
from dataclasses import replace

import pytest
from knowledge_ticket_agent import FixtureError
from knowledge_ticket_agent.chunking import build_index
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.retrieval import Retriever, applicable_documents, lexical_rank, terms
from knowledge_ticket_agent.schemas import Category, Dataset


def test_bm25_hand_calculation_tf_df_length_and_query_deduplication() -> None:
    docs = (("a", "apple apple pear"), ("b", "pear"))
    expected = math.log(1 + 1.5 / 1.5) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    ranking = lexical_rank(docs, "apple apple")
    assert ranking[0][0] == "a" and ranking[0][1] == pytest.approx(expected)
    assert ranking == lexical_rank(docs, "apple")
    # pear df=2; length changes score while tf remains1.
    expected_b = math.log(1 + 0.5 / 2.5) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 1 / 2))
    assert dict(lexical_rank(docs, "pear"))["b"] == pytest.approx(expected_b)


def test_keyword_unique_term_overlap_and_tie_order() -> None:
    docs = (("z", "apple apple pear"), ("b", "apple pear"), ("a", "apple"))
    assert lexical_rank(docs, "apple apple pear", method="keyword") == (
        ("b", 2),
        ("z", 2),
        ("a", 1),
    )
    assert [i for i, _ in lexical_rank((("z", "same"), ("a", "same")), "same")] == ["a", "z"]


@pytest.mark.parametrize(
    "docs,query",
    [
        ((), "apple"),
        ((("a", ""),), "x"),
        ((("a", "apple"),), ""),
        ((("a", "apple"),), "ZZZ"),
        ((("a", "apple"),), "!!!"),
    ],
)
def test_empty_or_no_match_returns_no_rank(docs: tuple[tuple[str, str], ...], query: str) -> None:
    assert lexical_rank(docs, query) == ()


def test_normalization_is_explicit_english_lexical() -> None:
    assert terms("ＡＰＰＬＥ Straße 42 知识") == ("apple", "strasse", "42")


@pytest.mark.parametrize(
    "k1,b", [(0, 0.75), (-1, 0.75), (math.nan, 0.75), (1.2, -1), (1.2, 1.01), (1.2, math.inf)]
)
def test_bad_bm25_parameters_rejected(k1: float, b: float) -> None:
    with pytest.raises(FixtureError):
        lexical_rank((("a", "one"),), "one", k1=k1, b=b)


def test_duplicate_index_identifier_rejected() -> None:
    with pytest.raises(FixtureError, match="duplicate_chunk"):
        lexical_rank((("a", "one"), ("a", "two")), "one")


def test_version_dates_supersession_and_category_filter(dataset: Dataset) -> None:
    active = applicable_documents(
        dataset.manifest.documents, as_of=dataset.manifest.as_of, category=Category.BILLING
    )
    assert all(d.version != "0.9" and Category.BILLING in d.scope for d in active)
    base = dataset.manifest.documents[0]
    at_start = applicable_documents((base,), as_of=base.effective_from, category=None)
    assert at_start == (base,)
    closed = replace(base, effective_to="2031-04-15T12:00:00Z")
    assert applicable_documents((closed,), as_of=dataset.manifest.as_of, category=None) == ()
    assert applicable_documents((base,), as_of="2030-01-01T00:00:00Z", category=None) == ()
    older = replace(base, version="0.8", supersedes=())
    current = replace(base, supersedes=(older.ref,))
    assert applicable_documents((older, current), as_of=dataset.manifest.as_of, category=None) == (
        current,
    )


def test_normal_and_conflict_snapshots_report_both_without_score_choice(dataset: Dataset) -> None:
    retriever = Retriever(dataset)
    normal = retriever.search("retention", category=Category.EXPORT)
    conflict = retriever.search("nonsense", category=Category.EXPORT, snapshot="conflict")
    assert not normal.catalog.conflicts
    assert not any("CONFLICT" in r for r in normal.catalog.applicable_refs)
    assert conflict.status == "policy_conflict" and not conflict.hits
    assert conflict.catalog.conflicts == (("CONFLICT-RET-A@1.0", "CONFLICT-RET-B@1.0"),)
    assert "EX-BILL-TEAM@1.0" in retriever.selection(category=Category.BILLING).exception_refs
    assert ("BILL-EX1", "BILL-01") in retriever.selection(category=Category.BILLING).override_edges


@pytest.mark.parametrize("k", [0, 6, True, -1])
def test_top_k_bound_before_search(dataset: Dataset, k: int) -> None:
    with pytest.raises(FixtureError, match="top_k"):
        Retriever(dataset).search("purchase", top_k=k)


@pytest.mark.parametrize("query", ["x" * 257, "\x00"])
def test_query_bounds(dataset: Dataset, query: str) -> None:
    with pytest.raises(FixtureError, match="query"):
        Retriever(dataset).search(query)


@pytest.mark.parametrize(
    "identifier",
    ["../outside", "/tmp/private", "file:///x", "POL-BILL-01@9.9:eligibility", "x" * 257],
)
def test_get_is_id_only_not_a_file_path(dataset: Dataset, identifier: str) -> None:
    with pytest.raises(FixtureError):
        Retriever(dataset).get_chunk(identifier)


def test_old_or_conflict_chunk_not_available_in_normal_context(dataset: Dataset) -> None:
    retriever = Retriever(dataset)
    for chunk in retriever.index.chunks:
        if "@0.9" in chunk.doc_ref or chunk.kind.value == "conflict":
            with pytest.raises(FixtureError, match="inapplicable_chunk"):
                retriever.get_chunk(chunk.chunk_id)


def test_index_damage_rejected_and_readonly_public_index(dataset: Dataset) -> None:
    index = build_index(dataset)
    for bad in (
        replace(index, sha256="bad"),
        replace(index, chunks=index.chunks[:-1]),
        replace(index, chunker_version="old"),
    ):
        with pytest.raises(FixtureError, match="index_integrity"):
            Retriever(dataset, bad)
    with pytest.raises(AttributeError):
        Retriever(dataset).__setattr__("index", index)


def test_search_same_candidate_pool_both_methods_and_no_gold_dependency(dataset: Dataset) -> None:
    retriever = Retriever(dataset)
    left = retriever.search("first purchase", category=Category.BILLING)
    right = retriever.search("first purchase", category=Category.BILLING, method="keyword")
    assert left.catalog == right.catalog
    assert left == retriever.search("first purchase", category=Category.BILLING)
    assert all(
        h.score > 0 and len(h.excerpt) <= 600 and h.trust == "low_trust_source_data"
        for h in left.hits
    )
    changed = replace(
        dataset, tickets=tuple(replace(t, body=t.body + " changed") for t in dataset.tickets)
    )
    assert Retriever(changed).search("first purchase", category=Category.BILLING) == left


def test_valid_citation_and_faq_navigation_only(dataset: Dataset) -> None:
    retriever = Retriever(dataset)
    for chunk in retriever.index.chunks:
        if chunk.doc_ref not in retriever.selection().applicable_refs:
            continue
        citation = cite(retriever, chunk.chunk_id)
        verify_citation(citation, dataset, retriever, require_authority=False)
        if chunk.kind.value == "faq":
            with pytest.raises(FixtureError, match="faq_not_authority"):
                verify_citation(citation, dataset, retriever)
        else:
            verify_citation(citation, dataset, retriever)


@pytest.mark.parametrize(
    "field,value",
    [
        ("doc_ref", "POL-BILL-01@9.9"),
        ("section_id", "wrong"),
        ("start_char", -1),
        ("end_char", 999999),
        ("quote", "pretend success"),
        ("start_line", 0),
        ("index_sha256", "old"),
        ("chunk_id", "../private"),
        ("start_char", True),
    ],
)
def test_forged_citation_cannot_satisfy_source_check(
    dataset: Dataset, field: str, value: object
) -> None:
    from typing import Any, cast

    retriever = Retriever(dataset)
    chunk = next(c for c in retriever.index.chunks if c.kind.value == "policy")
    citation = replace(cite(retriever, chunk.chunk_id), **{field: cast(Any, value)})
    with pytest.raises(FixtureError):
        verify_citation(citation, dataset, retriever)


def test_citation_cannot_trust_alternative_modified_dataset(dataset: Dataset) -> None:
    retriever = Retriever(dataset)
    chunk = next(c for c in retriever.index.chunks if c.kind.value == "policy")
    bad = replace(
        dataset,
        documents=tuple(
            replace(p, text=p.text.replace(chunk.text, "forged")) for p in dataset.documents
        ),
    )
    with pytest.raises(FixtureError):
        verify_citation(replace(cite(retriever, chunk.chunk_id), quote="forged"), bad, retriever)
