"""Public experiment guards; expected sections are never supplied to ranking."""

from dataclasses import replace

import pytest
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.retrieval import lexical_rank, terms
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.retrieval_strategies import strategy
from knowledge_ticket_agent.schemas import Category

from .retrieval_generalization_support import (
    DATA_HASHES,
    STOP_WORDS,
    PrototypeRetriever,
    batches,
    config,
    controls,
    corpus,
    digest,
    matrix,
    rank,
)


def test_fixed_public_matrix_shape_identity_and_disjoint_topics() -> None:
    dev, challenge = matrix("dev"), matrix("challenge")
    assert DATA_HASHES["dev"] != DATA_HASHES["challenge"]
    assert len(dev.cases) == len(challenge.cases) == 44
    assert {c.topic for c in dev.cases}.isdisjoint(c.topic for c in challenge.cases)
    assert {c.variant for c in dev.cases} == {c.variant for c in challenge.cases}
    assert len({c.variant for c in dev.cases}) == 11
    assert len(dev.control_ids) == len(challenge.control_ids) == 8
    assert all(len(c.query) <= 4000 for c in (*dev.cases, *challenge.cases))


def test_all_terms_retained_once_without_query_position_bias() -> None:
    query = " ".join("word" + str(i) for i in range(200))
    groups = batches(query)
    assert len(groups) > 1 and all(len(g) <= 256 for g in groups)
    assert tuple(t for g in groups for t in terms(g)) == tuple(sorted(set(terms(query))))
    assert groups == batches(" ".join(reversed(query.split())))
    assert batches("Could you please tell me about the ＬＯＣＫＥＲ keys?") == ("keys locker",)
    assert batches("and the to") == ()
    assert sorted(STOP_WORDS) == config("G1")["stop_words"]


def test_batch_sum_is_existing_bm25_arithmetic() -> None:
    documents = (("a", "red red blue"), ("b", "blue green"))
    expected = dict(lexical_rank(documents, "red blue"))
    actual = dict(rank(documents, ("red", "blue")))
    assert actual == pytest.approx(expected)
    assert digest(config("G1")) != digest(config("G2"))


def test_synthetic_corpus_rebound_without_modifying_packaged_scaffolding() -> None:
    base = load_dataset()
    before = digest(base.manifest_sha256)
    data, target = corpus(base, matrix("dev").cases[0])
    r = PrototypeRetriever(data, "G1")
    assert before == digest(base.manifest_sha256)
    assert base.documents != data.documents
    assert base.manifest_sha256 != data.manifest_sha256
    assert r.get_chunk(
        next(
            c.chunk_id for c in r.index.chunks if c.doc_ref == target and c.section_id == "rg_rule"
        ),
        category=Category.OTHER,
    ).text
    # Metadata/bytes are independently checked by the original index constructor.
    with pytest.raises(FixtureError, match="source_integrity"):
        PrototypeRetriever(
            replace(
                data, documents=(replace(data.documents[0], text="unbound"), *data.documents[1:])
            ),
            "G1",
        )


@pytest.mark.parametrize("name", ["A1", "G1", "G2"])
def test_public_filter_source_and_non_consuming_controls(name: str) -> None:
    rows = controls(load_dataset(), name)
    assert len(rows) == 8 and all(row["passed"] for row in rows)


@pytest.mark.parametrize("name", ["G1", "G2"])
def test_prototype_explicit_errors_and_unchanged_a1(name: str) -> None:
    base = load_dataset()
    r = PrototypeRetriever(base, name)
    a1 = A1Retriever(base)
    before = a1.search("first purchase credit")
    for query in ("\x00", "x" * 4001, "x" * 257):
        with pytest.raises(FixtureError):
            r.search(query)
    for value in (True, 0, 6):
        with pytest.raises(FixtureError, match="top_k"):
            r.search("credit", top_k=value)
    with pytest.raises(FixtureError, match="ranking_method"):
        r.search("credit", method="keyword")
    with pytest.raises(FixtureError, match="unknown_snapshot"):
        r.search("credit", snapshot="private")
    result = r.search("credit", category=Category.BILLING)
    assert result == r.search("credit", category=Category.BILLING)
    assert result.hits and all(h.trust == "low_trust_source_data" for h in result.hits)
    assert a1.search("first purchase credit") == before
    assert (
        strategy("A1").sha256 == "5dad0b47b42f6a1bb42fefcf3f6f592f8732970543edbfdd7a7c6949b13e2354"
    )
