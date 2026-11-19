from dataclasses import replace

import pytest
from knowledge_ticket_agent import FixtureError
from knowledge_ticket_agent.chunking import build_index, chunk_document
from knowledge_ticket_agent.evidence import EvidenceUnit, minimum_cover, score
from knowledge_ticket_agent.loaders import markdown
from knowledge_ticket_agent.schemas import Dataset


def test_deterministic_spans_and_ids(dataset: Dataset) -> None:
    first = build_index(dataset)
    assert first == build_index(dataset)
    assert len(first.chunks) == 21  # Actual explicit sections, not planned 60–100 estimate.
    docs = {d.metadata.ref: d for d in dataset.documents}
    for c in first.chunks:
        source = docs[c.doc_ref].text
        assert source[c.start_char : c.end_char] == c.text
        assert len(c.text.split()) <= 200
        assert c.start_line == source.count("\n", 0, c.start_char) + 1
        assert c.end_line == source.count("\n", 0, c.end_char - 1) + 1
        assert c.source_sha256 in c.chunk_id


def test_long_section_word_boundaries_overlap_and_source_mutation(dataset: Dataset) -> None:
    original = dataset.documents[0]
    body = " ".join(f"word{i}" for i in range(510))
    source = (
        f"# {original.metadata.title}\nSynthetic: true; data_origin: authored_fiction\n\n"
        f"## eligibility | First purchase eligibility\n{body}\n\n"
        "## manual | High amount review\nUnchanged.\n"
    )
    policy = markdown(original.metadata, source)
    parts = chunk_document(policy)
    assert [len(c.text.split()) for c in parts] == [200, 200, 150, 1]
    assert parts[0].text.split()[-20:] == parts[1].text.split()[:20]
    assert all(source[c.start_char : c.end_char] == c.text for c in parts)
    changed = chunk_document(markdown(original.metadata, source.replace("word10 ", "other10 ")))
    assert {c.chunk_id for c in changed}.isdisjoint(c.chunk_id for c in parts)


def test_sentence_boundary_preference(dataset: Dataset) -> None:
    original = dataset.documents[0]
    body = " ".join(["one"] * 120) + ". " + " ".join(["two"] * 180)
    text = original.text.replace(original.sections[0].body, body)
    chunks = chunk_document(markdown(original.metadata, text))
    assert chunks[0].text.endswith(".") and len(chunks[0].text.split()) == 120
    assert all(len(c.text.split()) <= 200 for c in chunks)


def test_damaged_in_memory_sections_and_corpus_rejected(dataset: Dataset) -> None:
    bad = replace(dataset.documents[0], text=dataset.documents[0].text + "injected")
    with pytest.raises(FixtureError, match="section_integrity"):
        chunk_document(bad)
    with pytest.raises(FixtureError, match="source_integrity"):
        build_index(replace(dataset, documents=(bad, *dataset.documents[1:])))


def test_long_unbreakable_chunk_is_explicit_failure(dataset: Dataset) -> None:
    policy = dataset.documents[0]
    text = policy.text.replace(policy.sections[0].body, "x" * 8193)
    with pytest.raises(FixtureError, match="chunk_size"):
        chunk_document(markdown(policy.metadata, text))


def test_equivalent_alternatives_do_not_inflate_evidence_recall() -> None:
    unit = EvidenceUnit("base", tuple((str(i),) for i in range(5)))
    result = score((unit,), ("0", "1", "2"))
    assert (result.hit, result.recall, result.reciprocal_rank) == (1, 1, 1)
    assert result.raw_chunk_recall == 3 / 5
    assert minimum_cover((unit,)) == 1


def test_independent_policies_cannot_be_merged() -> None:
    units = (EvidenceUnit("base", (("a",), ("b",), ("c",))), EvidenceUnit("exception", (("d",),)))
    result = score(units, ("a", "b", "c"))
    assert result.hit == result.reciprocal_rank == 1
    assert result.recall == 0.5 and result.missing_units == ("exception",)
    assert minimum_cover(units) == 2


def test_complementary_bundles_and_exact_minimum_cover() -> None:
    units = (
        EvidenceUnit("a", (("a1", "a2"), ("z",))),
        EvidenceUnit("b", (("b1", "b2"), ("z", "b1"))),
    )
    assert minimum_cover(units) == 2  # z+b1; greedy first alternatives would need four.
    assert score(units, ("a1",)).recall == 0
    assert score(units, ("a1", "a2", "b1")).recall == 0.5
    impossible = (EvidenceUnit("a", (("a1", "a2"),)), EvidenceUnit("b", (("b1", "b2"),)))
    assert minimum_cover(impossible) == 4


@pytest.mark.parametrize("ranking,rr", [((), 0), (("unknown", "a"), 0.5), (("x", "y", "a"), 1 / 3)])
def test_manual_ranking_arithmetic(ranking: tuple[str, ...], rr: float) -> None:
    assert score((EvidenceUnit("a", (("a",),)),), ranking).reciprocal_rank == rr


def test_duplicates_do_not_fill_top3_or_increase_recall() -> None:
    result = score((EvidenceUnit("a", (("a", "b"),)),), ("a", "a", "a", "b"))
    assert result.top3 == ("a", "b") and result.recall == 1
    with pytest.raises(FixtureError):
        score((), ())
    with pytest.raises(FixtureError):
        minimum_cover((EvidenceUnit("x", ((),)),))


def test_mechanical_anchor_mapping_retains_all_equivalent_and_complementary_chunks(
    dataset: Dataset,
) -> None:
    from knowledge_ticket_agent.evidence import map_anchor
    from knowledge_ticket_agent.gold import Anchor

    policy = dataset.documents[0]
    body = " ".join(f"word{i}" for i in range(400))
    text = policy.text.replace(policy.sections[0].body, body)
    policy = markdown(policy.metadata, text)
    chunks = chunk_document(policy)
    start = text.index("word185 ")
    end = text.index("word190 ") - 1
    anchor = Anchor("overlap", policy.metadata.ref, "eligibility", start, end, text[start:end])
    unit = map_anchor(anchor, chunks, text)
    assert len(unit.alternatives) == 2 and all(len(b) == 1 for b in unit.alternatives)
    whole = policy.sections[0]
    anchor = replace(anchor, start_char=whole.start_char, end_char=whole.end_char, quote=whole.body)
    complete = map_anchor(anchor, chunks, text)
    assert minimum_cover((complete,)) == 3
    assert score((complete,), complete.alternatives[0][:-1]).recall == 0


def test_frozen_gold_and_fixed_denominators(dataset: Dataset) -> None:
    from knowledge_ticket_agent import resource_root
    from knowledge_ticket_agent.gold import load_gold
    from knowledge_ticket_agent.retrieval_gold import load_retrieval_gold, structural_check

    from .conftest import DEMO

    business = load_gold(DEMO / "tests/gold/v1", dataset, public_root=resource_root())
    index = build_index(dataset)
    gold = load_retrieval_gold(
        DEMO / "tests/retrieval_gold/v1", dataset, business, index, public_root=resource_root()
    )
    values = structural_check(gold)
    assert len(values) == 18 and list(values.values()).count(1) == 11
    assert list(values.values()).count(2) == 7 and max(values.values()) <= 3
    assert sum(c.required_coverage for c in gold.cases if c.split.value == "eval") == 20
    assert gold.config.k1 == "1.2" and gold.config.b == "0.75"
