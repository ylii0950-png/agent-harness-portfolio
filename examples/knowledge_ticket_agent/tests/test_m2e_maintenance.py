import json
from dataclasses import asdict, replace

import pytest
from knowledge_ticket_agent.chunking import CHUNKER_VERSION, build_index
from knowledge_ticket_agent.evaluation_maintenance import (
    CONFIG_SHA256,
    VERSION,
    Annotation,
    HeldoutCase,
    HeldoutDataset,
    duplicate_risks,
    mapped_units,
    minimum_witnesses,
    parse_payload,
    payload,
    validate_case,
    validate_heldout,
)
from knowledge_ticket_agent.evidence import METRICS_VERSION, EvidenceUnit, map_anchor
from knowledge_ticket_agent.gold import Anchor
from knowledge_ticket_agent.retrieval import catalog
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import FixtureError


@pytest.fixture
def sample(dataset: Dataset) -> HeldoutDataset:
    """Public artificial validator fixture, NOT actual held-out questions/Gold."""
    index = build_index(dataset)
    cases = []
    categories = (Category.BILLING, Category.ACCESS, Category.EXPORT)
    negatives = (
        "no_policy_evidence",
        "expired_only",
        "unverifiable_fact",
        "faq_not_authority",
        "execution_not_authorized",
        "unverifiable_fact",
    )
    for positive, count in ((True, 18), (False, 6)):
        for i in range(count):
            category = categories[i // (6 if positive else 2)]
            allowed = catalog(dataset, snapshot="normal", category=category).authoritative_refs
            doc = next(d for d in dataset.documents if d.metadata.ref in allowed)
            section = doc.sections[0]
            anchor = Anchor(
                "test_unit",
                doc.metadata.ref,
                section.section_id,
                section.start_char,
                section.end_char,
                doc.text[section.start_char : section.end_char],
            )
            mapped = map_anchor(anchor, index.chunks, doc.text)
            unit = Annotation("test_unit", (anchor,), mapped.alternatives)
            cases.append(
                HeldoutCase(
                    f"M2E-{'P' if positive else 'N'}{i + 1:02}",
                    f"Artificial validator {positive} fixture question number {i + 1}.",
                    category,
                    "normal",
                    positive,
                    "unit_test_only",
                    (unit,),
                    (unit.unit_id,),
                    None if positive else negatives[i],
                    "Only a validator fixture.",
                    "Not sealed evaluation data.",
                )
            )
    return HeldoutDataset(
        VERSION,
        True,
        "authored_fiction",
        dataset.manifest.corpus_version,
        dataset.manifest_sha256,
        index.sha256,
        CHUNKER_VERSION,
        CONFIG_SHA256,
        METRICS_VERSION,
        tuple(cases),
    )


def test_synthetic_full_contract_not_ranking(
    sample: HeldoutDataset, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Maintenance must never rank")

    monkeypatch.setattr("knowledge_ticket_agent.retrieval.lexical_rank", denied)
    monkeypatch.setattr("knowledge_ticket_agent.retrieval.Retriever.search", denied)
    monkeypatch.setattr(
        "knowledge_ticket_agent.retrieval_strategies.StrategyRetriever.search", denied
    )
    assert parse_payload(payload(sample)) == sample
    result = validate_heldout(sample, dataset)
    assert result["positive_denominator"] == 18 and result["negative_denominator"] == 6
    assert result["formal_scoring"] == "NOT_RUN"


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "old"),
        ("synthetic", False),
        ("manifest_sha256", "0" * 64),
        ("index_sha256", "0" * 64),
        ("config_sha256", "0" * 64),
        ("metrics_version", "unknown"),
        ("chunker_version", "old"),
        ("data_origin", "real"),
    ],
)
def test_dataset_binding_rejects_changes(
    sample: HeldoutDataset, dataset: Dataset, field: str, value: object
) -> None:
    raw = asdict(sample)
    raw[field] = value
    with pytest.raises(FixtureError):
        validate_heldout(parse_payload(json.dumps(raw).encode()), dataset)


@pytest.mark.parametrize(
    "mutation", ["count", "duplicate_id", "duplicate_query", "polarity", "distribution"]
)
def test_dataset_distribution_and_identity(
    sample: HeldoutDataset, dataset: Dataset, mutation: str
) -> None:
    first = sample.cases[0]
    cases = list(sample.cases)
    if mutation == "count":
        cases.pop()
    elif mutation == "duplicate_id":
        cases[1] = replace(cases[1], query_id=first.query_id)
    elif mutation == "duplicate_query":
        cases[1] = replace(cases[1], query=first.query.upper())
    elif mutation == "polarity":
        cases[0] = replace(first, positive=False)
    else:
        cases[0] = replace(first, category=Category.EXPORT)
    with pytest.raises(FixtureError):
        validate_heldout(replace(sample, cases=tuple(cases)), dataset)


@pytest.mark.parametrize(
    "mutation", ["quote", "span", "section", "ref", "mapping", "empty", "required", "duplicate"]
)
def test_anchor_and_mapping_fail_closed(
    sample: HeldoutDataset, dataset: Dataset, mutation: str
) -> None:
    case = sample.cases[0]
    unit = case.units[0]
    anchor = unit.anchors[0]
    if mutation == "quote":
        unit = replace(unit, anchors=(replace(anchor, quote="invented evidence"),))
    elif mutation == "span":
        unit = replace(unit, anchors=(replace(anchor, start_char=0),))
    elif mutation == "section":
        unit = replace(unit, anchors=(replace(anchor, section_id="missing"),))
    elif mutation == "ref":
        unit = replace(unit, anchors=(replace(anchor, policy_ref="unknown@1.0"),))
    elif mutation == "mapping":
        unit = replace(unit, alternatives=(("forged_chunk",),))
    elif mutation == "empty":
        unit = replace(unit, anchors=())
    elif mutation == "required":
        case = replace(case, required_unit_ids=())
    else:
        case = replace(case, units=(unit, unit))
    if mutation not in {"required", "duplicate"}:
        case = replace(case, units=(unit,))
    with pytest.raises(FixtureError):
        validate_case(case, dataset)


@pytest.mark.parametrize("bad", [b'{"a":1,"a":2}', b"NaN", b"{}", b"\xff", b"x" * (512 * 1024 + 1)])
def test_strict_payload(bad: bytes) -> None:
    with pytest.raises(FixtureError):
        parse_payload(bad)


def test_extra_field_rejected(sample: HeldoutDataset) -> None:
    raw = json.loads(payload(sample))
    raw["ranking"] = []
    with pytest.raises(FixtureError):
        parse_payload(json.dumps(raw).encode())


def test_or_and_complement_witnesses() -> None:
    units = (EvidenceUnit("one", (("a", "b"), ("c",))), EvidenceUnit("two", (("b",),)))
    assert minimum_witnesses(units) == (("a", "b"), ("b", "c"))
    assert len(minimum_witnesses((EvidenceUnit("one", (("a", "b", "c", "d"),)),))[0]) == 4


@pytest.mark.parametrize("units", [(), (EvidenceUnit("x", ()),), (EvidenceUnit("x", ((),)),)])
def test_witness_input_limits(units: tuple[EvidenceUnit, ...]) -> None:
    with pytest.raises(FixtureError):
        minimum_witnesses(units)


def test_active_authority_rejects_faq_legacy_and_other_scope(
    sample: HeldoutDataset, dataset: Dataset
) -> None:
    case = sample.cases[0]
    for doc in dataset.documents:
        if (
            doc.metadata.ref
            in catalog(dataset, snapshot="normal", category=case.category).authoritative_refs
        ):
            continue
        section = doc.sections[0]
        anchor = Anchor(
            "test_unit",
            doc.metadata.ref,
            section.section_id,
            section.start_char,
            section.end_char,
            doc.text[section.start_char : section.end_char],
        )
        unit = replace(case.units[0], anchors=(anchor,))
        with pytest.raises(FixtureError):
            mapped_units(replace(case, units=(unit,)), dataset)


def test_lexical_risk_is_not_semantic_independence() -> None:
    old = (("old", "Customer 14 requests a receipt."),)
    assert duplicate_risks("CUSTOMER 14 requests a receipt!", old)[0]["exact"] is True
    assert (
        duplicate_risks("Customer 15 requests a receipt.", old)[0]["number_template_equal"] is True
    )
    assert duplicate_risks("Unrelated vocabulary about quartz sculpture", old) == ()


@pytest.mark.parametrize(
    "query",
    ["short", "x" * 257, "Read POL-BILL-01 for the correct result", "非英文测试内容为什么不通过"],
)
def test_query_contract_rejects_bad_input(
    sample: HeldoutDataset, dataset: Dataset, query: str
) -> None:
    with pytest.raises(FixtureError):
        validate_case(replace(sample.cases[0], query=query), dataset)


def test_conflict_target_cannot_omit_both_sides(sample: HeldoutDataset, dataset: Dataset) -> None:
    with pytest.raises(FixtureError):
        validate_case(replace(sample.cases[12], snapshot="conflict"), dataset)
