"""Integrity/structure only; never invoke retrieval scoring on held-out inputs."""

import hashlib
import itertools
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from knowledge_ticket_agent.chunking import CHUNKER_VERSION, build_index
from knowledge_ticket_agent.evidence import METRICS_VERSION, EvidenceUnit, map_anchor, minimum_cover
from knowledge_ticket_agent.gold import Anchor
from knowledge_ticket_agent.retrieval import catalog
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import decode, parse_json, require

VERSION = "lqd-heldout-retrieval-2.0.0"
CONFIG_SHA256 = "5dad0b47b42f6a1bb42fefcf3f6f592f8732970543edbfdd7a7c6949b13e2354"
NEGATIVE_KINDS = frozenset(
    {
        "no_policy_evidence",
        "expired_only",
        "unverifiable_fact",
        "faq_not_authority",
        "execution_not_authorized",
    }
)
CATEGORIES = (Category.BILLING, Category.ACCESS, Category.EXPORT)


@dataclass(frozen=True)
class Annotation:
    unit_id: str
    anchors: tuple[Anchor, ...]
    alternatives: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class HeldoutCase:
    query_id: str
    query: str
    category: Category
    snapshot: str
    positive: bool
    stratum: str
    units: tuple[Annotation, ...]
    required_unit_ids: tuple[str, ...]
    negative_kind: str | None
    expected_behavior: str
    review_notes: str


@dataclass(frozen=True)
class HeldoutDataset:
    version: str
    synthetic: bool
    data_origin: str
    corpus_version: str
    manifest_sha256: str
    index_sha256: str
    chunker_version: str
    config_sha256: str
    metrics_version: str
    cases: tuple[HeldoutCase, ...]


def payload(data: HeldoutDataset) -> bytes:
    return (json.dumps(asdict(data), indent=2, sort_keys=True) + "\n").encode()


def parse_payload(raw: bytes) -> HeldoutDataset:
    require(len(raw) <= 512 * 1024, "heldout_size")
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        require(False, "heldout_encoding")
        raise AssertionError("unreachable") from None
    return decode(HeldoutDataset, parse_json(text))


def minimum_witnesses(units: tuple[EvidenceUnit, ...]) -> tuple[tuple[str, ...], ...]:
    """Small independent exhaustive witness enumeration, not lexical ranking."""
    require(0 < len(units) <= 8, "unit_count")
    require(all(u.alternatives and all(u.alternatives) for u in units), "empty_evidence")
    require(
        all(len(u.alternatives) <= 8 and all(len(b) <= 8 for b in u.alternatives) for u in units),
        "evidence_complexity",
    )
    count = 1
    for unit in units:
        count *= len(unit.alternatives)
    require(count <= 4096, "evidence_complexity")
    choices = {
        frozenset(itertools.chain.from_iterable(parts))
        for parts in itertools.product(*(u.alternatives for u in units))
    }
    minimum = min(map(len, choices))
    return tuple(sorted(tuple(sorted(c)) for c in choices if len(c) == minimum))


def mapped_units(case: HeldoutCase, data: Dataset) -> tuple[EvidenceUnit, ...]:
    index = build_index(data)
    docs = {d.metadata.ref: d for d in data.documents}
    active = set(catalog(data, snapshot=case.snapshot, category=case.category).authoritative_refs)
    require(len(case.units) <= 8, "unit_count")
    require(len({u.unit_id for u in case.units}) == len(case.units), "duplicate_unit")
    result = []
    for unit in case.units:
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", unit.unit_id)), "unit_id")
        require(0 < len(unit.anchors) <= 8, "anchor_count")
        mapped: set[tuple[str, ...]] = set()
        refs = set()
        for anchor in unit.anchors:
            require(
                anchor.unit_id == unit.unit_id and anchor.policy_ref in active, "anchor_authority"
            )
            document = docs[anchor.policy_ref]
            section = next(
                (s for s in document.sections if s.section_id == anchor.section_id), None
            )
            require(section is not None, "anchor_section")
            assert section is not None
            require(
                section.start_char <= anchor.start_char < anchor.end_char <= section.end_char,
                "anchor_section_bounds",
            )
            require(len(anchor.quote.strip()) >= 8, "anchor_too_short")
            refs.add(anchor.policy_ref)
            mapped.update(map_anchor(anchor, index.chunks, document.text).alternatives)
        # Equivalence across independent policies needs a future explicit reviewed
        # contract; never silently merge distinct obligations into one denominator.
        require(len(refs) == 1, "cross_policy_unit")
        alternatives = tuple(sorted(mapped))
        require(unit.alternatives == alternatives, "anchor_mapping")
        result.append(EvidenceUnit(unit.unit_id, alternatives))
    return tuple(result)


def validate_case(case: HeldoutCase, data: Dataset) -> dict[str, object]:
    require(case.category in CATEGORIES, "heldout_category")
    require(case.snapshot in {"normal", "conflict"}, "heldout_snapshot")
    require(
        12 <= len(case.query) <= 256 and case.query.isascii() and bool(case.query.strip()),
        "heldout_query",
    )
    require(
        not re.search(r"\b(?:POL|EX|FAQ|CONFLICT)-|\b(?:KT|M2E)-[DENP]\d", case.query),
        "query_identifier",
    )
    require(
        bool(case.stratum.strip())
        and bool(case.expected_behavior.strip())
        and bool(case.review_notes.strip()),
        "case_description",
    )
    mapped = mapped_units(case, data)
    require(len(set(case.required_unit_ids)) == len(case.required_unit_ids), "required_duplicates")
    require(set(case.required_unit_ids) == {u.unit_id for u in mapped}, "required_units")
    if case.positive:
        require(case.negative_kind is None and bool(mapped), "positive_shape")
        witnesses = minimum_witnesses(mapped)
        m_q = len(witnesses[0])
        require(m_q == minimum_cover(mapped) and m_q <= 3, "top3_unreachable")
        selection = catalog(data, snapshot=case.snapshot, category=case.category)
        if case.snapshot == "conflict":
            refs = {a.policy_ref for u in case.units for a in u.anchors}
            require(
                bool(selection.conflicts) and all(set(g) <= refs for g in selection.conflicts),
                "conflict_incomplete",
            )
        return {
            "query_id": case.query_id,
            "positive": True,
            "minimum_chunks": m_q,
            "minimum_witness_count": len(witnesses),
        }
    require(case.negative_kind in NEGATIVE_KINDS, "negative_kind")
    require(case.negative_kind == "no_policy_evidence" or bool(mapped), "negative_rationale_anchor")
    return {
        "query_id": case.query_id,
        "positive": False,
        "minimum_chunks": None,
        "negative_kind": case.negative_kind,
    }


def validate_heldout(heldout: HeldoutDataset, data: Dataset) -> dict[str, object]:
    index = build_index(data)
    require(
        heldout.version == VERSION
        and heldout.synthetic is True
        and heldout.data_origin == "authored_fiction",
        "heldout_provenance",
    )
    require(
        heldout.corpus_version == data.manifest.corpus_version
        and heldout.manifest_sha256 == data.manifest_sha256
        and heldout.index_sha256 == index.sha256
        and heldout.chunker_version == CHUNKER_VERSION
        and heldout.config_sha256 == CONFIG_SHA256
        and heldout.metrics_version == METRICS_VERSION,
        "heldout_binding",
    )
    require(len(heldout.cases) == 24, "heldout_count")
    require(
        {c.query_id for c in heldout.cases}
        == {f"M2E-P{i:02}" for i in range(1, 19)} | {f"M2E-N{i:02}" for i in range(1, 7)},
        "heldout_ids",
    )
    require(len({normalize_query(c.query) for c in heldout.cases}) == 24, "duplicate_query")
    for category in CATEGORIES:
        require(
            sum(c.positive and c.category == category for c in heldout.cases) == 6,
            "positive_distribution",
        )
        require(
            sum(not c.positive and c.category == category for c in heldout.cases) == 2,
            "negative_distribution",
        )
    for case in heldout.cases:
        require(case.query_id.startswith("M2E-P") == case.positive, "id_polarity")
    proof = [validate_case(c, data) for c in heldout.cases]
    return {
        "version": heldout.version,
        "gold_sha256": hashlib.sha256(payload(heldout)).hexdigest(),
        "positive_denominator": 18,
        "negative_denominator": 6,
        "proof": proof,
        "formal_scoring": "NOT_RUN",
        "independent_access_isolation": "NOT_ESTABLISHED",
    }


def normalize_query(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", query).casefold()))


def duplicate_risks(
    query: str, existing: tuple[tuple[str, str], ...]
) -> tuple[dict[str, object], ...]:
    """Lexical alerts only, not a semantic duplicate oracle or retrieval score."""
    normalized = normalize_query(query)
    tokens = set(normalized.split())
    result = []
    for identifier, old in existing:
        other = normalize_query(old)
        terms = set(other.split())
        jaccard = len(tokens & terms) / len(tokens | terms) if tokens | terms else 1.0
        sequence = SequenceMatcher(None, normalized, other, autojunk=False).ratio()
        number_template_equal = re.sub(r"\b\d+\b", "#", normalized) == re.sub(
            r"\b\d+\b", "#", other
        )
        if normalized == other or jaccard >= 0.65 or sequence >= 0.85 or number_template_equal:
            result.append(
                {
                    "old_id": identifier,
                    "exact": normalized == other,
                    "jaccard": jaccard,
                    "sequence": sequence,
                    "number_template_equal": number_template_equal,
                }
            )
    return tuple(result)
