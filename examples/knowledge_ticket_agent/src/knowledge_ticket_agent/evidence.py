"""Pure evidence arithmetic: OR alternatives, AND bundles, bounded exact coverage."""

from dataclasses import dataclass

from knowledge_ticket_agent.chunking import Chunk
from knowledge_ticket_agent.gold import Anchor
from knowledge_ticket_agent.strict import require

METRICS_VERSION = "lqd-evidence-metrics-1.0.0"


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    alternatives: tuple[tuple[str, ...], ...]


def minimal_sets(options: set[frozenset[str]]) -> set[frozenset[str]]:
    require(len(options) <= 4096, "evidence_complexity")
    ordered = sorted(options, key=lambda s: (len(s), sorted(s)))
    kept: set[frozenset[str]] = set()
    for item in ordered:
        if not any(other <= item for other in kept):
            kept.add(item)
    return kept


def map_anchor(anchor: Anchor, chunks: tuple[Chunk, ...], source: str) -> EvidenceUnit:
    require(
        0 <= anchor.start_char < anchor.end_char <= len(source)
        and source[anchor.start_char : anchor.end_char] == anchor.quote,
        "evidence_anchor",
    )
    candidates = [
        c
        for c in chunks
        if c.doc_ref == anchor.policy_ref
        and c.section_id == anchor.section_id
        and c.start_char < anchor.end_char
        and c.end_char > anchor.start_char
    ]
    require(len(candidates) <= 128, "evidence_complexity")
    complete: set[frozenset[str]] = set()
    visits = 0

    def extend(position: int, selected: frozenset[str]) -> None:
        nonlocal visits
        visits += 1
        require(visits <= 4096, "evidence_complexity")
        while position < anchor.end_char and source[position].isspace():
            position += 1
        if position == anchor.end_char:
            complete.add(selected)
            return
        for chunk in candidates:
            if chunk.start_char <= position < chunk.end_char:
                extend(min(chunk.end_char, anchor.end_char), selected | {chunk.chunk_id})

    extend(anchor.start_char, frozenset())
    bundles = minimal_sets(complete)
    require(bool(bundles) and all(bundles), "unmapped_evidence")
    return EvidenceUnit(anchor.unit_id, tuple(sorted(tuple(sorted(b)) for b in bundles)))


def related(units: tuple[EvidenceUnit, ...]) -> frozenset[str]:
    return frozenset(c for u in units for b in u.alternatives for c in b)


def covered(unit: EvidenceUnit, retrieved: frozenset[str]) -> bool:
    require(bool(unit.alternatives) and all(unit.alternatives), "empty_evidence")
    return any(set(bundle) <= retrieved for bundle in unit.alternatives)


def minimum_cover(units: tuple[EvidenceUnit, ...]) -> int:
    require(len({u.unit_id for u in units}) == len(units), "duplicate_evidence_unit")
    states = {frozenset[str]()}
    for unit in units:
        require(bool(unit.alternatives) and all(unit.alternatives), "empty_evidence")
        states = minimal_sets({s | frozenset(b) for s in states for b in unit.alternatives})
    return min(map(len, states))


@dataclass(frozen=True)
class RetrievalMetrics:
    hit: float
    recall: float
    reciprocal_rank: float
    raw_chunk_recall: float
    first_relevant_rank: int | None
    covered_units: tuple[str, ...]
    missing_units: tuple[str, ...]
    top3: tuple[str, ...]


def score(units: tuple[EvidenceUnit, ...], ranking: tuple[str, ...]) -> RetrievalMetrics:
    require(bool(units), "no_positive_evidence")
    require(len({u.unit_id for u in units}) == len(units), "duplicate_evidence_unit")
    top = tuple(dict.fromkeys(ranking))[:3]
    retrieved = frozenset(top)
    relevant = related(units)
    require(bool(relevant), "no_positive_evidence")
    first = next((i for i, c in enumerate(top, 1) if c in relevant), None)
    matched = tuple(u.unit_id for u in units if covered(u, retrieved))
    missing = tuple(u.unit_id for u in units if not covered(u, retrieved))
    return RetrievalMetrics(
        float(first is not None),
        len(matched) / len(units),
        0.0 if first is None else 1 / first,
        len(retrieved & relevant) / len(relevant),
        first,
        matched,
        missing,
        top,
    )
