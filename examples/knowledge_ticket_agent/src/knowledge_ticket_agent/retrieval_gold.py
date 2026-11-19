"""Trusted, ranking-independent mechanical mapping; never imported by retrieval."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_harness_lab import WorkspaceFiles
from knowledge_ticket_agent.chunking import CHUNKER_VERSION, ChunkIndex, build_index
from knowledge_ticket_agent.evidence import (
    METRICS_VERSION,
    EvidenceUnit,
    map_anchor,
    minimum_cover,
    related,
)
from knowledge_ticket_agent.gold import BusinessGold
from knowledge_ticket_agent.loaders import inventory, read_text
from knowledge_ticket_agent.schemas import Category, Dataset, Split
from knowledge_ticket_agent.strict import decode, parse_json, require

GOLD_VERSION = "lqd-retrieval-gold-1.0.0"
CONFIG_VERSION = "lqd-lexical-1.0.0"


@dataclass(frozen=True)
class RetrievalConfig:
    version: str = CONFIG_VERSION
    k1: str = "1.2"
    b: str = "0.75"
    query_limit: int = 256
    top_k: int = 3
    normalization: str = "NFKC+casefold;[a-z0-9]+;unique-query"
    fields: str = "title+section_heading+body;once-each"
    baseline: str = "unique-term-overlap;positive-only;id-tie-break"
    category_source: str = "fixed-M1-expected-category;not-predicted"


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    split: Split
    query_id: str
    query: str
    category: Category
    snapshot: str
    positive: bool
    required_coverage: bool
    units: tuple[EvidenceUnit, ...]
    required_units: tuple[EvidenceUnit, ...]
    relevant_chunks: tuple[str, ...]
    minimum_chunks: int


@dataclass(frozen=True)
class RetrievalGold:
    version: str
    corpus_version: str
    ruleset_version: str
    manifest_sha256: str
    business_gold_semantic_sha256: str
    index_sha256: str
    chunker_version: str
    metrics_version: str
    config: RetrievalConfig
    cases: tuple[RetrievalCase, ...]


@dataclass(frozen=True)
class RetrievalFreeze:
    version: str
    sha256: str
    bytes: int
    positive_eval_queries: int
    required_eval_cases: int
    scoring_started: bool


def mapped_gold(data: Dataset, business: BusinessGold, index: ChunkIndex) -> RetrievalGold:
    require(index == build_index(data), "index_integrity")
    require(business.manifest_sha256 == data.manifest_sha256, "business_binding")
    docs = {d.metadata.ref: d for d in data.documents}
    cases = []
    for case in business.cases:
        units = tuple(map_anchor(a, index.chunks, docs[a.policy_ref].text) for a in case.anchors)
        require(len({u.unit_id for u in units}) == len(units), "duplicate_evidence_unit")
        retrieval = tuple(u for u in units if u.unit_id in case.retrieval_units)
        require(len(retrieval) == len(case.retrieval_units), "missing_unit")
        for unit in units:
            for identifier in related((unit,)):
                chunk = next(c for c in index.chunks if c.chunk_id == identifier)
                policy = docs[chunk.doc_ref].metadata
                require(policy.kind.value in {"policy", "exception", "conflict"}, "gold_authority")
                require(case.expected_category in policy.scope, "gold_category")
        cases.append(
            RetrievalCase(
                case.case_id,
                case.split,
                case.query_id,
                case.query,
                case.expected_category,
                case.snapshot,
                case.positive_retrieval,
                case.required_coverage,
                retrieval,
                units,
                tuple(sorted(related(retrieval))),
                minimum_cover(retrieval),
            )
        )
    # Semantic binding complements the separately preserved M1 release-file digest.
    business_payload = json.dumps(asdict(business), sort_keys=True, ensure_ascii=False)
    return RetrievalGold(
        GOLD_VERSION,
        data.manifest.corpus_version,
        data.manifest.ruleset_version,
        data.manifest_sha256,
        hashlib.sha256(business_payload.encode()).hexdigest(),
        index.sha256,
        CHUNKER_VERSION,
        METRICS_VERSION,
        RetrievalConfig(),
        tuple(cases),
    )


def structural_check(gold: RetrievalGold) -> dict[str, int]:
    positives = [c for c in gold.cases if c.split == Split.EVAL and c.positive]
    required = [c for c in gold.cases if c.split == Split.EVAL and c.required_coverage]
    require(len(positives) == 18 and len(required) == 20, "fixed_denominators")
    require(sum(len(c.units) for c in positives) == 25, "fixed_semantic_units")
    result = {c.case_id: minimum_cover(c.units) for c in positives}
    require(all(result[c.case_id] == c.minimum_chunks for c in positives), "minimum_cover")
    return result  # Caller must retain and report incompatible (>3) cases.


def write_freeze(root: Path, gold: RetrievalGold) -> None:
    require(not root.exists(), "freeze_exists")
    root.mkdir(parents=True)
    payload = json.dumps(asdict(gold), indent=2, ensure_ascii=False) + "\n"
    freeze = RetrievalFreeze(
        GOLD_VERSION,
        hashlib.sha256(payload.encode()).hexdigest(),
        len(payload.encode()),
        18,
        20,
        False,
    )
    (root / "retrieval_gold.json").write_text(payload, encoding="utf-8")
    (root / "freeze.json").write_text(json.dumps(asdict(freeze), indent=2) + "\n", encoding="utf-8")


def load_retrieval_gold(
    root: Path, data: Dataset, business: BusinessGold, index: ChunkIndex, *, public_root: Path
) -> RetrievalGold:
    require(not root.resolve().is_relative_to(public_root.resolve()), "gold_in_public_corpus")
    require(inventory(root) == {"retrieval_gold.json", "freeze.json"}, "gold_inventory")
    with WorkspaceFiles(root, max_bytes=256 * 1024) as workspace:
        freeze = decode(RetrievalFreeze, parse_json(read_text(workspace, "freeze.json")))
        payload = read_text(workspace, "retrieval_gold.json")
        require(
            len(payload.encode()) == freeze.bytes
            and hashlib.sha256(payload.encode()).hexdigest() == freeze.sha256,
            "retrieval_gold_integrity",
        )
        gold = decode(RetrievalGold, parse_json(payload))
    require(freeze.version == GOLD_VERSION and freeze.scoring_started is False, "freeze_state")
    require(freeze.positive_eval_queries == 18 and freeze.required_eval_cases == 20, "denominators")
    require(gold == mapped_gold(data, business, index), "retrieval_gold_binding")
    structural_check(gold)
    return gold
