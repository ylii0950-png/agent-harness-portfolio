"""Explicit v2 lexical strategies; no case, Gold, filesystem or model access."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from knowledge_ticket_agent.retrieval import (
    Retriever,
    SearchHit,
    SearchResult,
    lexical_rank,
    terms,
)
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind
from knowledge_ticket_agent.strict import require

STRATEGY_VERSION = "lqd-authority-fields-2.0.0"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    scoring: str
    weights: tuple[int, int, int]
    diversity_milli: int
    version: str = STRATEGY_VERSION
    k1: str = "1.2"
    b: str = "0.75"
    normalizer: str = "v1-NFKC-casefold-[a-z0-9]+-unique-query"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


CANDIDATES = (
    StrategyConfig("A1", "flat", (1, 1, 1), 0),
    StrategyConfig("A2", "field_sum", (2, 2, 1), 0),
    StrategyConfig("A3", "field_sum", (1, 1, 2), 0),
    StrategyConfig("A4", "field_sum", (2, 2, 1), 250),
    StrategyConfig("A5", "field_sum", (2, 2, 1), 500),
)


def strategy(name: str) -> StrategyConfig:
    config = next((c for c in CANDIDATES if c.name == name), None)
    require(config is not None, "unknown_strategy")
    assert config is not None
    return config


@dataclass(frozen=True)
class FieldDocument:
    identifier: str
    title: str
    heading: str
    body: str

    @property
    def fields(self) -> tuple[str, str, str]:
        return self.title, self.heading, self.body

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(terms("\n".join(self.fields)))


def field_rank(
    documents: tuple[FieldDocument, ...], query: str, *, weights: tuple[int, int, int]
) -> tuple[tuple[str, float], ...]:
    require(
        type(weights) is tuple
        and len(weights) == 3
        and all(type(w) is int and 0 < w <= 10 for w in weights),
        "field_weights",
    )
    require(len({d.identifier for d in documents}) == len(documents), "duplicate_chunk")
    scores: dict[str, float] = {}
    for field, weight in enumerate(weights):
        ranking = lexical_rank(tuple((d.identifier, d.fields[field]) for d in documents), query)
        for identifier, value in ranking:
            scores[identifier] = scores.get(identifier, 0.0) + weight * value
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True)
class RankedScore:
    identifier: str
    base_score: float
    score: float


def diversify(
    ranking: tuple[tuple[str, float], ...],
    documents: tuple[FieldDocument, ...],
    *,
    strength: float,
    top_k: int,
) -> tuple[RankedScore, ...]:
    require(
        type(strength) in {int, float} and math.isfinite(strength) and 0 <= strength <= 1,
        "diversity_strength",
    )
    require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
    by_id = {d.identifier: d for d in documents}
    require(len(by_id) == len(documents), "duplicate_chunk")
    require(len({identifier for identifier, _ in ranking}) == len(ranking), "duplicate_rank")
    require(
        all(
            identifier in by_id
            and type(value) in {int, float}
            and math.isfinite(value)
            and value > 0
            for identifier, value in ranking
        ),
        "invalid_rank",
    )
    if not ranking:
        return ()
    maximum = max(value for _, value in ranking)
    tokens = {identifier: by_id[identifier].tokens for identifier, _ in ranking}
    remaining = dict(ranking)
    chosen: list[RankedScore] = []
    while remaining and len(chosen) < top_k:
        options = []
        for identifier, base in remaining.items():
            redundancy = 0.0
            for previous in chosen:
                union = tokens[identifier] | tokens[previous.identifier]
                similarity = (
                    len(tokens[identifier] & tokens[previous.identifier]) / len(union)
                    if union
                    else 0.0
                )
                redundancy = max(redundancy, similarity)
            gain = base / maximum - strength * redundancy
            options.append(RankedScore(identifier, base, gain))
        best = min(options, key=lambda x: (-x.score, x.identifier))
        chosen.append(best)
        del remaining[best.identifier]
    return tuple(chosen)


@dataclass(frozen=True)
class StrategyHit(SearchHit):
    base_score: float = 0.0
    score_kind: str = "bm25"


class StrategyRetriever(Retriever):
    """New opt-in version/modes. Inherited v1 chunk/citation safety is unchanged."""

    def __init__(self, data: Dataset, *, config: StrategyConfig) -> None:
        require(config in CANDIDATES, "unregistered_strategy")
        super().__init__(data)
        self._config = config

    @property
    def config(self) -> StrategyConfig:
        return self._config

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
        mode: str = "authority",
    ) -> SearchResult:
        require(type(query) is str and len(query) <= 256 and "\x00" not in query, "query")
        require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
        require(method in {"bm25", "keyword"}, "ranking_method")
        require(mode in {"authority", "faq_navigation"}, "search_mode")
        selection = self.selection(snapshot=snapshot, category=category)
        chunks = tuple(
            c
            for c in self.index.chunks
            if c.doc_ref in selection.applicable_refs
            and ((c.kind == DocKind.FAQ) == (mode == "faq_navigation"))
        )
        documents = tuple(FieldDocument(c.chunk_id, c.title, c.heading, c.text) for c in chunks)
        if method == "keyword" or self.config.scoring == "flat":
            ranking = lexical_rank(
                tuple((d.identifier, "\n".join(d.fields)) for d in documents), query, method=method
            )
        else:
            ranking = field_rank(documents, query, weights=self.config.weights)
        use_diversity = bool(self.config.diversity_milli) and method == "bm25"
        if use_diversity:
            ordered = diversify(
                ranking, documents, strength=self.config.diversity_milli / 1000, top_k=top_k
            )
        else:
            ordered = tuple(RankedScore(i, value, value) for i, value in ranking[:top_k])
        by_id = {c.chunk_id: c for c in chunks}
        hits = []
        for item in ordered:
            chunk = by_id[item.identifier]
            hits.append(
                StrategyHit(
                    chunk.chunk_id,
                    chunk.doc_ref,
                    chunk.section_id,
                    chunk.source_path,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.start_line,
                    chunk.end_line,
                    item.score,
                    chunk.text[:600],
                    len(chunk.text) > 600,
                    base_score=item.base_score,
                    score_kind="mmr_gain" if use_diversity else method,
                )
            )
        status = "policy_conflict" if selection.conflicts else ("ok" if hits else "no_match")
        identity = f"{self.config.version}:{self.config.name}:{self.config.sha256}:{mode}:{method}"
        return SearchResult(
            status, self.index.corpus_version, self.index.sha256, identity, tuple(hits), selection
        )
