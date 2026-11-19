"""Versioned local semantic candidate/reranking; no Gold, facts or write authority."""

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass

from knowledge_ticket_agent.retrieval import Retriever, SearchHit, SearchResult, lexical_rank, terms
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.semantic_model import VectorEncoder
from knowledge_ticket_agent.strict import require

QUERY_VERSION = "lqd-semantic-query-windows-1.0.1"
STRATEGY_VERSION = "lqd-semantic-hybrid-1.0.1"


@dataclass(frozen=True)
class VNextConfig:
    variant: str = "H"
    version: str = STRATEGY_VERSION
    query_version: str = QUERY_VERSION
    dimensions: int = 384
    candidate_limit: int = 16
    similarity_floor: float = 0.70
    lexical_weight: float = 0.10
    redundancy_weight: float = 0.08
    facet_weight: float = 0.03
    redundancy_floor: float = 0.90

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def segments(query: str) -> tuple[str, ...]:
    require(type(query) is str and len(query) <= 4000 and "\x00" not in query, "query")
    query = unicodedata.normalize("NFKC", query).strip()
    if not any(c.isalnum() for c in query):
        return ()
    fragments = re.split(r"(?<=[.!?。！？;；\n])\s*", query)
    parts = []
    for fragment in fragments:
        for offset in range(0, len(fragment), 224):
            part = fragment[offset : offset + 256].strip()
            if part and any(c.isalnum() for c in part) and part not in parts:
                parts.append(part)
    require(len(parts) <= 32, "semantic_query_segments")
    return tuple(parts)


def _vectors(values: tuple[tuple[float, ...], ...], size: int) -> tuple[tuple[float, ...], ...]:
    require(type(values) is tuple and len(values) == size, "embedding_batch")
    normalized = []
    for row in values:
        require(type(row) is tuple and len(row) == 384, "embedding_dimensions")
        require(all(type(v) in {int, float} and math.isfinite(v) for v in row), "embedding_finite")
        norm = math.sqrt(sum(v * v for v in row))
        require(norm > 0 and math.isfinite(norm), "embedding_norm")
        normalized.append(tuple(float(v / norm) for v in row))
    return tuple(normalized)


def _cos(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _lexical(documents: tuple[tuple[str, str], ...], query: str) -> dict[str, float]:
    words = sorted(set(terms(query)))
    require(all(len(w) <= 256 for w in words), "query_term")
    batches, current = [], ""
    for word in words:
        if len(current) + len(word) + bool(current) > 256:
            batches.append(current)
            current = ""
        current = (current + " " + word).strip()
    if current:
        batches.append(current)
    values: dict[str, float] = {}
    for batch in batches:
        for identifier, value in lexical_rank(documents, batch):
            values[identifier] = values.get(identifier, 0.0) + value
    return values


class VNextRetriever(Retriever):
    """Original source/index/catalog boundary; embeddings never authorize execution."""

    def __init__(self, data: Dataset, encoder: VectorEncoder, *, variant: str = "H") -> None:
        require(variant in {"D", "H"}, "vnext_variant")
        require(bool(re.fullmatch(r"[a-f0-9]{64}", encoder.identity)), "model_identity")
        super().__init__(data)
        require(len(self.index.chunks) <= 512, "semantic_index_capacity")
        self.config = VNextConfig(variant=variant)
        self.encoder = encoder
        self.method_identity = (
            f"{STRATEGY_VERSION}:{variant}:{self.config.identity}:"
            f"{encoder.identity}:{QUERY_VERSION}"
        )
        self._vectors: dict[str, tuple[float, ...]] = {}

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
    ) -> SearchResult:
        require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
        require(method == "bm25", "vnext_method")
        parts = segments(query)
        scope = self.selection(category=category, snapshot=snapshot)
        chunks = tuple(c for c in self.index.chunks if c.doc_ref in scope.authoritative_refs)
        ranking: list[tuple[str, float]] = []
        if chunks and parts:
            missing = tuple(c for c in chunks if c.chunk_id not in self._vectors)
            if missing:
                texts = tuple(f"{c.title}\n{c.heading}\n{c.text}" for c in missing)
                values = _vectors(self.encoder.embed(texts), len(texts))
                self._vectors.update((c.chunk_id, v) for c, v in zip(missing, values, strict=True))
            questions = _vectors(self.encoder.embed(parts, query=True), len(parts))
            similarities = {
                c.chunk_id: tuple(_cos(q, self._vectors[c.chunk_id]) for q in questions)
                for c in chunks
            }
            dense = {identifier: max(values) for identifier, values in similarities.items()}
            lexical = _lexical(
                tuple((c.chunk_id, f"{c.title}\n{c.heading}\n{c.text}") for c in chunks), query
            )
            allowed = {
                identifier
                for identifier, value in dense.items()
                if value >= self.config.similarity_floor
            }
            if self.config.variant == "D":
                ranking = sorted(
                    ((i, dense[i]) for i in allowed), key=lambda item: (-item[1], item[0])
                )[: min(top_k, 3)]
            else:
                dense_ids = sorted(allowed, key=lambda i: (-dense[i], i))[:16]
                lexical_ids = sorted(lexical, key=lambda i: (-lexical[i], i))[:16]
                candidates = (set(dense_ids) | set(lexical_ids)) & allowed
                maximum = max(lexical.values(), default=1.0)
                coverage = [0.0] * len(questions)
                while candidates and len(ranking) < min(top_k, 3):
                    options = []
                    for identifier in candidates:
                        redundancy = max(
                            (
                                _cos(self._vectors[identifier], self._vectors[previous])
                                for previous, _ in ranking
                            ),
                            default=0.0,
                        )
                        novelty = sum(
                            max(0.0, value - before)
                            for value, before in zip(
                                similarities[identifier], coverage, strict=True
                            )
                        ) / len(questions)
                        gain = (
                            dense[identifier]
                            + self.config.lexical_weight * lexical.get(identifier, 0.0) / maximum
                            - self.config.redundancy_weight
                            * max(0.0, redundancy - self.config.redundancy_floor)
                            + self.config.facet_weight * novelty
                        )
                        options.append((identifier, gain))
                    chosen = min(options, key=lambda item: (-item[1], item[0]))
                    ranking.append(chosen)
                    candidates.remove(chosen[0])
                    coverage = [
                        max(before, value)
                        for before, value in zip(coverage, similarities[chosen[0]], strict=True)
                    ]
        by_id = {c.chunk_id: c for c in chunks}
        hits = []
        for identifier, score in ranking:
            c = by_id[identifier]
            hits.append(
                SearchHit(
                    identifier,
                    c.doc_ref,
                    c.section_id,
                    c.source_path,
                    c.start_char,
                    c.end_char,
                    c.start_line,
                    c.end_line,
                    score,
                    c.text[:600],
                    len(c.text) > 600,
                )
            )
        return SearchResult(
            "policy_conflict" if scope.conflicts else "ok" if hits else "no_match",
            self.index.corpus_version,
            self.index.sha256,
            self.method_identity,
            tuple(hits),
            scope,
        )
