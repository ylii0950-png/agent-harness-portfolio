"""Independent candidates, selective support and query-facet evidence selection.

No Gold, task identifiers, business facts or write authority. All source access
and eligibility remain on the original manifest-bound index/catalog boundary.
"""

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass

from knowledge_ticket_agent.chunking import Chunk
from knowledge_ticket_agent.retrieval import Retriever, SearchHit, SearchResult, terms
from knowledge_ticket_agent.retrieval_vnext import _cos, _lexical, _vectors, segments
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.semantic_model import VectorEncoder
from knowledge_ticket_agent.strict import require

VERSION = "lqd-structural-retrieval-2.1.0"
QUERY_VERSION = "lqd-request-facets-2.1.0"
SUPPORT_VERSION = "lqd-centered-null-support-1.0.0"


@dataclass(frozen=True)
class StructuralConfig:
    version: str = VERSION
    query_version: str = QUERY_VERSION
    support_version: str = SUPPORT_VERSION
    channel_k: int = 16
    candidate_limit: int = 64
    facet_limit: int = 4
    rrf_k: int = 60
    null_samples: int = 63
    alpha: float = 0.05

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ScopeFields:
    by_id: dict[str, Chunk]
    titles: dict[str, str]
    title_vectors: dict[str, tuple[float, ...]]
    title_mean: tuple[float, ...]
    title_centered: dict[str, tuple[float, ...]]
    body_vectors: dict[str, tuple[float, ...]]
    heading_vectors: dict[str, tuple[float, ...]]


def generate_candidates(
    dense: dict[str, float],
    lexical: dict[str, float],
    facet_channels: tuple[tuple[str, ...], ...],
    config: StructuralConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Independent channel union. No similarity gate, Gold or source authorization."""
    dense_ids = _order(dense)[: config.channel_k]
    lexical_ids = _order(lexical)[: config.channel_k]
    fused = _rrf((dense_ids, lexical_ids, *facet_channels), config.rrf_k)
    return dense_ids, lexical_ids, _order(fused)[: config.candidate_limit]


@dataclass(frozen=True)
class StageTrace:
    legal: tuple[str, ...]
    dense: tuple[str, ...]
    lexical: tuple[str, ...]
    intersection: tuple[str, ...]
    generated: tuple[str, ...]
    post_decision: tuple[str, ...]
    excluded: dict[str, str]
    facets: tuple[str, ...]
    topic_p: float
    body_p: float
    rare_title_terms: tuple[str, ...]
    decision: str
    selected_facets: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SearchAudit:
    result: SearchResult
    trace: StageTrace


def request_facets(query: str) -> tuple[str, ...]:
    pieces = segments(query)  # Full validation and long-query preservation.
    if not pieces:
        return ()
    result = []
    for piece in pieces:
        for part in re.split(
            r"[;；]|\b(?:and|also|as well as)\b|以及|并且|同时|并|、", piece, flags=re.I
        ):
            part = part.strip(" ,，。.!?！？")
            if part and part not in result:
                result.append(part)
    # Long unrelated narration does not erase a late request. Preserve both ends;
    # original complete query segments are still independently searched below.
    return tuple(result if len(result) <= 4 else (result[0], *result[-3:]))


def _center(
    values: dict[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]]]:
    mean = tuple(sum(row[j] for row in values.values()) / len(values) for j in range(384))
    centered = {
        i: _unit(tuple(a - b for a, b in zip(row, mean, strict=True))) for i, row in values.items()
    }
    return mean, centered


def _unit(row: tuple[float, ...]) -> tuple[float, ...]:
    norm = sum(v * v for v in row) ** 0.5
    return tuple(v / norm for v in row) if norm > 1e-12 else (0.0,) * 384


def _residual(vector: tuple[float, ...], topic: tuple[float, ...]) -> tuple[float, ...]:
    projection = _cos(vector, topic)
    return _unit(tuple(a - projection * b for a, b in zip(vector, topic, strict=True)))


def _order(scores: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda i: (-scores[i], i)))


def _rrf(rankings: tuple[tuple[str, ...], ...], k: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, 1):
            values[identifier] = values.get(identifier, 0.0) + 1 / (k + rank)
    return values


def _null_p(
    query: tuple[float, ...], documents: dict[str, tuple[float, ...]], seed: str, samples: int
) -> float:
    """Query-specific contrast against sign-randomized directions; no labeled queries."""
    observed = max((_cos(query, v) for v in documents.values()), default=0.0)
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    exceeded = 0
    for _ in range(samples):
        null = tuple(v if rng.getrandbits(1) else -v for v in query)
        exceeded += max((_cos(null, v) for v in documents.values()), default=0.0) >= observed
    return (1 + exceeded) / (1 + samples)


class StructuralRetriever(Retriever):
    """Read-only S profile. A lexical source never needs to cross a dense floor."""

    def __init__(self, data: Dataset, encoder: VectorEncoder) -> None:
        super().__init__(data)
        require(bool(re.fullmatch(r"[a-f0-9]{64}", encoder.identity)), "model_identity")
        require(len(self.index.chunks) <= 512, "semantic_index_capacity")
        self.encoder = encoder
        self.config = StructuralConfig()
        self.method_identity = (
            f"{VERSION}:S:{self.config.identity}:{encoder.identity}:"
            f"{QUERY_VERSION}:{SUPPORT_VERSION}"
        )
        self._cache: dict[str, tuple[float, ...]] = {}

    def _embed(
        self, texts: tuple[str, ...], *, query: bool = False
    ) -> tuple[tuple[float, ...], ...]:
        if query:
            return _vectors(self.encoder.embed(texts, query=True), len(texts))
        missing = tuple(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            vectors = _vectors(self.encoder.embed(missing), len(missing))
            self._cache.update(zip(missing, vectors, strict=True))
        return tuple(self._cache[t] for t in texts)

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
    ) -> SearchResult:
        return self.explain(
            query, category=category, snapshot=snapshot, top_k=top_k, method=method
        ).result

    def _select_evidence(
        self,
        query: str,
        facets: tuple[str, ...],
        generated: tuple[str, ...],
        fields: ScopeFields,
        rare: tuple[str, ...],
        top_k: int,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, str]]]:
        """Reranking/coverage only: consumes candidates, never changes eligibility."""
        q = self._embed((query,), query=True)[0]
        title_words = {ref: set(terms(title)) for ref, title in fields.titles.items()}
        facet_vectors = self._embed(facets, query=True)
        chosen: list[tuple[str, float]] = []
        selections: list[tuple[str, str]] = []
        facet_orders = []
        for facet, vector in zip(facets, facet_vectors, strict=True):
            words = set(terms(facet))
            context_words = words if any(w in rare for w in words) else set(terms(query))
            # Generic trailing facets inherit whole-query topic, not an oracle source.
            context = vector if any(w in rare for w in words) else q
            centered_context = _unit(
                tuple(a - b for a, b in zip(context, fields.title_mean, strict=True))
            )
            doc_order = sorted(
                fields.titles,
                key=lambda ref: (
                    -len(context_words & title_words[ref])
                    if any(w in rare for w in context_words)
                    else 0,
                    -_cos(centered_context, fields.title_centered[ref]),
                    ref,
                ),
            )
            doc_rank = {ref: rank for rank, ref in enumerate(doc_order, 1)}
            scores: dict[str, float] = {}
            for ref in fields.titles:
                candidates = tuple(i for i in generated if fields.by_id[i].doc_ref == ref)
                field_text = re.sub(re.escape(fields.titles[ref]), " ", facet, flags=re.I).strip(
                    " ,，:："
                )
                field_text = field_text if any(c.isalnum() for c in field_text) else facet
                field_vector = self._embed((field_text,), query=True)[0]
                topic_vector = fields.title_vectors[ref]
                residual_query = _residual(field_vector, topic_vector)
                lexical_body = _lexical(
                    tuple(
                        (i, fields.by_id[i].heading + "\n" + fields.by_id[i].text)
                        for i in candidates
                    ),
                    field_text,
                )
                ranks = (
                    _order({i: _cos(field_vector, fields.heading_vectors[i]) for i in candidates}),
                    _order(
                        {
                            i: _cos(residual_query, _residual(fields.body_vectors[i], topic_vector))
                            for i in candidates
                        }
                    ),
                    _order(lexical_body),
                )
                scores.update(
                    {i: s / doc_rank[ref] for i, s in _rrf(ranks, self.config.rrf_k).items()}
                )
            facet_orders.append(scores)
        selected_sections: set[tuple[str, str]] = set()
        for facet, scores in zip(facets, facet_orders, strict=True):
            eligible = _order(scores)
            if eligible:
                identifier = eligible[0]
                section = fields.by_id[identifier].doc_ref, fields.by_id[identifier].section_id
                if section in selected_sections:
                    selections.append((facet, identifier))
                elif len(chosen) < min(top_k, 3):
                    chosen.append((identifier, scores[identifier]))
                    selections.append((facet, identifier))
                    selected_sections.add(section)
        fill = {i: max(s.get(i, 0.0) for s in facet_orders) for i in generated}
        for i in _order(fill):
            if len(chosen) >= min(top_k, 3):
                break
            section = fields.by_id[i].doc_ref, fields.by_id[i].section_id
            if section not in selected_sections:
                chosen.append((i, fill[i]))
                selected_sections.add(section)
        return chosen, selections

    def explain(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
    ) -> SearchAudit:
        require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
        require(method == "bm25", "structural_method")
        parts, facets = segments(query), request_facets(query)
        scope = self.selection(category=category, snapshot=snapshot)
        chunks = tuple(c for c in self.index.chunks if c.doc_ref in scope.authoritative_refs)
        excluded = {
            c.chunk_id: "not_authoritative_in_scope" for c in self.index.chunks if c not in chunks
        }
        dense_ids: tuple[str, ...] = ()
        lexical_ids: tuple[str, ...] = ()
        generated: tuple[str, ...] = ()
        chosen: list[tuple[str, float]] = []
        selections: list[tuple[str, str]] = []
        topic_p = body_p = 1.0
        rare: tuple[str, ...] = ()
        decision = "empty_query" if not facets else "no_eligible_source"
        if chunks and facets:
            by_id = {c.chunk_id: c for c in chunks}
            full_texts = tuple(f"{c.title}\n{c.heading}\n{c.text}" for c in chunks)
            full_vectors = dict(zip(by_id, self._embed(full_texts), strict=True))
            body_vectors = dict(
                zip(
                    by_id,
                    self._embed(tuple(c.heading + "\n" + c.text for c in chunks)),
                    strict=True,
                )
            )
            heading_vectors = dict(
                zip(by_id, self._embed(tuple(c.heading for c in chunks)), strict=True)
            )
            titles = {c.doc_ref: c.title for c in chunks}
            title_vectors = dict(zip(titles, self._embed(tuple(titles.values())), strict=True))
            title_mean, title_centered = _center(title_vectors)
            body_mean, body_centered = _center(body_vectors)
            fields = ScopeFields(
                by_id,
                titles,
                title_vectors,
                title_mean,
                title_centered,
                body_vectors,
                heading_vectors,
            )
            question_vectors = self._embed(parts, query=True)
            facet_vectors = self._embed(facets, query=True)
            dense = {i: max(_cos(q, v) for q in question_vectors) for i, v in full_vectors.items()}
            lexical = _lexical(tuple(zip(by_id, full_texts, strict=True)), query)
            channels = []
            for facet, q in zip(facets, facet_vectors, strict=True):
                channels.append(_order({i: _cos(q, v) for i, v in body_vectors.items()})[:4])
                channels.append(
                    _order(
                        _lexical(
                            tuple((c.chunk_id, c.heading + "\n" + c.text) for c in chunks), facet
                        )
                    )[:4]
                )
            dense_ids, lexical_ids, generated = generate_candidates(
                dense, lexical, tuple(channels), self.config
            )
            excluded.update({i: "candidate_budget" for i in by_id if i not in generated})
            q = self._embed((query,), query=True)[0]
            tq = _unit(tuple(a - b for a, b in zip(q, title_mean, strict=True)))
            bq = _unit(tuple(a - b for a, b in zip(q, body_mean, strict=True)))
            topic_p = _null_p(tq, title_centered, query + ":topic", self.config.null_samples)
            body_p = _null_p(bq, body_centered, query + ":body", self.config.null_samples)
            title_words = {ref: set(terms(title)) for ref, title in titles.items()}
            rare = tuple(
                sorted(
                    w
                    for w in set(terms(query))
                    if 0 < sum(w in words for words in title_words.values()) <= len(titles) / 2
                )
            )
            supported = body_p <= self.config.alpha and (topic_p <= self.config.alpha or bool(rare))
            decision = "supported" if supported else "no_query_specific_support"
            if supported:
                chosen, selections = self._select_evidence(
                    query, facets, generated, fields, rare, top_k
                )
            else:
                excluded.update({i: decision for i in generated})
        hits = tuple(self._hit(next(c for c in chunks if c.chunk_id == i), s) for i, s in chosen)
        result = SearchResult(
            "policy_conflict" if scope.conflicts else "ok" if hits else "no_match",
            self.index.corpus_version,
            self.index.sha256,
            self.method_identity,
            hits,
            scope,
        )
        trace = StageTrace(
            tuple(c.chunk_id for c in chunks),
            dense_ids,
            lexical_ids,
            tuple(sorted(set(dense_ids) & set(lexical_ids))),
            generated,
            generated if decision == "supported" else (),
            excluded,
            facets,
            topic_p,
            body_p,
            rare,
            decision,
            tuple(selections),
        )
        return SearchAudit(result, trace)

    @staticmethod
    def _hit(c: Chunk, score: float) -> SearchHit:
        return SearchHit(
            c.chunk_id,
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
