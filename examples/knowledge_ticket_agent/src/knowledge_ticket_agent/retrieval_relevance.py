"""Prospective relevance ranking; candidate admission is not answer sufficiency."""

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass

from knowledge_ticket_agent.pair_model import PairScorer
from knowledge_ticket_agent.retrieval import Retriever, SearchResult
from knowledge_ticket_agent.retrieval_structural import (
    StructuralConfig,
    StructuralRetriever,
    _order,
    generate_candidates,
)
from knowledge_ticket_agent.retrieval_vnext import _cos, _lexical, _vectors, segments
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.semantic_model import VectorEncoder
from knowledge_ticket_agent.strict import require

VERSION = "lqd-relevance-retrieval-3.0.1"
PLAN_VERSION = "lqd-lossless-request-plan-1.0.0"
VIEW_VERSION = "lqd-assertion-search-view-1.0.0"


@dataclass(frozen=True)
class QueryPlan:
    query: str
    facets: tuple[str, ...]
    version: str = PLAN_VERSION


@dataclass(frozen=True)
class RetrievalView:
    """Trusted caller's low-trust ranking inputs, never proof observations."""

    query: str
    passages: tuple[tuple[str, str], ...]
    version: str = VIEW_VERSION


def plan_query(query: str) -> QueryPlan:
    segments(query)  # Original validation; do not use its character windows as clauses.
    query = unicodedata.normalize("NFKC", query).strip()
    if not any(c.isalnum() for c in query):
        return QueryPlan(query, ())
    # Split requests, not arbitrary conjunctions inside conditions or noun phrases.
    parts = re.split(
        r"[;；\n]|(?<=[?？])\s*|\b(?:and|also)\s+(?=(?:who|what|which|how|when|where)\b)|以及(?=必须|需要|哪些|什么|谁)|并且(?=需要|哪些|什么|谁)",
        query,
        flags=re.I,
    )
    facets = tuple(dict.fromkeys(p.strip() for p in parts if any(c.isalnum() for c in p)))
    require(len(facets) <= 16, "request_plan_capacity")
    # No first/last clause selection: every substantive character remains represented.
    return QueryPlan(query, facets)


@dataclass(frozen=True)
class RelevanceConfig:
    variant: str
    version: str = VERSION
    query_version: str = PLAN_VERSION
    shortlist_limit: int = 24
    support_version: str = "lqd-source-contrast-evidence-1.0.0"
    contrast_margin: float = 2.0

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RelevanceTrace:
    legal: tuple[str, ...]
    dense: tuple[str, ...]
    lexical: tuple[str, ...]
    intersection: tuple[str, ...]
    generated: tuple[str, ...]
    post_decision: tuple[str, ...]
    excluded: dict[str, str]
    facets: tuple[str, ...]
    selected_facets: tuple[tuple[str, str], ...]
    raw_scores: dict[str, float]
    decision: str = "relevance_only_not_answerability"


@dataclass(frozen=True)
class RelevanceAudit:
    result: SearchResult
    trace: RelevanceTrace


class RelevanceRetriever(Retriever):
    """Independent E5/lexical recall, whole-request anchor, uncovered-facet fill."""

    def __init__(
        self,
        data: Dataset,
        encoder: VectorEncoder,
        *,
        variant: str = "R1",
        pair: PairScorer | None = None,
    ) -> None:
        super().__init__(data)
        require(bool(re.fullmatch(r"[a-f0-9]{64}", encoder.identity)), "model_identity")
        require(len(self.index.chunks) <= 512, "semantic_index_capacity")
        self.encoder = encoder
        self.config = StructuralConfig()
        self._cache: dict[str, tuple[float, ...]] = {}
        require(variant in {"R0", "R1"}, "relevance_variant")
        require((pair is not None) == (variant == "R1"), "explicit_pair_backend")
        require(pair is None or bool(re.fullmatch(r"[a-f0-9]{64}", pair.identity)), "pair_identity")
        self.relevance_config = RelevanceConfig(variant)
        self.pair = pair
        self.identity_components = {
            "version": VERSION,
            "variant": variant,
            "config": self.relevance_config.identity,
            "generation": self.config.identity,
            "encoder": encoder.identity,
            "pair": pair.identity if pair else "dense",
            "query": PLAN_VERSION,
            "score_transport": "positive_monotonic_v1",
        }
        fingerprint = hashlib.sha256(
            json.dumps(self.identity_components, sort_keys=True).encode()
        ).hexdigest()
        self.method_identity = f"{VERSION}:{variant}:{fingerprint}"

    def _embed(
        self, texts: tuple[str, ...], *, query: bool = False
    ) -> tuple[tuple[float, ...], ...]:
        if query:
            return _vectors(self.encoder.embed(texts, query=True), len(texts))
        missing = tuple(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            self._cache.update(
                zip(missing, _vectors(self.encoder.embed(missing), len(missing)), strict=True)
            )
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

    _hit = staticmethod(StructuralRetriever._hit)

    def _scores(self, query: str, texts: tuple[str, ...]) -> tuple[float, ...]:
        if self.pair is not None:
            values = self.pair.score(tuple((query, text) for text in texts))
            require(
                len(values) == len(texts) and all(math.isfinite(v) for v in values), "pair_scores"
            )
            return values
        q = self._embed((query,), query=True)[0]
        return tuple(_cos(q, v) for v in self._embed(texts))

    def explain(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
        view: RetrievalView | None = None,
    ) -> RelevanceAudit:
        require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
        require(method == "bm25", "relevance_method")
        plan = plan_query(query)
        if view is not None:
            plan = plan_query(view.query)
        scope = self.selection(category=category, snapshot=snapshot)
        chunks = tuple(c for c in self.index.chunks if c.doc_ref in scope.authoritative_refs)
        by_id = {c.chunk_id: c for c in chunks}
        excluded = {
            c.chunk_id: "not_authoritative_in_scope"
            for c in self.index.chunks
            if c.doc_ref not in scope.authoritative_refs
        }
        method_identity = self.method_identity
        if view is not None:
            require(type(view) is RetrievalView and view.version == VIEW_VERSION, "retrieval_view")
            inputs = dict(view.passages)
            require(
                0 < len(inputs) == len(view.passages) <= 32 and inputs.keys() <= by_id.keys(),
                "retrieval_view_scope",
            )
            require(
                all(
                    type(t) is str and len(t) <= 16384 and "\x00" not in t for t in inputs.values()
                ),
                "retrieval_view_text",
            )
            excluded.update({i: "outside_policy_read_plan" for i in by_id if i not in inputs})
            by_id = {i: c for i, c in by_id.items() if i in inputs}
            chunks = tuple(by_id.values())
            method_identity += ":view:" + hashlib.sha256(view.version.encode()).hexdigest()
        dense_ids: tuple[str, ...] = ()
        lexical_ids: tuple[str, ...] = ()
        generated: tuple[str, ...] = ()
        shortlisted: tuple[str, ...] = ()
        chosen: list[str] = []
        selections: list[tuple[str, str]] = []
        scores: dict[str, float] = {}
        if chunks and plan.facets:
            text_by_id = (
                {c.chunk_id: f"{c.title}\n{c.heading}\n{c.text}" for c in chunks}
                if view is None
                else dict(view.passages)
            )
            texts = tuple(text_by_id[i] for i in by_id)
            vectors = self._embed(texts)
            questions = self._embed(segments(plan.query), query=True)
            dense = {
                i: max(_cos(q, v) for q in questions) for i, v in zip(by_id, vectors, strict=True)
            }
            lexical = _lexical(tuple(zip(by_id, texts, strict=True)), plan.query)
            channels = tuple(
                _order(
                    dict(
                        zip(
                            by_id,
                            self._scores(f, texts)
                            if self.pair is None
                            else tuple(_cos(self._embed((f,), query=True)[0], v) for v in vectors),
                            strict=True,
                        )
                    )
                )[:4]
                for f in plan.facets
            )
            dense_ids, lexical_ids, generated = generate_candidates(
                dense, lexical, channels, self.config
            )
            shortlisted = generated[: self.relevance_config.shortlist_limit]
            excluded.update({i: "candidate_budget" for i in by_id if i not in generated})
            excluded.update({i: "shortlist_budget" for i in generated if i not in shortlisted})
            passages = tuple(text_by_id[i] for i in shortlisted)
            scores = dict(zip(shortlisted, self._scores(plan.query, passages), strict=True))
            anchor = _order(scores)[0]
            chosen.append(anchor)
            # Keep the whole-request best first. Additional facets can share a source;
            # no mandatory distinct-section quota, topic projection or title stripping.
            for facet in plan.facets:
                local = self._scores(plan.query + "\nSpecific request: " + facet, passages)
                contextual = {i: min(scores[i], v) for i, v in zip(shortlisted, local, strict=True)}
                best = _order(contextual)[0]
                selections.append((facet, best))
                if best not in chosen and len(chosen) < min(top_k, 3):
                    chosen.append(best)
            for i in _order(scores):
                if len(chosen) >= min(top_k, 3):
                    break
                if i not in chosen:
                    chosen.append(i)
        hits = tuple(
            self._hit(by_id[i], 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, scores[i])))))
            for i in chosen
        )
        result = SearchResult(
            "policy_conflict" if scope.conflicts else "ok" if hits else "no_match",
            self.index.corpus_version,
            self.index.sha256,
            method_identity,
            hits,
            scope,
        )
        return RelevanceAudit(
            result,
            RelevanceTrace(
                tuple(by_id),
                dense_ids,
                lexical_ids,
                tuple(sorted(set(dense_ids) & set(lexical_ids))),
                generated,
                shortlisted,
                excluded,
                plan.facets,
                tuple(selections),
                scores,
            ),
        )
