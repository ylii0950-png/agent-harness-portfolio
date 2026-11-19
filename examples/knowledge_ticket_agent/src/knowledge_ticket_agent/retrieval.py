"""Read-only lexical retrieval. No Gold, ticket IDs, model, tools or network."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType

from knowledge_ticket_agent.chunking import Chunk, ChunkIndex, build_index
from knowledge_ticket_agent.loaders import utc
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind, Document
from knowledge_ticket_agent.strict import require

CONFIG_VERSION = "lqd-lexical-1.0.0"


def terms(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()))


def applicable_documents(
    documents: tuple[Document, ...], *, as_of: str, category: Category | None
) -> tuple[Document, ...]:
    moment = utc(as_of)
    active = tuple(
        d
        for d in documents
        if utc(d.effective_from) <= moment
        and (d.effective_to is None or moment < utc(d.effective_to))
        and (category is None or category in d.scope)
    )
    superseded = {ref for d in active for ref in d.supersedes}
    return tuple(d for d in active if d.ref not in superseded and d.kind != DocKind.LEGACY)


@dataclass(frozen=True)
class Catalog:
    applicable_refs: tuple[str, ...]
    authoritative_refs: tuple[str, ...]
    exception_refs: tuple[str, ...]
    override_edges: tuple[tuple[str, str], ...]
    conflicts: tuple[tuple[str, ...], ...]


def catalog(data: Dataset, *, snapshot: str, category: Category | None) -> Catalog:
    selected = next((s for s in data.manifest.snapshots if s.name == snapshot), None)
    require(selected is not None, "unknown_snapshot")
    assert selected is not None
    docs = applicable_documents(
        tuple(d for d in data.manifest.documents if d.ref in selected.documents),
        as_of=data.manifest.as_of,
        category=category,
    )
    authority = tuple(d for d in docs if d.kind != DocKind.FAQ)
    families: dict[str, list[str]] = {}
    for d in authority:
        families.setdefault(d.policy_family, []).append(d.ref)
    # Conservative structural ambiguity, not natural-language contradiction inference.
    # Explicit supersession was removed above. Never choose a winner by score.
    conflicts = tuple(sorted(tuple(sorted(refs)) for refs in families.values() if len(refs) > 1))
    rule_ids = {r for d in authority for r in d.rule_ids}
    return Catalog(
        tuple(sorted(d.ref for d in docs)),
        tuple(sorted(d.ref for d in authority)),
        tuple(sorted(d.ref for d in authority if d.kind == DocKind.EXCEPTION)),
        tuple(
            sorted(
                (r.rule_id, other)
                for r in data.rules
                if r.rule_id in rule_ids
                for other in r.overrides
                if other in rule_ids
            )
        ),
        conflicts,
    )


def lexical_rank(
    documents: tuple[tuple[str, str], ...],
    query: str,
    *,
    method: str = "bm25",
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[tuple[str, float], ...]:
    require(type(query) is str and len(query) <= 256 and "\x00" not in query, "query")
    require(method in {"bm25", "keyword"}, "ranking_method")
    require(type(k1) in {int, float} and math.isfinite(k1) and k1 > 0, "bm25_k1")
    require(type(b) in {int, float} and math.isfinite(b) and 0 <= b <= 1, "bm25_b")
    require(len({identifier for identifier, _ in documents}) == len(documents), "duplicate_chunk")
    query_terms = sorted(set(terms(query)))
    counts = [(identifier, Counter(terms(text))) for identifier, text in documents]
    size = len(counts)
    average = sum(sum(c.values()) for _, c in counts) / size if size else 0
    if not query_terms or not size or not average:
        return ()
    df = {word: sum(word in c for _, c in counts) for word in query_terms}
    results = []
    for identifier, frequencies in counts:
        if method == "keyword":
            score = float(len(set(query_terms) & frequencies.keys()))
        else:
            length = sum(frequencies.values())
            score = 0.0
            for word in query_terms:
                tf = frequencies[word]
                if tf:
                    idf = math.log(1 + (size - df[word] + 0.5) / (df[word] + 0.5))
                    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        if score > 0:
            results.append((identifier, score))
    return tuple(sorted(results, key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_ref: str
    section_id: str
    source_path: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    score: float
    excerpt: str
    truncated: bool
    trust: str = "low_trust_source_data"


@dataclass(frozen=True)
class SearchResult:
    status: str
    corpus_version: str
    index_sha256: str
    method: str
    hits: tuple[SearchHit, ...]
    catalog: Catalog


class Retriever:
    """Immutable corpus index, frozen as_of; caller may narrow category/snapshot."""

    def __init__(self, data: Dataset, index: ChunkIndex | None = None) -> None:
        expected = build_index(data)
        require(index is None or index == expected, "index_integrity")
        self._data = data
        self._index = expected
        self._chunks = MappingProxyType({c.chunk_id: c for c in expected.chunks})

    @property
    def index(self) -> ChunkIndex:
        return self._index

    def selection(self, *, snapshot: str = "normal", category: Category | None = None) -> Catalog:
        require(category is None or isinstance(category, Category), "category")
        return catalog(self._data, snapshot=snapshot, category=category)

    def get_chunk(
        self, chunk_id: str, *, snapshot: str = "normal", category: Category | None = None
    ) -> Chunk:
        require(type(chunk_id) is str and len(chunk_id) <= 256, "chunk_id")
        require(chunk_id in self._chunks, "unknown_chunk")
        chunk = self._chunks[chunk_id]
        require(
            chunk.doc_ref in self.selection(snapshot=snapshot, category=category).applicable_refs,
            "inapplicable_chunk",
        )
        return chunk

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
    ) -> SearchResult:
        require(type(query) is str and len(query) <= 256 and "\x00" not in query, "query")
        require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
        selection = self.selection(snapshot=snapshot, category=category)
        candidates = tuple(c for c in self.index.chunks if c.doc_ref in selection.applicable_refs)
        ranking = lexical_rank(
            tuple((c.chunk_id, f"{c.title}\n{c.heading}\n{c.text}") for c in candidates),
            query,
            method=method,
        )
        hits = []
        for identifier, score in ranking[:top_k]:
            chunk = self._chunks[identifier]
            hits.append(
                SearchHit(
                    identifier,
                    chunk.doc_ref,
                    chunk.section_id,
                    chunk.source_path,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.start_line,
                    chunk.end_line,
                    score,
                    chunk.text[:600],
                    len(chunk.text) > 600,
                )
            )
        status = "policy_conflict" if selection.conflicts else ("ok" if hits else "no_match")
        return SearchResult(
            status, self.index.corpus_version, self.index.sha256, method, tuple(hits), selection
        )
