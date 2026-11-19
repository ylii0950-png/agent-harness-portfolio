"""Opt-in DEV strategies. Original A1 and all frozen paths remain unchanged."""

from knowledge_ticket_agent.retrieval import Retriever, SearchHit, SearchResult
from knowledge_ticket_agent.retrieval_strategies import FieldDocument, field_rank
from knowledge_ticket_agent.schemas import Category, DocKind
from knowledge_ticket_agent.strict import require

QUALITY_VERSION = "lqd-quality-dev-3.0.0"
QUALITY_CANDIDATES = ("Q1", "Q2")


class QualityRetriever(Retriever):
    def __init__(self, data: object, *, candidate: str) -> None:
        from knowledge_ticket_agent.schemas import Dataset

        require(type(data) is Dataset, "quality_dataset")
        assert isinstance(data, Dataset)
        require(candidate in QUALITY_CANDIDATES, "quality_candidate")
        super().__init__(data)
        self.candidate = candidate

    def ranking(
        self, query: str, *, category: Category | None = None, snapshot: str = "normal"
    ) -> tuple[tuple[str, float], ...]:
        selection = self.selection(category=category, snapshot=snapshot)
        chunks = tuple(c for c in self.index.chunks if c.doc_ref in selection.authoritative_refs)
        ranked = field_rank(
            tuple(FieldDocument(c.chunk_id, c.title, c.heading, c.text) for c in chunks),
            query,
            weights=(2, 2, 1),
        )
        if self.candidate == "Q1":
            return ranked
        refs = {c.chunk_id: c.doc_ref for c in chunks}
        seen: set[str] = set()
        first: list[tuple[str, float]] = []
        rest: list[tuple[str, float]] = []
        for row in ranked:
            ref = refs[row[0]]
            (rest if ref in seen else first).append(row)
            seen.add(ref)
        return tuple(first + rest)

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
    ) -> SearchResult:
        require(method == "bm25", "quality_bm25_only")
        require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
        selection = self.selection(category=category, snapshot=snapshot)
        chunks = {c.chunk_id: c for c in self.index.chunks}
        hits = []
        for identifier, value in self.ranking(query, category=category, snapshot=snapshot)[:top_k]:
            c = chunks[identifier]
            assert c.kind is not DocKind.FAQ
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
                    float(value),
                    c.text[:600],
                    len(c.text) > 600,
                )
            )
        return SearchResult(
            "policy_conflict" if selection.conflicts else "ok" if hits else "no_match",
            self.index.corpus_version,
            self.index.sha256,
            f"{QUALITY_VERSION}:{self.candidate}:title2-heading2-body1",
            tuple(hits),
            selection,
        )
