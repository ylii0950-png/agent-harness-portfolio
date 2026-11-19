"""Versioned prospective M5 profile, reusing unchanged A1 and citation wire types."""

import re
import unicodedata
from dataclasses import fields, replace

from knowledge_ticket_agent.retrieval import SearchHit, SearchResult
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import require

A1_SUITE = "kt-m5-restricted-24-3.0.0"
A1_EVALUATOR = "lqd-m5-formal-3.0.0"
QUERY_VERSION = "lqd-bounded-customer-query-1.0.0"
A1_METHOD = f"lqd-m5-initial-A1-3.0.0:{strategy('A1').sha256}:{QUERY_VERSION}:top3"


class A1Retriever(StrategyRetriever):
    """Existing business A1 adapter: cap3, preserve ordering, canonical tool fields."""

    def __init__(self, data: Dataset) -> None:
        super().__init__(data, config=strategy("A1"))

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
        require(type(query) is str and len(query) <= 4000 and "\x00" not in query, "query")
        original = query
        if len(query) > 256:
            tokens = list(
                dict.fromkeys(
                    re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", query).casefold())
                )
            )
            vocabulary = {
                t
                for c in self.index.chunks
                for t in re.findall(r"[a-z0-9]+", (c.title + " " + c.heading).casefold())
            }
            ordered = [t for t in tokens if t in vocabulary] + [
                t for t in tokens if t not in vocabulary
            ]
            kept: list[str] = []
            for token in ordered:
                if len(" ".join((*kept, token))) <= 256:
                    kept.append(token)
            query = " ".join(kept)
        result = super().search(
            query,
            category=category,
            snapshot=snapshot,
            top_k=min(top_k, 3),
            method=method,
            mode=mode,
        )
        return replace(
            result,
            method=(QUERY_VERSION + ":" + result.method) if original != query else result.method,
            hits=tuple(
                SearchHit(**{f.name: getattr(h, f.name) for f in fields(SearchHit)})
                for h in result.hits
            ),
        )
