"""Offline public search demonstration; no hidden Gold or Agent execution."""

import argparse
import json
from dataclasses import asdict

from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.schemas import Category, DocKind


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic fictional knowledge search; no model"
    )
    parser.add_argument("query")
    parser.add_argument("--category", choices=[c.value for c in Category])
    parser.add_argument("--snapshot", choices=["normal", "conflict"], default="normal")
    parser.add_argument("--method", choices=["bm25", "keyword"], default="bm25")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    phase = "loading"
    try:
        data = load_dataset()
        phase = "retrieval"
        retriever = Retriever(data)
        category = Category(args.category) if args.category else None
        result = retriever.search(
            args.query,
            category=category,
            snapshot=args.snapshot,
            top_k=args.top_k,
            method=args.method,
        )
        verified = []
        for hit in result.hits:
            citation = cite(retriever, hit.chunk_id, snapshot=args.snapshot, category=category)
            chunk = retriever.get_chunk(hit.chunk_id, snapshot=args.snapshot, category=category)
            verify_citation(
                citation,
                data,
                retriever,
                snapshot=args.snapshot,
                category=category,
                require_authority=chunk.kind != DocKind.FAQ,
            )
            verified.append(
                {"chunk_id": hit.chunk_id, "valid": True, "authority": chunk.kind != DocKind.FAQ}
            )
        print(
            json.dumps(
                {
                    "search": asdict(result),
                    "citation_checks": verified,
                    "semantic_support": "not_established_by_citation_validity",
                    "model_requests": 0,
                },
                sort_keys=True,
            )
        )
    except (FixtureError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "load_error" if phase == "loading" else "retrieval_error",
                    "error_code": exc.code if isinstance(exc, FixtureError) else "unavailable",
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
