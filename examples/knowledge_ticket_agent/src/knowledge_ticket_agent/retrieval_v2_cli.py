"""Explicit new-version public CLI, never replacing lqd-search v1 defaults."""

import argparse
import json
from dataclasses import asdict

from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.retrieval_strategies import CANDIDATES, StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category


def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in v2 retrieval; no Agent or model")
    parser.add_argument("query")
    parser.add_argument("--strategy", choices=[c.name for c in CANDIDATES], required=True)
    parser.add_argument("--mode", choices=["authority", "faq_navigation"], default="authority")
    parser.add_argument("--category", choices=[c.value for c in Category])
    parser.add_argument("--snapshot", choices=["normal", "conflict"], default="normal")
    args = parser.parse_args()
    try:
        data = load_dataset()
        retriever = StrategyRetriever(data, config=strategy(args.strategy))
        category = Category(args.category) if args.category else None
        result = retriever.search(
            args.query, mode=args.mode, category=category, snapshot=args.snapshot
        )
        for hit in result.hits:
            verify_citation(
                cite(retriever, hit.chunk_id, snapshot=args.snapshot, category=category),
                data,
                retriever,
                snapshot=args.snapshot,
                category=category,
                require_authority=args.mode == "authority",
            )
        print(
            json.dumps(
                {
                    "config": asdict(retriever.config),
                    "search": asdict(result),
                    "citation_validity": True,
                    "semantic_support": "not_inferred",
                    "model_requests": 0,
                },
                sort_keys=True,
            )
        )
    except (FixtureError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": exc.code
                    if isinstance(exc, FixtureError)
                    else "resource_unavailable",
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
