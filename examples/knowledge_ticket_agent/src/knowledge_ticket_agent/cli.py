"""Offline static data validation. No model, retrieval, tools or state execution."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from knowledge_ticket_agent import FixtureError, load_dataset, resource_root
from knowledge_ticket_agent.gold import load_gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate wholly fictional M1 data; no inference")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=None,
        help="Explicit trusted Gold directory outside public resources",
    )
    args = parser.parse_args()
    try:
        data = load_dataset(args.root)
        gold = (
            load_gold(args.gold_root, data, public_root=args.root or resource_root())
            if args.gold_root
            else None
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "counts": asdict(data.manifest.counts),
                    "corpus_version": data.manifest.corpus_version,
                    "manifest_sha256": data.manifest_sha256,
                    "business_gold": gold.gold_version if gold else "not_loaded",
                    "gold_cases": len(gold.cases) if gold else None,
                    "retrieval_gold": "not_created",
                    "model_requests": 0,
                },
                sort_keys=True,
            )
        )
    except (FixtureError, OSError) as exc:
        code = exc.code if isinstance(exc, FixtureError) else "resource_unavailable"
        print(json.dumps({"status": "invalid", "error_code": code}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
