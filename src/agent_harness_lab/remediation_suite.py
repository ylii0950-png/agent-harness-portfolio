"""Explicit repair experiment entry; no calls on import or in default tests."""

import argparse
import json
from pathlib import Path

from agent_harness_lab.integration_suite import evaluate_integration


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned Phase 2.6R verification; no auto-retry")
    parser.add_argument("mode", choices=("mock", "real"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--max-requests", type=int, default=30)
    args = parser.parse_args()
    report = evaluate_integration(
        args.output,
        mode=args.mode,
        model=args.model,
        max_requests=args.max_requests,
        suite="phase26r",
    )
    print(
        json.dumps({"mode": args.mode, "counts": report["counts"], "requests": report["requests"]})
    )


if __name__ == "__main__":
    main()
