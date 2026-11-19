"""Explicit M2E integrity CLI, never retrieval scoring or Gold output."""

import argparse
import json
from pathlib import Path

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.chunking import build_index
from knowledge_ticket_agent.evaluation_maintenance import parse_payload, validate_heldout
from knowledge_ticket_agent.strict import FixtureError, require


def main() -> None:
    parser = argparse.ArgumentParser(description="M2E local integrity, no formal scoring")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--public-check", action="store_true")
    mode.add_argument("--vault", type=Path)
    parser.add_argument("--entry", default="reviewed.aesgcm")
    args = parser.parse_args()
    try:
        data = load_dataset()
        if args.public_check:
            result: dict[str, object] = {
                "public_data": "valid",
                "index_sha256": build_index(data).sha256,
                "private_gold_loaded": False,
                "formal_scoring": "NOT_RUN",
            }
        else:
            from knowledge_ticket_agent.evaluation_custody import LocalVault

            vault = LocalVault(args.vault)
            heldout = parse_payload(vault.open(args.entry, actor="maintenance_cli"))
            result = validate_heldout(heldout, data)
            audit = vault.audit_status()
            require(audit["pending_seals"] == [], "vault_pending_seals")
            result["custody_audit"] = audit
        print(json.dumps(result, sort_keys=True))
    except ImportError:
        print(json.dumps({"status": "error", "error_code": "maintenance_dependency_unavailable"}))
        raise SystemExit(1) from None
    except (FixtureError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": exc.code
                    if isinstance(exc, FixtureError)
                    else "maintenance_unavailable",
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
