"""Operator-facing schemas and non-executing validation, without claims or ranking."""

import json
import types
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from knowledge_ticket_agent.business_evaluation import validate_manifest
from knowledge_ticket_agent.evaluation_scope import digest
from knowledge_ticket_agent.formal_evaluation import GoldContract
from knowledge_ticket_agent.formal_scope import operator_scope
from knowledge_ticket_agent.quality_formal import QualityGold, QualityInputs
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.rules import validate_rule_contract
from knowledge_ticket_agent.schemas import BusinessDecision, Category, Dataset, Status
from knowledge_ticket_agent.strict import require

MODES = frozenset(
    {
        "normal",
        "conflict",
        "path_probe",
        "child_write",
        "missing_document",
        "false_child",
        "readback_order",
        "injection",
        "compaction",
    }
)
FAULTS = frozenset(
    {
        "none",
        "static_deny",
        "missing_evidence",
        "expired_only",
        "missing_facts",
        "child_failure",
        "invalid_provenance",
        "budget",
        "atomic_abort",
        "readback",
        "quota",
        "timeout",
        "provider",
        "infrastructure",
        "not_run",
        "stale_approval",
        "corrupt_evidence",
    }
)


def validate_business(data: Dataset, contract: GoldContract) -> None:
    """No answers are computed; all constraints precede freezing/scoring."""
    validate_rule_contract(data)
    require(bool(contract.version), "Gold_version")
    manifest = contract.manifest
    with operator_scope("STRUCTURAL_VALIDATION_ONLY", manifest.identity, digest(asdict(data))):
        validate_manifest(manifest, data)
    chunks = {c.chunk_id for c in Retriever(data).index.chunks}
    for case in manifest.cases:
        t, oracle = case.task, case.oracle
        require(
            t.fixture.removeprefix("sealed:").removeprefix("a1/") in MODES and t.fault in FAULTS,
            "formal_runtime_fixture",
        )
        require(
            t.actor_id in {s.actor_id for s in data.staff}
            and t.approval_fixture in {"approve", "refuse", "missing", "invalid", "throw"},
            "formal_actor_or_approval",
        )
        require(
            oracle.category in {c.value for c in Category}
            and oracle.decision in {d.value for d in BusinessDecision}
            and oracle.final_status in {s.value for s in Status}
            and oracle.final_revision in {0, 1}
            and oracle.allowed_stops == ("completed",)
            and oracle.outcome
            in {
                "business_success",
                "expected_denial",
                "permission_denial",
                "tool_error",
            },
            "formal_business_acceptance_semantics",
        )
        require(
            oracle.outcome != "tool_error" or t.fault == "corrupt_evidence",
            "formal_error_is_not_success",
        )
        for units in (case.retrieval_units, oracle.required_units):
            require(len({u.unit_id for u in units}) == len(units), "duplicate_evidence_unit")
            require(
                all(
                    u.unit_id
                    and u.alternatives
                    and all(b and set(b) <= chunks for b in u.alternatives)
                    for u in units
                ),
                "formal_evidence_mapping",
            )


def schema(model: type[object]) -> dict[str, Any]:
    """JSON Schema mirrors strict decoder field types; semantic checks stay explicit."""
    definitions: dict[str, Any] = {}

    def visit(kind: Any) -> dict[str, Any]:
        origin, args = get_origin(kind), get_args(kind)
        if origin in (Union, types.UnionType):
            return {"anyOf": [visit(k) for k in args]}
        if kind is type(None):
            return {"type": "null"}
        if kind in (str, int, bool):
            result = {"type": {str: "string", int: "integer", bool: "boolean"}[kind]}
            return result | ({"maxLength": 32768} if kind is str else {})
        if origin is tuple:
            require(len(args) == 2 and args[1] is Ellipsis, "schema_tuple")
            return {"type": "array", "items": visit(args[0]), "maxItems": 256}
        if isinstance(kind, type) and issubclass(kind, Enum):
            return {"type": "string", "enum": [e.value for e in kind]}
        require(isinstance(kind, type) and is_dataclass(kind), "schema_model")
        name = kind.__name__
        if name not in definitions:
            definitions[name] = {}
            hints = get_type_hints(kind)
            definitions[name] = {
                "type": "object",
                "additionalProperties": False,
                "required": [f.name for f in fields(kind)],
                "properties": {f.name: visit(hints[f.name]) for f in fields(kind)},
            }
        return {"$ref": "#/$defs/" + name}

    reference = visit(model)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **reference,
        "$defs": definitions,
    }


def main() -> None:
    import argparse
    from pathlib import Path

    from knowledge_ticket_agent.formal_evaluation import _read
    from knowledge_ticket_agent.quality_formal import validate
    from knowledge_ticket_agent.strict import decode, parse_json

    models = {
        "M5-data": Dataset,
        "M5-gold": GoldContract,
        "M2-data": QualityInputs,
        "M2-gold": QualityGold,
    }
    parser = argparse.ArgumentParser(
        description="Schema or structural validation only; never scores"
    )
    parser.add_argument("--schema", choices=tuple(models))
    parser.add_argument("--purpose", choices=("M2", "M5"))
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--gold", type=Path)
    args = parser.parse_args()
    if args.schema:
        require(not any((args.purpose, args.payload, args.gold)), "schema_no_inputs")
        print(json.dumps(schema(models[args.schema]), indent=2))
        return
    require(all((args.purpose, args.payload, args.gold)), "validation_inputs_required")
    if args.purpose == "M5":
        data = decode(Dataset, parse_json(_read(args.payload).decode()))
        contract = decode(GoldContract, parse_json(_read(args.gold).decode()))
        validate_business(data, contract)
        counts = [18, 20, 24]
    else:
        inputs = decode(QualityInputs, parse_json(_read(args.payload).decode()))
        gold = decode(QualityGold, parse_json(_read(args.gold).decode()))
        validate(inputs, gold)
        counts = [12, 8, 8]
    print(
        json.dumps(
            {
                "structural_validation": "PASS",
                "purpose": args.purpose,
                "denominators": counts,
                "scoring": "NOT_RUN",
                "claim_created": False,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError):
        print(
            json.dumps(
                {"structural_validation": "FAIL", "scoring": "NOT_RUN", "claim_created": False}
            )
        )
        raise SystemExit(1) from None
