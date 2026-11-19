"""Prospective public M2 acceptance, separate from frozen M2F and formal quality.

Targets/configuration are trusted public declarations, never inferred from query
text. A1 order is unchanged; initial ranking and acquired coverage are separate.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.evidence import EvidenceUnit, score
from knowledge_ticket_agent.quality_evidence import AnswerSpec, resolve_policy
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category
from knowledge_ticket_agent.strict import require

DATASET_SHA256 = "f16e1ab3db2dd9327f48b720eaa5effacd14a4e1dde49c3426a4310fb46b4a0a"
VERSION = "lqd-policy-quality-3.0.0"


def evaluate(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == DATASET_SHA256, "quality_frozen_dataset")
    protocol = json.loads(raw)
    data = load_dataset()
    retriever = StrategyRetriever(data, config=strategy("A1"))
    ranked = []
    for row in protocol["ranking"]:
        found = retriever.search(row["query"], category=Category(row["category"]), top_k=3)
        units = tuple(
            EvidenceUnit(
                str(n),
                (
                    tuple(
                        c.chunk_id
                        for c in retriever.index.chunks
                        if c.doc_ref == ref and c.section_id == section
                    ),
                ),
            )
            for n, (ref, section) in enumerate(row["sections"])
        )
        metrics = score(units, tuple(h.chunk_id for h in found.hits))
        ranked.append(
            {
                "query": row["query"],
                "metrics": asdict(metrics),
                "initial_ids": [h.chunk_id for h in found.hits],
            }
        )
    rows = []
    for kind in ("documentary", "negative"):
        for row in protocol[kind]:
            result = resolve_policy(
                data,
                row["ticket"],
                row["query"],
                AnswerSpec(
                    Category(row["category"]),
                    row["intent"],
                    tuple(tuple(s) for s in row["sections"]),
                    row["snapshot"],
                ),
                retriever=retriever,
            )
            rows.append(
                {
                    "kind": kind,
                    "expected": row["expected"],
                    "observed": asdict(result),
                    "passed": result.decision.status == row["expected"],
                }
            )
    recall = sum(r["metrics"]["recall"] for r in ranked) / len(ranked)
    mrr = sum(r["metrics"]["reciprocal_rank"] for r in ranked) / len(ranked)
    read_only = all(
        row["observed"]["approval_calls"] == row["observed"]["commits"] == 0
        and row["observed"]["unchanged_state_verified"]
        for row in rows
    )
    return {
        "contract": VERSION,
        "dataset_version": protocol["version"],
        "corpus_index_sha256": retriever.index.sha256,
        "retriever_config_sha256": strategy("A1").sha256,
        "dataset_sha256": DATASET_SHA256,
        "scope": protocol["scope"],
        "historical_M2": "FAIL_UNCHANGED",
        "revised_formal_quality": "NOT_RUN",
        "top_k": 3,
        "ranking_denominator": len(ranked),
        "recall_at_3": recall,
        "mrr_at_3": mrr,
        "documentary_denominator": len(protocol["documentary"]),
        "negative_denominator": len(protocol["negative"]),
        "engineering_contract": "PASS"
        if recall >= 0.9 and mrr >= 0.8 and all(r["passed"] for r in rows) and read_only
        else "FAIL",
        "read_only_state_verified": read_only,
        "ranking": ranked,
        "cases": rows,
        "limitations": "Public/non-blind; explicit application-selected documentary intent, "
        "not semantic entailment, business facts, permission or initial top3 improvement",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Frozen public prospective M2 checks; no formal M2F"
    )
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    result = evaluate(args.cases)
    print(json.dumps(result, indent=2))
    if result["engineering_contract"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
