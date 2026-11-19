"""Trusted offline retrieval-only evaluation, separate from runtime search/Agents."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from knowledge_ticket_agent import FixtureError, load_dataset, resource_root
from knowledge_ticket_agent.chunking import build_index
from knowledge_ticket_agent.evidence import score
from knowledge_ticket_agent.gold import BusinessGold, load_gold
from knowledge_ticket_agent.retrieval import CONFIG_VERSION, Retriever
from knowledge_ticket_agent.retrieval_gold import (
    RetrievalGold,
    load_retrieval_gold,
    mapped_gold,
    structural_check,
)
from knowledge_ticket_agent.schemas import Dataset, Split
from knowledge_ticket_agent.strict import require


def evaluate(
    retriever: Retriever,
    gold: RetrievalGold,
    split: Split,
    method: str,
    *,
    data: Dataset,
    business: BusinessGold,
) -> dict[str, object]:
    require(gold == mapped_gold(data, business, retriever.index), "evaluation_gold_binding")
    require(method in {"bm25", "keyword"}, "ranking_method")
    require(
        gold.index_sha256 == retriever.index.sha256 and gold.config.version == CONFIG_VERSION,
        "evaluation_binding",
    )
    require(gold.config.k1 == "1.2" and gold.config.b == "0.75", "locked_configuration")
    structure = structural_check(gold)
    require(all(n <= 3 for n in structure.values()), "top3_incompatible")
    cases = [c for c in gold.cases if c.split == split and c.positive]
    require(len(cases) == (12 if split == Split.DEV else 18), "fixed_denominators")
    outcomes = []
    totals = {
        "hit_at_3": 0.0,
        "macro_recall_at_3": 0.0,
        "mrr_at_3": 0.0,
        "auxiliary_raw_chunk_recall_at_3": 0.0,
    }
    for case in cases:
        error = None
        try:
            result = retriever.search(
                case.query, category=case.category, snapshot=case.snapshot, method=method, top_k=3
            )
            ranking = tuple(hit.chunk_id for hit in result.hits)
            status = result.status
        except (FixtureError, OSError) as exc:
            ranking = ()
            status = "retrieval_error"
            error = exc.code if isinstance(exc, FixtureError) else "resource_unavailable"
        metrics = score(case.units, ranking)
        totals["hit_at_3"] += metrics.hit
        totals["macro_recall_at_3"] += metrics.recall
        totals["mrr_at_3"] += metrics.reciprocal_rank
        totals["auxiliary_raw_chunk_recall_at_3"] += metrics.raw_chunk_recall
        outcomes.append(
            {
                "case_id": case.case_id,
                "query_id": case.query_id,
                "query": case.query,
                "category": case.category.value,
                "snapshot": case.snapshot,
                "status": status,
                "error_code": error,
                "units": [asdict(u) for u in case.units],
                "relevant_chunks": case.relevant_chunks,
                "minimum_chunks": case.minimum_chunks,
                "metrics": asdict(metrics),
                "failure_reason": error
                or ("lexical_top3_missing_evidence" if metrics.missing_units else None),
            }
        )
    averages = {key: value / len(cases) for key, value in totals.items()}
    return {
        "scope": "retrieval_only_no_agent_or_business_execution",
        "split": split.value,
        "method": method,
        "config": asdict(gold.config),
        "gold_version": gold.version,
        "index_sha256": gold.index_sha256,
        "corpus_version": gold.corpus_version,
        "manifest_sha256": gold.manifest_sha256,
        "chunker_version": gold.chunker_version,
        "metrics_version": gold.metrics_version,
        "denominator": len(cases),
        "aggregate": averages,
        "quality_gate_met": averages["macro_recall_at_3"] >= 0.90 and averages["mrr_at_3"] >= 0.80,
        "structure": structure,
        "required_evidence": {
            "fixed_eval_denominator": 20,
            "mapping_ready": True,
            "runtime_coverage": None,
            "status": "not_run_M4_M5",
        },
        "excluded_from_positive_metrics": [
            c.case_id for c in gold.cases if c.split == split and not c.positive
        ],
        "business_evaluation": "not_run_M2",
        "model_requests": 0,
        "api_token_usage": None,
        "actual_cost": None,
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted offline retrieval-only fixed evaluation")
    parser.add_argument("--business-gold-root", required=True, type=Path)
    parser.add_argument("--gold-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["dev", "eval"])
    parser.add_argument(
        "--output", type=Path, help="New report only; existing reports never overwritten"
    )
    args = parser.parse_args()
    try:
        data = load_dataset()
        index = build_index(data)
        business = load_gold(args.business_gold_root, data, public_root=resource_root())
        gold = load_retrieval_gold(
            args.gold_root, data, business, index, public_root=resource_root()
        )
        retriever = Retriever(data, index)
        reports = {
            method: evaluate(
                retriever, gold, Split(args.split), method, data=data, business=business
            )
            for method in ("bm25", "keyword")
        }
        payload = json.dumps(reports, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as target:
                target.write(payload)
        else:
            print(payload, end="")
    except (FixtureError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": exc.code
                    if isinstance(exc, FixtureError)
                    else "report_or_resource_unavailable",
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
