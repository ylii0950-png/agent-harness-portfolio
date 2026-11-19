"""Trusted dev-only experiment; no new-strategy held-out evaluation entry point."""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_harness_lab import WorkspaceFiles
from knowledge_ticket_agent import load_dataset, resource_root
from knowledge_ticket_agent.chunking import build_index
from knowledge_ticket_agent.evidence import score
from knowledge_ticket_agent.loaders import inventory, read_text
from knowledge_ticket_agent.retrieval import Retriever, SearchResult
from knowledge_ticket_agent.retrieval_gold import RetrievalCase
from knowledge_ticket_agent.retrieval_strategies import CANDIDATES, StrategyRetriever
from knowledge_ticket_agent.schemas import Dataset, DocKind, Split
from knowledge_ticket_agent.strict import FixtureError, decode, parse_json, require

DEV_VERSION = "lqd-m2r-dev-1.0.0"
DEV_SHA256 = "8fa47e735e7f250521e458c1898c12dd28a3010f6d2fb6d31a1410a2a389b600"


@dataclass(frozen=True)
class DevGold:
    version: str
    source_gold_sha256: str
    manifest_sha256: str
    index_sha256: str
    metrics_version: str
    cases: tuple[RetrievalCase, ...]


@dataclass(frozen=True)
class DevFreeze:
    version: str
    sha256: str
    bytes: int
    cases: int
    split: str


def validate_dev(dev: DevGold, data: Dataset) -> None:
    # Exact released dev-only object binding also protects direct Python callers.
    payload = json.dumps(asdict(dev), indent=2) + "\n"
    require(hashlib.sha256(payload.encode()).hexdigest() == DEV_SHA256, "dev_release_binding")
    require(
        dev.version == DEV_VERSION
        and dev.manifest_sha256 == data.manifest_sha256
        and dev.index_sha256 == build_index(data).sha256,
        "dev_dataset_binding",
    )
    require(
        len(dev.cases) == 12
        and all(c.split == Split.DEV and c.positive for c in dev.cases)
        and {c.case_id for c in dev.cases} == {f"KT-D{i:02}" for i in range(1, 13)},
        "dev_only",
    )


def load_dev(root: Path, data: Dataset) -> DevGold:
    require(not root.resolve().is_relative_to(resource_root().resolve()), "gold_in_public_corpus")
    require(inventory(root) == {"dev_gold.json", "freeze.json"}, "dev_inventory")
    with WorkspaceFiles(root, max_bytes=256 * 1024) as workspace:
        freeze = decode(DevFreeze, parse_json(read_text(workspace, "freeze.json")))
        payload = read_text(workspace, "dev_gold.json")
    require(
        freeze.version == DEV_VERSION
        and freeze.sha256 == DEV_SHA256
        and freeze.cases == 12
        and freeze.split == "dev",
        "dev_freeze",
    )
    require(
        len(payload.encode()) == freeze.bytes
        and hashlib.sha256(payload.encode()).hexdigest() == DEV_SHA256,
        "dev_integrity",
    )
    dev = decode(DevGold, parse_json(payload))
    validate_dev(dev, data)
    return dev


Search = Callable[[RetrievalCase], SearchResult]


def summarize(outcomes: list[dict[str, object]]) -> dict[str, float]:
    # Named primitive metric fields; deliberately independent of rounded presentation.
    require(bool(outcomes), "empty_dev_group")
    totals = {"hit": 0.0, "recall": 0.0, "reciprocal_rank": 0.0, "raw_chunk_recall": 0.0}
    for item in outcomes:
        values = item["metrics"]
        require(isinstance(values, dict), "metric_shape")
        assert isinstance(values, dict)
        for key in totals:
            totals[key] += values[key]
    return {key: value / len(outcomes) for key, value in totals.items()}


def evaluate_dev(data: Dataset, dev: DevGold, search: Search) -> dict[str, object]:
    validate_dev(dev, data)
    chunks = {c.chunk_id: c for c in build_index(data).chunks}
    outcomes: list[dict[str, object]] = []
    for case in dev.cases:
        error = None
        hits: list[dict[str, object]] = []
        try:
            result = search(case)
            ranking = tuple(h.chunk_id for h in result.hits)
            hits = [
                {
                    "chunk_id": h.chunk_id,
                    "doc_ref": h.doc_ref,
                    "score": h.score,
                    "base_score": getattr(h, "base_score", h.score),
                    "score_kind": getattr(h, "score_kind", result.method),
                }
                for h in result.hits
            ]
            # Runtime can rank only valid scoped chunks, never fabricate metric IDs.
            valid = Retriever(data).selection(snapshot=case.snapshot, category=case.category)
            require(
                all(
                    identifier in chunks and chunks[identifier].doc_ref in valid.applicable_refs
                    for identifier in ranking
                ),
                "invalid_search_result",
            )
            status = result.status
        except (FixtureError, OSError) as exc:
            ranking = ()
            status = "retrieval_error"
            error = exc.code if isinstance(exc, FixtureError) else "unavailable"
        metrics = score(case.units, ranking)
        exception = any(
            chunks[i].kind == DocKind.EXCEPTION
            for u in case.required_units
            for b in u.alternatives
            for i in b
        )
        outcomes.append(
            {
                "case_id": case.case_id,
                "query_id": case.query_id,
                "query": case.query,
                "category": case.category.value,
                "snapshot": case.snapshot,
                "status": status,
                "error_code": error,
                "hits": hits,
                "exception_required": exception,
                "conflict_case": case.snapshot == "conflict",
                "metrics": asdict(metrics),
                "units": [asdict(u) for u in case.units],
            }
        )
    groups = {}
    group_specs: tuple[tuple[str, Callable[[dict[str, object]], bool]], ...] = (
        ("authority_all", lambda item: True),
        ("exception_required", lambda item: bool(item["exception_required"])),
        ("conflict", lambda item: bool(item["conflict_case"])),
        ("ordinary", lambda item: not item["exception_required"] and not item["conflict_case"]),
    )
    for name, predicate in group_specs:
        subset = [item for item in outcomes if predicate(item)]
        groups[name] = {
            "denominator": len(subset),
            "metrics": summarize(subset) if subset else None,
        }
    return {
        "denominator": 12,
        "aggregate": summarize(outcomes),
        "groups": groups,
        "outcomes": outcomes,
    }


def run_experiment(data: Dataset, dev: DevGold) -> dict[str, object]:
    validate_dev(dev, data)
    original = Retriever(data)
    controls: dict[str, dict[str, object]] = {}
    for method in ("bm25", "keyword"):

        def control_search(case: RetrievalCase, method_name: str = method) -> SearchResult:
            return original.search(
                case.query, category=case.category, snapshot=case.snapshot, method=method_name
            )

        controls[method] = evaluate_dev(data, dev, control_search)
    candidates: list[dict[str, object]] = []
    for config in CANDIDATES:
        retriever = StrategyRetriever(data, config=config)

        def candidate_search(
            case: RetrievalCase, selected: StrategyRetriever = retriever
        ) -> SearchResult:
            return selected.search(case.query, category=case.category, snapshot=case.snapshot)

        report = evaluate_dev(data, dev, candidate_search)
        navigation = []
        refs = {d.ref: d.source_refs for d in data.manifest.documents}
        for case in dev.cases:
            result = retriever.search(
                case.query, category=case.category, snapshot=case.snapshot, mode="faq_navigation"
            )
            navigation.append(
                {
                    "case_id": case.case_id,
                    "status": result.status,
                    "faq_hits": [
                        {
                            "chunk_id": h.chunk_id,
                            "doc_ref": h.doc_ref,
                            "source_refs": refs[h.doc_ref],
                        }
                        for h in result.hits
                    ],
                }
            )
        candidates.append(
            {
                "config": asdict(config),
                "config_sha256": config.sha256,
                "authority": report,
                "faq_navigation": {
                    "denominator": 12,
                    "with_hits": sum(bool(n["faq_hits"]) for n in navigation),
                    "quality": "not_scored_no_FAQ_gold",
                    "outcomes": navigation,
                },
            }
        )

    # Selection uses only macro dev ranking metrics; FAQ diagnostics do not enter it.
    def key(item: dict[str, object]) -> tuple[float, float, float]:
        authority = item["authority"]
        assert isinstance(authority, dict)
        metrics = authority["aggregate"]
        return metrics["recall"], metrics["reciprocal_rank"], metrics["hit"]

    best = max(candidates, key=key)  # Stable declaration-order tiebreak.
    baseline = controls["bm25"]["aggregate"]
    assert isinstance(baseline, dict)
    best_key = key(best)
    improved = best_key[0] > baseline["recall"] or (
        best_key[0] == baseline["recall"] and best_key[1] > baseline["reciprocal_rank"]
    )
    best_config = best["config"]
    assert isinstance(best_config, dict)
    return {
        "scope": "dev_selection_only",
        "dev_version": dev.version,
        "dev_sha256": DEV_SHA256,
        "manifest_sha256": dev.manifest_sha256,
        "index_sha256": dev.index_sha256,
        "controls": controls,
        "candidates": candidates,
        "selection": {
            "recommended": best_config["name"] if improved else None,
            "best_candidate": best_config["name"],
            "improved_over_v1_dev": improved,
            "rule": "recall_then_MRR_then_hit_then_declaration_order",
        },
        "held_out_v2": "not_created_not_run",
        "v1_sealed_new_strategy_runs": 0,
        "model_requests": 0,
        "api_token_usage": None,
        "actual_cost": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed12-query development experiment only")
    parser.add_argument("--dev-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New file only, never overwrite")
    args = parser.parse_args()
    try:
        data = load_dataset()
        result = run_experiment(data, load_dev(args.dev_root, data))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (FixtureError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": exc.code if isinstance(exc, FixtureError) else "unavailable",
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
