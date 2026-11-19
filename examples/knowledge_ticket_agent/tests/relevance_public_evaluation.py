"""Versioned PUBLIC-only scorer. Runtime receives sources and queries, never oracle."""

import argparse
import copy
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.evidence import EvidenceUnit, covered, score
from knowledge_ticket_agent.pair_model import BgePairScorer
from knowledge_ticket_agent.relevance_evidence import (
    select_sources,
    source_method_identity,
    verify_support,
)
from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever
from knowledge_ticket_agent.schemas import Category
from knowledge_ticket_agent.semantic_model import E5Encoder

from .relevance_corpus import build
from .vnext_public_evaluation import runtime_source_fingerprint

VERSION = "public-relevance-evidence-scorer-3.0.0"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "recall": statistics.mean(r["metrics"]["recall"] for r in rows),
        "mrr": statistics.mean(r["metrics"]["reciprocal_rank"] for r in rows),
        "complete": statistics.mean(not r["metrics"]["missing_units"] for r in rows),
        "candidate_recall": statistics.mean(r["candidate_recall"] for r in rows),
        "shortlist_recall": statistics.mean(r["shortlist_recall"] for r in rows),
        "search_false_rejections": sum(not r["hits"] for r in rows),
        "evidence_false_rejections": sum(r["evidence"]["status"] != "SUPPORTED" for r in rows),
    }


def evaluate(
    folder: Path | None,
    method: str,
    model: Path,
    pair_model: Path,
    *,
    split: str = "dev",
    inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assert method == "R1", "fixed retriever; no new candidate comparison"
    if inputs is None:
        assert folder is not None
        lock = json.loads((folder / (split + "-lock.json")).read_text())
        for name in ("corpus", "requests", "oracle"):
            p = folder / (split + "-" + name + ".json")
            assert hashlib.sha256(p.read_bytes()).hexdigest() == lock["files"][p.name]
        corpus, requests, oracle = (
            json.loads((folder / (split + "-" + name + ".json")).read_text())
            for name in ("corpus", "requests", "oracle")
        )
        input_identity = hashlib.sha256((folder / (split + "-lock.json")).read_bytes()).hexdigest()
    else:
        assert split == "dev", "generated inputs are development only"
        corpus, requests, oracle = inputs
        input_identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    assert corpus["version"].startswith("public-domain-evidence-"), "old exposed challenge disabled"
    expected = {c["id"]: c for c in oracle["cases"]}
    data, refs = build(load_dataset(), corpus, replace_sections=True)
    start = time.perf_counter()
    pair = BgePairScorer(pair_model)
    encoder = E5Encoder(model)
    r = RelevanceRetriever(data, encoder, pair=pair)
    startup = time.perf_counter() - start
    rows = []
    for case in requests["cases"]:
        start = time.perf_counter()
        selection = select_sources(
            data, r, case["query"], category=Category.OTHER, snapshot=case["snapshot"]
        )
        found = selection.result
        trace = asdict(selection.ranking.trace)
        citations = tuple(cite(r, h.chunk_id, category=Category.OTHER) for h in found.hits)
        verdict = verify_support(data, r, case["query"], citations, category=Category.OTHER)
        duration = time.perf_counter() - start
        assert found == select_sources(data, r, case["query"], category=Category.OTHER).result
        for citation in citations:
            verify_citation(citation, data, r, category=Category.OTHER)
        label = expected[case["id"]]  # scoring only, after inference completes

        def identifier(source: dict[str, Any]) -> str:
            return str(
                next(
                    c.chunk_id
                    for c in r.index.chunks
                    if c.doc_ref == refs[source["topic"]] and c.section_id == source["section"]
                )
            )

        units = tuple(
            EvidenceUnit(
                u["id"], tuple(tuple(identifier(s) for s in bundle) for bundle in u["alternatives"])
            )
            for u in label["units"]
        )
        metrics = asdict(score(units, tuple(h.chunk_id for h in found.hits))) if units else None
        raw_metrics = (
            asdict(score(units, tuple(h.chunk_id for h in selection.ranking.result.hits)))
            if units
            else None
        )
        rows.append(
            {
                "id": case["id"],
                "language": case["language"],
                "kind": case["kind"],
                "metrics": metrics,
                "view_metrics": raw_metrics,
                "view_candidate_recall": sum(
                    covered(u, frozenset(trace["generated"])) for u in units
                )
                / len(units)
                if units
                else None,
                "candidate_recall": sum(covered(u, frozenset(trace["generated"])) for u in units)
                / len(units)
                if units
                else None,
                "shortlist_recall": sum(
                    covered(u, frozenset(trace["post_decision"])) for u in units
                )
                / len(units)
                if units
                else None,
                "hits": [h.chunk_id for h in found.hits],
                "evidence": {
                    "status": verdict.status,
                    "missing": verdict.missing,
                    "unknown": verdict.unknown,
                    "bound_requirements": len(verdict.bindings),
                    "conflicts": len(verdict.conflicts),
                },
                "read_plan_recall": sum(
                    covered(u, frozenset(selection.read_dependencies)) for u in units
                )
                / len(units)
                if units
                else None,
                "read_plan_count": len(selection.read_dependencies),
                "candidate_count": len(trace["generated"]),
                "metadata_added": len(selection.metadata_added),
                "seconds": duration,
            }
        )
        print(json.dumps({"completed": case["id"], "method": method}), flush=True)
    evidence_rows = []
    eoracle = {c["id"]: c for c in oracle["evidence_cases"]}
    for case in requests["evidence_cases"]:
        modified = copy.deepcopy(corpus)
        for override in case.get("source_overrides", []):
            section = next(
                s
                for s in modified["topics"][override["topic"]]["sections"]
                if s["id"] == override["section"]
            )
            section["body"] = override["body"]
        evidence_data, evidence_refs = build(load_dataset(), modified, replace_sections=True)
        # No ranking needed: this test supplies canonical source observations only.
        evidence_r = RelevanceRetriever(evidence_data, encoder, pair=pair)
        citations = tuple(
            cite(
                evidence_r,
                next(
                    c.chunk_id
                    for c in evidence_r.index.chunks
                    if c.doc_ref == evidence_refs[s["topic"]] and c.section_id == s["section"]
                ),
                category=Category.OTHER,
            )
            for s in case["available"]
        )
        verdict = verify_support(
            evidence_data, evidence_r, case["query"], citations, category=Category.OTHER
        )
        label = eoracle[case["id"]]
        evidence_rows.append(
            {
                "id": case["id"],
                "expected": label["expected"],
                "actual": {
                    "status": verdict.status,
                    "missing": verdict.missing,
                    "unknown": verdict.unknown,
                    "conflicts": len(verdict.conflicts),
                },
                "passed": verdict.status == label["expected"]
                and not verdict.unknown
                and (
                    bool(verdict.conflicts)
                    if label["reason"] == "conflict"
                    else not verdict.conflicts
                ),
                "reason": label["reason"],
            }
        )
    positives = [r for r in rows if r["metrics"] is not None]
    negatives = [r for r in rows if r["metrics"] is None]
    durations = sorted(r["seconds"] for r in rows)
    return {
        "scope": "PUBLIC_ENGINEERING_NOT_FORMAL",
        "version": VERSION,
        "split": split,
        "profile": method,
        "source_sha256": runtime_source_fingerprint(),
        "method_identity": r.method_identity,
        "source_selection_method": source_method_identity(r),
        "candidate_basis": "source-owned eligibility; E5/lexical generation; BGE ranking",
        "pair_identity": pair.identity,
        "model_identity": encoder.identity if encoder else None,
        "data_lock_sha256": input_identity,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregate": aggregate(positives),
        "source_view_R1": {
            "recall": statistics.mean(row["view_metrics"]["recall"] for row in positives),
            "mrr": statistics.mean(row["view_metrics"]["reciprocal_rank"] for row in positives),
            "candidate_recall": statistics.mean(row["view_candidate_recall"] for row in positives),
        },
        "languages": {
            lang: aggregate([r for r in positives if r["language"] == lang])
            for lang in ("en", "zh", "mixed")
        },
        "requirements": {
            kind: aggregate([r for r in positives if r["kind"] == kind])
            for kind in ("single", "multi")
        },
        "negative": {
            "n": len(negatives),
            "raw_search_hits": sum(bool(r["hits"]) for r in negatives),
            "evidence_false_acceptance": sum(
                r["evidence"]["status"] == "SUPPORTED" for r in negatives
            ),
        },
        "evidence_cases": {
            "n": len(evidence_rows),
            "passed": sum(r["passed"] for r in evidence_rows),
            "rows": evidence_rows,
        },
        "resource": {
            "startup_s": startup,
            "query_and_verification_p95_s": durations[math.ceil(0.95 * len(durations)) - 1],
            "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "CPU_threads": 1,
            "CPU_user_s": resource.getrusage(resource.RUSAGE_SELF).ru_utime - cpu_before.ru_utime,
            "CPU_system_s": resource.getrusage(resource.RUSAGE_SELF).ru_stime - cpu_before.ru_stime,
            "CPU_scope": "full evaluator: startup, determinism checks, controls",
        },
        "rows": rows,
        "formal_runs": 0,
        "paid_API_calls": 0,
        "citation_integrity": True,
        "deterministic": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "challenge"))
    parser.add_argument("profile", choices=("R1",))
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--pair-model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--fixture-directory", type=Path)
    parser.add_argument("--development-v3", action="store_true")
    args = parser.parse_args()
    if args.split == "challenge":
        assert args.decision is not None
        decision = json.loads(args.decision.read_text())
        assert decision["scope"] == "FROZEN_DOMAIN_EVIDENCE_DEV_2.0.0"
        assert decision["source_sha256"] == runtime_source_fingerprint()
        assert decision["helper_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert not args.output.exists(), "result already exists; no overwrite"
    inputs = None
    if args.development_v3:
        assert args.split == "dev" and args.fixture_directory is None
        from .evidence_contract_helpers import development_inputs

        inputs = development_inputs()
    result = evaluate(
        args.fixture_directory,
        args.profile,
        args.model_directory,
        args.pair_model_directory,
        split=args.split,
        inputs=inputs,
    )
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("aggregate", "languages", "requirements", "negative", "resource")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
