"""M2F fixed offline scoring. Caller owns custody and one-run enforcement."""

import math
from collections.abc import Callable
from dataclasses import asdict
from fractions import Fraction
from typing import Any, cast

from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.evaluation_maintenance import (
    HeldoutCase,
    HeldoutDataset,
    mapped_units,
    parse_payload,
    validate_heldout,
)
from knowledge_ticket_agent.evidence import covered, related, score
from knowledge_ticket_agent.formal_ledger import canonical
from knowledge_ticket_agent.retrieval import Retriever, SearchHit, SearchResult, catalog
from knowledge_ticket_agent.retrieval_strategies import StrategyHit, StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Dataset, DocKind
from knowledge_ticket_agent.strict import require

PROTOCOL = "lqd-formal-offline-1.0.0"
METHODS = ("A1", "bm25", "keyword")
METRICS = (
    "hit_at_3",
    "evidence_recall_at_3",
    "mrr_at_3",
    "chunk_recall_at_3",
    "required_evidence_coverage",
    "full_evidence_case_rate",
)


def number(value: Fraction) -> dict[str, object]:
    return {"exact": str(value), "value": float(value)}


def retrieve(data: Dataset, case: HeldoutCase, method: str) -> SearchResult:
    require(method in METHODS, "formal_method")
    if method == "A1":
        return StrategyRetriever(data, config=strategy("A1")).search(
            case.query, category=case.category, snapshot=case.snapshot, top_k=3
        )
    return Retriever(data).search(
        case.query, category=case.category, snapshot=case.snapshot, top_k=3, method=method
    )


def method_identity(method: str) -> str:
    require(method in METHODS, "formal_method")
    config = strategy("A1")
    return f"{config.version}:A1:{config.sha256}:authority:bm25" if method == "A1" else method


def observation(data: Dataset, case: HeldoutCase, method: str) -> dict[str, object]:
    """No exception messages or private text escape via errors."""
    return grade(data, case, method, retrieve(data, case, method))


def grade(data: Dataset, case: HeldoutCase, method: str, result: SearchResult) -> dict[str, object]:
    """Grade supplied results. Recovery calls this without searching again."""
    units = mapped_units(case, data)
    ranking = tuple(hit.chunk_id for hit in result.hits)
    require(len(ranking) <= 3 and len(set(ranking)) == len(ranking), "formal_ranking")
    retriever = Retriever(data)
    selection = catalog(data, snapshot=case.snapshot, category=case.category)
    require(
        result.corpus_version == data.manifest.corpus_version
        and result.index_sha256 == retriever.index.sha256
        and result.method == method_identity(method)
        and result.catalog == selection,
        "formal_result_version",
    )
    expected_status = (
        "policy_conflict" if selection.conflicts else ("ok" if ranking else "no_match")
    )
    require(result.status == expected_status, "formal_result_status")
    chunks = tuple(
        retriever.get_chunk(c, category=case.category, snapshot=case.snapshot) for c in ranking
    )
    for hit, chunk in zip(result.hits, chunks, strict=True):
        require(math.isfinite(hit.score) and hit.score > 0, "formal_score")
        for field in (
            "doc_ref",
            "section_id",
            "source_path",
            "start_char",
            "end_char",
            "start_line",
            "end_line",
        ):
            require(getattr(hit, field) == getattr(chunk, field), "formal_hit_source")
        require(
            hit.excerpt == chunk.text[:600]
            and hit.truncated == (len(chunk.text) > 600)
            and hit.trust == "low_trust_source_data",
            "formal_hit_excerpt",
        )
        require(method != "A1" or chunk.kind != DocKind.FAQ, "formal_a1_authority")
    citations = tuple(
        cite(retriever, c, category=case.category, snapshot=case.snapshot) for c in ranking
    )
    for citation in citations:
        verify_citation(
            citation,
            data,
            retriever,
            category=case.category,
            snapshot=case.snapshot,
            require_authority=False,
        )
    retrieved = frozenset(ranking)
    matched = tuple(u.unit_id for u in units if covered(u, retrieved))
    missing = tuple(u.unit_id for u in units if not covered(u, retrieved))
    metrics: dict[str, str] = {}
    if case.positive:
        existing = score(units, ranking)
        first = existing.first_relevant_rank
        values = (
            Fraction(bool(first)),
            Fraction(len(matched), len(units)),
            Fraction(1, first) if first else Fraction(0),
            Fraction(len(retrieved & related(units)), len(related(units))),
            Fraction(len(matched), len(case.required_unit_ids)),
            Fraction(not missing),
        )
        metrics = {name: str(value) for name, value in zip(METRICS, values, strict=True)}
    negative_pass: bool | None = None
    if not case.positive:
        if case.negative_kind == "no_policy_evidence":
            negative_pass = not ranking
        else:
            negative_pass = bool(units) and not missing
    return {
        "method": method,
        "query_id": case.query_id,
        "positive": case.positive,
        "status": "OK",
        "error_type": None,
        "metrics": metrics,
        "negative_kind": case.negative_kind,
        "negative_pass": negative_pass,
        "ranking": [asdict(h) for h in result.hits],
        "citations": [asdict(c) for c in citations],
        "covered_units": matched,
        "missing_units": missing,
        "faq_hits": sum(c.kind == DocKind.FAQ for c in chunks),
        "search_status": result.status,
        "result_method": result.method,
        "corpus_version": result.corpus_version,
        "index_sha256": result.index_sha256,
        "catalog": asdict(result.catalog),
    }


def empty(case: HeldoutCase, method: str, status: str, error: str | None) -> dict[str, object]:
    return {
        "method": method,
        "query_id": case.query_id,
        "positive": case.positive,
        "status": status,
        "error_type": error,
        "metrics": dict.fromkeys(METRICS, "0") if case.positive else {},
        "negative_kind": case.negative_kind,
        "negative_pass": False if not case.positive else None,
        "ranking": [],
        "citations": [],
        "covered_units": [],
        "missing_units": list(case.required_unit_ids),
        "faq_hits": 0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Allowlisted public output: no query, chunk IDs, Gold units or excerpts."""
    require(len(rows) == 72, "formal_report_count")
    results = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        require(
            len(selected) == 24 and len({r["query_id"] for r in selected}) == 24,
            "formal_report_ids",
        )
        positives = [r for r in selected if r["positive"] is True]
        negatives = [r for r in selected if r["positive"] is False]
        require(len(positives) == 18 and len(negatives) == 6, "formal_denominators")
        sums = dict.fromkeys(METRICS, Fraction(0))
        for row in positives:
            metrics = row["metrics"]
            assert isinstance(metrics, dict)
            require(metrics.keys() == set(METRICS), "formal_metric_keys")
            for name in METRICS:
                value = Fraction(str(metrics[name]))
                require(0 <= value <= 1, "formal_metric_range")
                sums[name] += value
        means = {name: value / 18 for name, value in sums.items()}
        results[method] = {
            "positive_denominator": 18,
            "negative_denominator": 6,
            "metrics": {name: number(value) for name, value in means.items()},
            "quality_thresholds_met": means["evidence_recall_at_3"] >= Fraction(9, 10)
            and means["mrr_at_3"] >= Fraction(4, 5),
            "errors": sum(r["status"] == "ERROR" for r in selected),
            "not_run": sum(r["status"] == "NOT_RUN" for r in selected),
            "negative_passed": sum(r["negative_pass"] is True for r in negatives),
            "negative_results": [
                {
                    k: r[k]
                    for k in ("query_id", "negative_kind", "status", "negative_pass", "faq_hits")
                }
                for r in negatives
            ],
        }
    return {
        "protocol": PROTOCOL,
        "top_k": 3,
        "methods": results,
        "evaluation_mode": "PROCESS_SEPARATED_NON_BLIND",
        "agent_business_safety": "NOT_EVALUATED",
    }


def run(
    data: Dataset, heldout: HeldoutDataset, save: Callable[[dict[str, object]], None]
) -> dict[str, object]:
    validate_heldout(heldout, data)
    rows = [empty(case, method, "NOT_RUN", None) for method in METHODS for case in heldout.cases]
    # Save all denominator cells before any retrieval, and each completed cell.
    save({"event": "INPUT", "gold": asdict(heldout), "rows": list(rows)})
    for i, (method, case) in enumerate((m, c) for m in METHODS for c in heldout.cases):
        try:
            row = observation(data, case, method)
        except Exception as exc:
            row = empty(case, method, "ERROR", type(exc).__name__)
        rows[i] = row
        save({"event": "OBSERVATION", "ordinal": i, "row": row})
    return {"rows": rows, "summary": summarize(rows)}


def verify_records(
    data: Dataset, values: list[dict[str, object]], *, complete: bool
) -> dict[str, object]:
    """Recompute saved evidence math, never retrieval; bind the entire transcript."""
    require(
        bool(values)
        and values[0].keys() == {"event", "gold", "rows"}
        and values[0]["event"] == "INPUT",
        "formal_saved_input",
    )
    heldout = parse_payload(canonical(values[0]["gold"]))
    validate_heldout(heldout, data)
    pairs = [(m, c) for m in METHODS for c in heldout.cases]
    rows = [empty(c, m, "NOT_RUN", None) for m, c in pairs]
    require(canonical(rows) == canonical(values[0]["rows"]), "formal_saved_denominators")
    observations = values[1:-1] if complete else values[1:]
    require(len(observations) == 72 if complete else len(observations) <= 72, "formal_saved_count")
    for i, entry in enumerate(observations):
        require(
            entry.keys() == {"event", "ordinal", "row"}
            and entry["event"] == "OBSERVATION"
            and type(entry["ordinal"]) is int
            and entry["ordinal"] == i,
            "formal_saved_order",
        )
        row = entry["row"]
        require(isinstance(row, dict), "formal_saved_row")
        assert isinstance(row, dict)
        method, case = pairs[i]
        if row.get("status") == "OK":
            ranking = row.get("ranking")
            require(isinstance(ranking, list), "formal_saved_ranking")
            assert isinstance(ranking, list)
            require(all(isinstance(h, dict) for h in ranking), "formal_saved_hit")
            hits = tuple(
                (StrategyHit if method == "A1" else SearchHit)(**cast(dict[str, Any], h))
                for h in ranking
            )
            require(
                all(
                    isinstance(row.get(k), str)
                    for k in ("search_status", "corpus_version", "index_sha256", "result_method")
                ),
                "formal_saved_version",
            )
            selection = catalog(data, snapshot=case.snapshot, category=case.category)
            require(
                canonical(row.get("catalog")) == canonical(asdict(selection)),
                "formal_saved_catalog",
            )
            result = SearchResult(
                str(row["search_status"]),
                str(row["corpus_version"]),
                str(row["index_sha256"]),
                str(row["result_method"]),
                hits,
                selection,
            )
            expected = grade(data, case, method, result)
        else:
            require(
                row.get("status") == "ERROR" and isinstance(row.get("error_type"), str),
                "formal_saved_status",
            )
            error = row["error_type"]
            assert isinstance(error, str)
            expected = empty(case, method, "ERROR", error)
        require(canonical(row) == canonical(expected), "formal_saved_math")
        rows[i] = expected
    report: dict[str, object] = {"rows": rows, "summary": summarize(rows)}
    if complete:
        require(
            values[-1].keys() == {"event", "report"}
            and values[-1]["event"] == "REPORT"
            and canonical(values[-1]["report"]) == canonical(report),
            "formal_saved_report",
        )
    return report
