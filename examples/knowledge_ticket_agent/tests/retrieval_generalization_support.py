"""Finite public experiment; never imports a formal operator or private inputs."""

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.evidence import EvidenceUnit, score
from knowledge_ticket_agent.loaders import markdown
from knowledge_ticket_agent.quality_evidence import AnswerSpec, resolve_policy
from knowledge_ticket_agent.retrieval import Retriever, SearchHit, SearchResult, lexical_rank, terms
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind, SectionSpec
from knowledge_ticket_agent.strict import decode, parse_json, require

VERSION = "lqd-public-generalization-prototypes-1.0.0"
STOP_WORDS = frozenset(
    "a about an and are as at be been being but by can could did do does for from had has have "
    "how i if in into is it its me my of on or our please should so some tell that the their "
    "them there these they this to us was we were what when where which who will with would "
    "you your".split()
)
DATA_HASHES = {
    "dev": "8395fbe80a9f5ca1ae6fdc6f1511d058058ce1a86513e0c13711f6bf33487f68",
    "challenge": "b7783772124b41201c13c54870a6a08591fa0896f5e6ad7df4ec6eb871cf9b6a",
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def config(candidate: str) -> dict[str, object]:
    require(candidate in {"G1", "G2"}, "candidate")
    return {
        "version": VERSION,
        "candidate": candidate,
        "stop_words": sorted(STOP_WORDS),
        "normalizer": "NFKC-casefold-ascii-alnum-unique-all-terms",
        "max_query": 4000,
        "query_batch": 256,
        "bm25": {"k1": 1.2, "b": 0.75},
        "fusion": "none" if candidate == "G1" else "equal-flat-metadata-body-RRF60",
        "scope": "authority-only-current-snapshot-category",
        "tie": "chunk_id",
    }


def batches(query: str) -> tuple[str, ...]:
    require(type(query) is str and len(query) <= 4000 and "\x00" not in query, "query")
    words = sorted(set(terms(query)) - STOP_WORDS)
    require(all(len(w) <= 256 for w in words), "query_term")
    result: list[str] = []
    current = ""
    for word in words:
        if len(current) + bool(current) + len(word) > 256:
            result.append(current)
            current = ""
        current = (current + " " + word).strip()
    if current:
        result.append(current)
    return tuple(result)


def rank(
    documents: tuple[tuple[str, str], ...], queries: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    scores: dict[str, float] = {}
    for query in queries:
        for identifier, value in lexical_rank(documents, query):
            scores[identifier] = scores.get(identifier, 0.0) + value
    return tuple(sorted(scores.items(), key=lambda row: (-row[1], row[0])))


class PrototypeRetriever(Retriever):
    """Research-only opt-in: no business routing or legacy configuration mutation."""

    def __init__(self, data: Dataset, candidate: str) -> None:
        super().__init__(data)
        self.candidate = candidate
        self.config_sha256 = digest(config(candidate))

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
    ) -> SearchResult:
        require(type(top_k) is int and 1 <= top_k <= 5, "top_k")
        require(method == "bm25", "ranking_method")
        queries = batches(query)
        selection = self.selection(snapshot=snapshot, category=category)
        chunks = tuple(c for c in self.index.chunks if c.doc_ref in selection.authoritative_refs)
        flat = tuple((c.chunk_id, f"{c.title}\n{c.heading}\n{c.text}") for c in chunks)
        ranking = rank(flat, queries)
        if self.candidate == "G2":
            scores: dict[str, float] = {}
            for ordered in (
                ranking,
                rank(tuple((c.chunk_id, c.title + "\n" + c.heading) for c in chunks), queries),
                rank(tuple((c.chunk_id, c.text) for c in chunks), queries),
            ):
                for position, (identifier, _) in enumerate(ordered, 1):
                    scores[identifier] = scores.get(identifier, 0.0) + 1 / (60 + position)
            ranking = tuple(sorted(scores.items(), key=lambda row: (-row[1], row[0])))
        by_id = {c.chunk_id: c for c in chunks}
        hits = []
        for identifier, value in ranking[:top_k]:
            c = by_id[identifier]
            hits.append(
                SearchHit(
                    identifier,
                    c.doc_ref,
                    c.section_id,
                    c.source_path,
                    c.start_char,
                    c.end_char,
                    c.start_line,
                    c.end_line,
                    value,
                    c.text[:600],
                    len(c.text) > 600,
                )
            )
        return SearchResult(
            "policy_conflict" if selection.conflicts else "ok" if hits else "no_match",
            self.index.corpus_version,
            self.index.sha256,
            f"{VERSION}:{self.candidate}:{self.config_sha256}",
            tuple(hits),
            selection,
        )


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    topic: str
    statement: str
    query: str
    variant: str
    target_slot: int
    required_sections: tuple[str, ...]


@dataclass(frozen=True)
class Matrix:
    version: str
    scope: str
    origin: str
    cases: tuple[PublicCase, ...]
    control_ids: tuple[str, ...]


def matrix(split: str) -> Matrix:
    require(split in DATA_HASHES, "public_split")
    folder = Path(__file__).resolve().parents[1] / "verification/retrieval-generalization-v1"
    raw = (folder / (split + ".json")).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == DATA_HASHES[split], "public_data_hash")
    result = decode(Matrix, parse_json(raw.decode()))
    require(len(result.cases) == 44 and len({c.case_id for c in result.cases}) == 44, "matrix")
    return result


def corpus(base: Dataset, case: PublicCase) -> tuple[Dataset, str]:
    """New source-bound in-memory fiction, retaining schema/rule-linked scaffolding."""
    refs = sorted(d.ref for d in base.manifest.documents if d.kind == DocKind.POLICY)
    target = refs[case.target_slot]
    documents = []
    for policy in base.documents:
        metadata = policy.metadata
        sections = [(s.section_id, s.heading, s.body) for s in policy.sections]
        if metadata.kind == DocKind.POLICY:
            title = "Fictional " + " ".join(case.topic.split()[:2]) + " inventory"
            if metadata.ref == target:
                title = "Fictional service guide"
                heading = case.topic
                body = case.topic + ". " + case.statement
                if case.variant == "title_only":
                    title, heading = case.topic, "Conditions"
                    body = "Members must submit this fictional service request by noon."
                elif case.variant == "heading_only":
                    body = "Members must submit this fictional service request by noon."
                elif case.variant == "body_only":
                    heading = "Detailed conditions"
                sections.append(("rg_rule", heading, body))
                if case.variant == "complementary":
                    sections.append(
                        (
                            "rg_record",
                            case.topic + " records",
                            "Record the "
                            + case.topic
                            + " request in the service log. This record is also required.",
                        )
                    )
            else:
                if case.variant == "body_only":
                    title = case.topic + " statistics"
                sections.append(
                    (
                        "rg_other",
                        " ".join(case.topic.split()[:2]) + " inventory",
                        "This unrelated policy lists monthly inventory totals for "
                        + " ".join(case.topic.split()[:2])
                        + ". It provides statistics, not service conditions.",
                    )
                )
            metadata = replace(metadata, title=title)
        if metadata.kind not in {DocKind.CONFLICT, DocKind.LEGACY}:
            metadata = replace(
                metadata, scope=tuple(dict.fromkeys((*metadata.scope, Category.OTHER)))
            )
        metadata = replace(metadata, sections=tuple(SectionSpec(i, h) for i, h, _ in sections))
        text = f"# {metadata.title}\nSynthetic: true; data_origin: authored_fiction\n\n"
        text += "".join(f"## {i} | {h}\n{body}\n\n" for i, h, body in sections)
        documents.append(markdown(metadata, text))
    raw_by_path = {p.metadata.path: p.text.encode() for p in documents}
    resources = tuple(
        replace(
            r,
            sha256=hashlib.sha256(raw_by_path[r.path]).hexdigest(),
            bytes=len(raw_by_path[r.path]),
        )
        if r.path in raw_by_path
        else r
        for r in base.manifest.resources
    )
    manifest = replace(
        base.manifest,
        corpus_version="lqd-public-robustness-corpus-1.0.0",
        documents=tuple(p.metadata for p in documents),
        resources=resources,
    )
    return replace(
        base,
        manifest=manifest,
        manifest_sha256=digest(asdict(manifest)),
        documents=tuple(documents),
    ), target


def retriever(data: Dataset, name: str) -> Retriever:
    return A1Retriever(data) if name == "A1" else PrototypeRetriever(data, name)


def controls(base: Dataset, name: str) -> list[dict[str, object]]:
    r = retriever(base, name)
    rows: list[dict[str, object]] = []
    for label, query in (
        ("empty", ""),
        ("out_of_vocabulary", "zzxqvv"),
        ("function_words_only", "and the to"),
    ):
        found = r.search(query)
        rows.append(
            {
                "id": label,
                "passed": not found.hits if label != "function_words_only" else True,
                "hits": len(found.hits),
                "status": found.status,
            }
        )
    found = r.search("retention", category=Category.BILLING)
    rows.append(
        {
            "id": "wrong_category",
            "passed": all(h.doc_ref in found.catalog.authoritative_refs for h in found.hits),
        }
    )
    conflict = r.search("retention", category=Category.EXPORT, snapshot="conflict")
    rows.append(
        {
            "id": "conflict",
            "passed": conflict.status == "policy_conflict" and bool(conflict.catalog.conflicts),
        }
    )
    faq = next(c for c in r.index.chunks if c.kind == DocKind.FAQ)
    try:
        verify_citation(cite(r, faq.chunk_id), base, r)
        rejected = False
    except FixtureError as exc:
        rejected = exc.code == "faq_not_authority"
    rows.append(
        {
            "id": "faq_authority",
            "passed": rejected
            and all(h.doc_ref in found.catalog.authoritative_refs for h in found.hits),
        }
    )
    resolution = resolve_policy(
        base,
        "KT-D01",
        "public missing documentary evidence",
        AnswerSpec(
            Category.BILLING, "credit_guidance", (("POL-BILL-01@1.0", "missing_public_section"),)
        ),
        retriever=r,
    )
    rows.append(
        {
            "id": "missing_evidence",
            "passed": resolution.decision.status != "answerable"
            and resolution.approval_calls == resolution.commits == 0
            and resolution.unchanged_state_verified,
            "decision": resolution.decision.status,
        }
    )
    damaged = replace(
        base,
        documents=(
            replace(base.documents[0], text=base.documents[0].text + "unbound"),
            *base.documents[1:],
        ),
    )
    try:
        retriever(damaged, name)
        rejected = False
    except FixtureError as exc:
        rejected = exc.code == "source_integrity"
    rows.append({"id": "source_tamper", "passed": rejected})
    return rows


def measure(split: str) -> dict[str, Any]:
    fixture = matrix(split)
    base = load_dataset()
    methods: dict[str, Any] = {}
    for name in ("A1", "G1", "G2"):
        rows: list[dict[str, Any]] = []
        for case in fixture.cases:
            data, target = corpus(base, case)
            start = time.perf_counter_ns()
            r = retriever(data, name)
            construct_ms = (time.perf_counter_ns() - start) / 1e6
            timings = []
            repeats = []
            error = None
            for _ in range(5):
                start = time.perf_counter_ns()
                try:
                    found = r.search(case.query, category=Category.OTHER)
                except FixtureError as exc:
                    error = exc.code
                    found = SearchResult(
                        "retrieval_error",
                        r.index.corpus_version,
                        r.index.sha256,
                        name,
                        (),
                        r.selection(category=Category.OTHER),
                    )
                timings.append((time.perf_counter_ns() - start) / 1e6)
                repeats.append(asdict(found))
            assert all(item == repeats[0] for item in repeats)
            units = tuple(
                EvidenceUnit(
                    section,
                    (
                        tuple(
                            c.chunk_id
                            for c in r.index.chunks
                            if c.doc_ref == target and c.section_id == section
                        ),
                    ),
                )
                for section in case.required_sections
            )
            metric = score(units, tuple(h.chunk_id for h in found.hits))
            for h in found.hits:
                verify_citation(
                    cite(r, h.chunk_id, category=Category.OTHER), data, r, category=Category.OTHER
                )
                assert h.doc_ref in found.catalog.authoritative_refs
                assert h.trust == "low_trust_source_data"
            rows.append(
                {
                    "case_id": case.case_id,
                    "variant": case.variant,
                    "query": case.query,
                    "index_sha256": r.index.sha256,
                    "metrics": asdict(metric),
                    "verified_coverage": metric.recall,
                    "complete_evidence": not metric.missing_units,
                    "status": found.status,
                    "error": error,
                    "hits": [asdict(h) for h in found.hits],
                    "deterministic": True,
                    "construct_ms": construct_ms,
                    "search_median_ms": statistics.median(timings),
                    "query_batches": len(batches(case.query)) if name != "A1" else None,
                }
            )
        methods[name] = {
            "rows": rows,
            "controls": controls(base, name),
            "by_variant": {
                variant: {
                    "denominator": sum(row["variant"] == variant for row in rows),
                    "recall": statistics.mean(
                        row["metrics"]["recall"] for row in rows if row["variant"] == variant
                    ),
                    "mrr": statistics.mean(
                        row["metrics"]["reciprocal_rank"]
                        for row in rows
                        if row["variant"] == variant
                    ),
                }
                for variant in sorted({row["variant"] for row in rows})
            },
            "aggregate": {
                "recall": statistics.mean(row["metrics"]["recall"] for row in rows),
                "mrr": statistics.mean(row["metrics"]["reciprocal_rank"] for row in rows),
                "verified_coverage": statistics.mean(row["verified_coverage"] for row in rows),
                "complete_evidence": statistics.mean(row["complete_evidence"] for row in rows),
                "search_median_ms": statistics.median(row["search_median_ms"] for row in rows),
                "construct_median_ms": statistics.median(row["construct_ms"] for row in rows),
            },
        }
    for name in ("G1", "G2"):
        item, baseline = methods[name], methods["A1"]
        regressed = [
            c["case_id"]
            for c, b in zip(item["rows"], baseline["rows"], strict=True)
            if c["metrics"]["recall"] < b["metrics"]["recall"]
            or c["complete_evidence"] < b["complete_evidence"]
        ]
        mrr_regressed = [
            c["case_id"]
            for c, b in zip(item["rows"], baseline["rows"], strict=True)
            if c["metrics"]["reciprocal_rank"] < b["metrics"]["reciprocal_rank"]
        ]
        a, b = item["aggregate"], baseline["aggregate"]
        improved = a["recall"] - b["recall"] >= 0.03 or (
            a["recall"] >= b["recall"] and a["mrr"] - b["mrr"] >= 0.05
        )
        item.update(
            recall_or_complete_regressions=regressed,
            mrr_regressions=mrr_regressed,
            qualifies=improved
            and not regressed
            and not any(row["error"] for row in item["rows"])
            and len(mrr_regressed) <= 0.05 * 44
            and all(c["passed"] for c in item["controls"])
            and a["search_median_ms"] < 50
            and a["search_median_ms"] <= 5 * b["search_median_ms"],
        )
    valid = [n for n in ("G1", "G2") if methods[n]["qualifies"]]
    selected = (
        max(
            valid,
            key=lambda n: (
                methods[n]["aggregate"]["recall"],
                methods[n]["aggregate"]["mrr"],
                n == "G1",
            ),
        )
        if valid
        else None
    )
    return {
        "scope": fixture.scope,
        "version": fixture.version,
        "data_sha256": DATA_HASHES[split],
        "prototype_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configs": {n: {"config": config(n), "sha256": digest(config(n))} for n in ("G1", "G2")},
        "ranking_denominator": 44,
        "controls_per_method": 8,
        "methods": methods,
        "dev_selected": selected if split == "dev" else None,
        "formal_execution": "NOT_RUN",
        "model_requests": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed public research only; new output only")
    parser.add_argument("split", choices=("dev", "challenge"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-report", type=Path)
    args = parser.parse_args()
    decision = None
    if args.split == "challenge":
        require(args.dev_report is not None, "dev_decision_required")
        dev = json.loads(args.dev_report.read_text())
        require(
            dev["data_sha256"] == DATA_HASHES["dev"]
            and dev["prototype_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frozen_dev_decision",
        )
        decision = dev["dev_selected"]
    result = measure(args.split)
    result["frozen_dev_selection"] = decision
    result["adopt"] = bool(decision and result["methods"][decision]["qualifies"])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "split": args.split,
                "selection": result["dev_selected"],
                "adopt": result["adopt"],
                "methods": {
                    n: {
                        "aggregate": m["aggregate"],
                        "qualifies": m.get("qualifies"),
                        "recall_regressions": m.get("recall_or_complete_regressions"),
                        "mrr_regressions": m.get("mrr_regressions"),
                        "controls": m["controls"],
                    }
                    for n, m in result["methods"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
