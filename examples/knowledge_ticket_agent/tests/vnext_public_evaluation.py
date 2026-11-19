"""Fixed public vNext measurement. No formal operator, private path or Gold loader."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import knowledge_ticket_agent
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.agent_grading import grade_run
from knowledge_ticket_agent.business_evaluation import public_manifest
from knowledge_ticket_agent.citations import cite, verify_citation
from knowledge_ticket_agent.evaluation_runtime import execute_public_task
from knowledge_ticket_agent.evidence import EvidenceUnit, score
from knowledge_ticket_agent.loaders import markdown
from knowledge_ticket_agent.retrieval import Retriever, SearchResult
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.retrieval_vnext import VNextRetriever
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind, SectionSpec
from knowledge_ticket_agent.semantic_model import E5Encoder

import agent_harness_lab

HASHES = {
    "dev": "57ae516cc3f1cde82be1ef181433f23c0759a58fb496083aedc122bc97b92d4d",
    "challenge": "5ca6cbbeb4c7ae6ace7f3b81fa2da1d91665a588c37bbd2ef043c99182de1eeb",
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_source_fingerprint() -> str:
    rows = []
    for package in (agent_harness_lab, knowledge_ticket_agent):
        file_path = package.__file__
        assert file_path is not None
        for source in sorted(Path(file_path).parent.glob("*.py")):
            rows.append(
                (
                    package.__name__ + "." + source.stem,
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                )
            )
    return digest(rows)


def source_corpus(base: Dataset, fixture: dict[str, Any]) -> tuple[Dataset, list[str]]:
    refs = sorted(d.ref for d in base.manifest.documents if d.kind == DocKind.POLICY)
    policies = []
    for policy in base.documents:
        metadata = policy.metadata
        sections = [(s.section_id, s.heading, s.body) for s in policy.sections]
        if metadata.ref in refs:
            topic = fixture["topics"][refs.index(metadata.ref)]
            metadata = replace(metadata, title=topic["title"])
            sections.extend(
                (
                    ("vn_rule", "Conditions", topic["rule"]),
                    ("vn_record", "Service record", topic["record"]),
                )
            )
        if metadata.kind not in {DocKind.LEGACY, DocKind.CONFLICT}:
            metadata = replace(
                metadata, scope=tuple(dict.fromkeys((*metadata.scope, Category.OTHER)))
            )
        metadata = replace(metadata, sections=tuple(SectionSpec(i, h) for i, h, _ in sections))
        text = f"# {metadata.title}\nSynthetic: true; data_origin: authored_fiction\n\n"
        text += "".join(f"## {i} | {h}\n{body}\n\n" for i, h, body in sections)
        policies.append(markdown(metadata, text))
    raw = {p.metadata.path: p.text.encode() for p in policies}
    resources = tuple(
        replace(r, sha256=hashlib.sha256(raw[r.path]).hexdigest(), bytes=len(raw[r.path]))
        if r.path in raw
        else r
        for r in base.manifest.resources
    )
    manifest = replace(
        base.manifest,
        corpus_version=fixture["version"] + "-corpus",
        documents=tuple(p.metadata for p in policies),
        resources=resources,
    )
    return replace(
        base, manifest=manifest, manifest_sha256=digest(asdict(manifest)), documents=tuple(policies)
    ), refs


def evaluate(split: str, method: str, directory: Path | None) -> dict[str, Any]:
    folder = Path(__file__).resolve().parents[1] / "verification/retrieval-vnext-v1"
    raw = (folder / (split + ".json")).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[split]
    fixture = json.loads(raw)
    base = load_dataset()
    data, refs = source_corpus(base, fixture)
    start = time.perf_counter()
    encoder = None if method == "A1" else E5Encoder(directory or Path("missing-explicit-model"))
    startup = time.perf_counter() - start

    def factory(data: Dataset) -> Retriever:
        return (
            A1Retriever(data) if encoder is None else VNextRetriever(data, encoder, variant=method)
        )

    r = factory(data)
    rows: list[dict[str, Any]] = []
    durations = []
    for case in fixture["cases"]:
        units = tuple(
            EvidenceUnit(
                section,
                (
                    tuple(
                        c.chunk_id
                        for c in r.index.chunks
                        if c.doc_ref == refs[case["topic"]] and c.section_id == "vn_" + section
                    ),
                ),
            )
            for section in case["sections"]
        )
        error = None
        elapsed = []
        observed = []
        for _ in range(3):
            start = time.perf_counter()
            try:
                found = r.search(case["query"], category=Category.OTHER)
                for h in found.hits:
                    verify_citation(
                        cite(r, h.chunk_id, category=Category.OTHER),
                        data,
                        r,
                        category=Category.OTHER,
                    )
                    assert h.doc_ref in found.catalog.authoritative_refs
            except (FixtureError, ValueError, OSError) as exc:
                error = exc.code if isinstance(exc, FixtureError) else type(exc).__name__
                found = SearchResult(
                    "retrieval_error",
                    r.index.corpus_version,
                    r.index.sha256,
                    method,
                    (),
                    r.selection(category=Category.OTHER),
                )
            elapsed.append(time.perf_counter() - start)
            observed.append(asdict(found))
        metric = score(units, tuple(h.chunk_id for h in found.hits))
        durations.append(elapsed[0])
        rows.append(
            {
                "case_id": case["id"],
                "variant": case["variant"],
                "query": case["query"],
                "method": found.method,
                "index_sha256": found.index_sha256,
                "metrics": asdict(metric),
                "complete_evidence": not metric.missing_units,
                "hits": [asdict(h) for h in found.hits],
                "error": error,
                "deterministic": all(o == observed[0] for o in observed),
                "cold_s": elapsed[0],
                "warm_s": statistics.median(elapsed[1:]),
            }
        )
    controls: list[dict[str, Any]] = []
    for name, query in (("empty", ""), ("oov", "zzxqvvqzz")):
        result = r.search(query, category=Category.OTHER)
        controls.append(
            {
                "id": name,
                "hits": len(result.hits),
                "status": result.status,
                "passed": not result.hits,
            }
        )
    result = r.search("retention", category=Category.EXPORT, snapshot="conflict")
    controls.append({"id": "conflict", "passed": result.status == "policy_conflict"})
    result = r.search("first purchase", category=Category.BILLING)
    controls.append(
        {
            "id": "category_scope",
            "passed": all(h.doc_ref in result.catalog.authoritative_refs for h in result.hits),
        }
    )
    faq = next(c for c in r.index.chunks if c.kind == DocKind.FAQ)
    try:
        verify_citation(cite(r, faq.chunk_id), data, r)
        rejected = False
    except FixtureError:
        rejected = True
    controls.append({"id": "faq_authority", "passed": rejected})
    damaged = replace(
        data, documents=(replace(data.documents[0], text="unbound"), *data.documents[1:])
    )
    try:
        factory(damaged)
        rejected = False
    except FixtureError:
        rejected = True
    controls.append({"id": "source_tamper", "passed": rejected})
    try:
        r.search("x" * 4001)
        rejected = False
    except FixtureError:
        rejected = True
    controls.append({"id": "bad_query", "passed": rejected})
    # AND-unit partial source remains incomplete in the unchanged evidence scorer.
    two = (EvidenceUnit("two", (("a", "b"),)),)
    controls.append({"id": "partial_evidence", "passed": score(two, ("a",)).recall == 0})
    business: list[dict[str, Any]] = []
    public_cases = public_manifest(base).cases
    # Twelve public business oracles plus the separately declared refusal case.
    # Intentional transport/not-run cases are retained below as failure controls.
    for case in public_cases[:13]:
        run = execute_public_task(base, case.task, retriever_factory=factory)
        grade = grade_run(run)
        state = run.app.workflow.store.read(case.task.ticket_id)
        searches = [
            row
            for row in run.app.registry.records
            if row.call.name == "kb_search" and row.result.ok
        ]
        observed_methods = [json.loads(row.result.content)["result"]["method"] for row in searches]
        consistent = (
            bool(searches)
            and json.loads(searches[0].call.arguments)["query"] == case.task.query
            and len(set(observed_methods)) == 1
        )
        assessments = [
            json.loads(row.result.content)
            for row in run.app.registry.records
            if row.call.name == "ticket_assess" and row.result.ok
        ]
        decision = assessments[-1].get("decision") if assessments else None
        correct = (
            not grade.issues
            and state.status.value == case.oracle.final_status
            and state.revision == case.oracle.final_revision
            and decision == case.oracle.decision
            and consistent
        )
        business.append(
            {
                "case_id": case.case_id,
                "correct": correct,
                "issues": grade.issues,
                "status": state.status.value,
                "revision": state.revision,
                "outcome": grade.outcome,
                "decision": decision,
                "commits": grade.commits,
                "method_consistent": consistent,
                "methods": sorted(set(observed_methods)),
                "model_rounds": run.result.model_rounds,
                "tools": run.result.tool_calls,
                "stop": run.result.stop_reason.value,
            }
        )
    faults: list[dict[str, Any]] = []
    fault_cases = list(public_cases[13:]) + [
        replace(
            public_cases[0],
            case_id="VN-CONTROL-" + fault,
            task=replace(public_cases[0].task, fault=fault),
        )
        for fault in (
            "atomic_abort",
            "readback",
            "stale_approval",
            "corrupt_evidence",
            "missing_evidence",
            "budget",
            "invalid_provenance",
        )
    ]
    for case in fault_cases:
        try:
            run = execute_public_task(base, case.task, retriever_factory=factory)
            grade = grade_run(run)
            expected_commit = case.task.fault in {"readback", "invalid_provenance"}
            safe = grade.commits == int(expected_commit)
            if case.task.fault == "invalid_provenance":
                safe = safe and bool(grade.issues)
            if case.task.fault == "readback":
                safe = (
                    safe
                    and run.result.stop_reason.value == "tool_error"
                    and not grade.readback_verified
                )
            faults.append(
                {
                    "case_id": case.case_id,
                    "fault": case.task.fault,
                    "passed": safe,
                    "commits": grade.commits,
                    "issues": grade.issues,
                    "stop": run.result.stop_reason.value,
                    "readback_verified": grade.readback_verified,
                }
            )
        except (OSError, ValueError) as exc:
            expected = case.task.fault == "not_run" and str(exc) == "declared_not_run"
            faults.append(
                {
                    "case_id": case.case_id,
                    "fault": case.task.fault,
                    "passed": expected,
                    "commits": 0 if expected else None,
                    "error": str(exc),
                }
            )
    selected = sorted(durations)
    aggregate = {
        "recall": statistics.mean(row["metrics"]["recall"] for row in rows),
        "mrr": statistics.mean(row["metrics"]["reciprocal_rank"] for row in rows),
        "complete_evidence": statistics.mean(row["complete_evidence"] for row in rows),
        "english_recall": statistics.mean(
            row["metrics"]["recall"] for row in rows if row["variant"] not in {"zh", "mixed"}
        ),
        "query_p95_s": selected[math.ceil(0.95 * len(selected)) - 1],
        "startup_s": startup,
        "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "business_correct": sum(row["correct"] for row in business),
        "business_denominator": len(business),
        "fault_controls_passed": sum(row["passed"] for row in faults),
        "fault_controls_denominator": len(faults),
        "unauthorized_effects": sum(
            row["commits"] > case.oracle.final_revision
            for row, case in zip(business, public_cases[:13], strict=True)
        ),
    }
    return {
        "scope": "PUBLIC_ENGINEERING_ONLY",
        "version": fixture["version"],
        "data_sha256": HASHES[split],
        "method": method,
        "profile": getattr(r, "method_identity", "A1"),
        "model_identity": encoder.identity if encoder else None,
        "runtime": {
            "python": platform.python_version(),
            "OS": platform.system(),
            "machine": platform.machine(),
            "semantic_packages": {
                n: importlib.metadata.version(n) for n in ("onnxruntime", "tokenizers", "numpy")
            }
            if encoder
            else {},
        },
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": runtime_source_fingerprint(),
        "rows": rows,
        "aggregate": aggregate,
        "by_variant": {
            v: {
                "recall": statistics.mean(
                    row["metrics"]["recall"] for row in rows if row["variant"] == v
                ),
                "mrr": statistics.mean(
                    row["metrics"]["reciprocal_rank"] for row in rows if row["variant"] == v
                ),
            }
            for v in sorted({row["variant"] for row in rows})
        },
        "controls": controls,
        "business": business,
        "fault_controls": faults,
        "local_embedding_windows": encoder.windows if encoder else 0,
        "external_model_API_requests": 0,
        "new_formal_executions": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed public vNext measurement; new outputs only")
    parser.add_argument("split", choices=("dev", "challenge"))
    parser.add_argument("method", choices=("A1", "D", "H"))
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-decision", type=Path)
    args = parser.parse_args()
    if args.split == "challenge":
        assert args.dev_decision is not None
        decision = json.loads(args.dev_decision.read_text())
        assert (
            decision["scope"] == "FROZEN_PUBLIC_DEV_VNEXT"
            and decision["helper_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        assert decision["runtime_source_sha256"] == runtime_source_fingerprint()
    result = evaluate(args.split, args.method, args.model_directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "method": args.method,
                "split": args.split,
                "aggregate": result["aggregate"],
                "controls": result["controls"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
