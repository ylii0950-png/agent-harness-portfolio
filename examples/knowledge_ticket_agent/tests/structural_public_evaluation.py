"""Fresh public-only stage diagnostics. Expected units stay in this scorer."""

import argparse
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
from knowledge_ticket_agent.retrieval import lexical_rank
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.retrieval_strategies import FieldDocument, field_rank
from knowledge_ticket_agent.retrieval_vnext import VNextRetriever, _cos, _lexical, segments
from knowledge_ticket_agent.schemas import Category, DocKind
from knowledge_ticket_agent.semantic_model import E5Encoder

from .structural_corpus import VERSION as CORPUS_VERSION
from .structural_corpus import build
from .vnext_public_evaluation import runtime_source_fingerprint

HASHES = {
    "dev": "164e101cd92fc0088ea9844b6ac2fddbf88058c9ab8b2e210adc8fb222f479ec",
    "challenge": "881155a1be7320a1dc01b86c46ac3c2768e70ce90520d187049d8f1b693e3366",
}


def legacy_trace(r: Any, query: str) -> dict[str, Any]:
    scope = r.selection(category=Category.OTHER)
    chunks = tuple(c for c in r.index.chunks if c.doc_ref in scope.authoritative_refs)
    if isinstance(r, A1Retriever):
        import re
        import unicodedata

        q = query
        if len(q) > 256:
            words = list(
                dict.fromkeys(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", q).casefold()))
            )
            vocabulary = {
                w
                for c in r.index.chunks
                for w in re.findall(r"[a-z0-9]+", (c.title + " " + c.heading).casefold())
            }
            kept: list[str] = []
            for w in [w for w in words if w in vocabulary] + [
                w for w in words if w not in vocabulary
            ]:
                if len(" ".join((*kept, w))) <= 256:
                    kept.append(w)
            q = " ".join(kept)
        a1_ranking = field_rank(
            tuple(FieldDocument(c.chunk_id, c.title, c.heading, c.text) for c in chunks),
            q,
            weights=r.config.weights,
        )
        if r.config.scoring == "flat":
            a1_ranking = lexical_rank(
                tuple((c.chunk_id, f"{c.title}\n{c.heading}\n{c.text}") for c in chunks), q
            )
        ids = [i for i, _ in a1_ranking[:32]]
        return {
            "legal": [c.chunk_id for c in chunks],
            "dense": [],
            "lexical": ids,
            "intersection": [],
            "generated": ids,
            "post_decision": ids,
            "excluded": {},
        }
    parts = segments(query)
    questions = r.encoder.embed(parts, query=True)
    dense = (
        {
            c.chunk_id: max((_cos(q, r._vectors[c.chunk_id]) for q in questions), default=-1.0)
            for c in chunks
        }
        if parts
        else {}
    )
    lexical = _lexical(
        tuple((c.chunk_id, f"{c.title}\n{c.heading}\n{c.text}") for c in chunks), query
    )
    dense_ids = sorted(dense, key=lambda i: (-dense[i], i))[:16]
    lexical_ids = sorted(lexical, key=lambda i: (-lexical[i], i))[:16]
    allowed = {i for i, v in dense.items() if v >= r.config.similarity_floor}
    actual_dense = sorted(allowed, key=lambda i: (-dense[i], i))[:16]
    actual = set(actual_dense) | set(lexical_ids) if r.config.variant == "H" else set(actual_dense)
    generated = set(dense_ids) | set(lexical_ids) if r.config.variant == "H" else set(dense_ids)
    return {
        "legal": [c.chunk_id for c in chunks],
        "dense": dense_ids,
        "lexical": lexical_ids,
        "intersection": sorted(set(dense_ids) & set(lexical_ids)),
        "generated": sorted(generated),
        "post_decision": sorted(actual & allowed),
        "excluded": {i: "dense_floor" for i in generated if i not in allowed},
    }


def evaluate(split: str, method: str, directory: Path) -> dict[str, Any]:
    folder = Path(__file__).resolve().parents[1] / "verification/retrieval-structural-v1"
    raw = (folder / f"{split}.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[split]
    fixture = json.loads(raw)
    data, refs = build(load_dataset(), fixture)
    start = time.perf_counter()
    encoder = None if method == "A1" else E5Encoder(directory)
    if method == "S":
        from knowledge_ticket_agent.retrieval_structural import StructuralRetriever

        assert encoder is not None
        r: Any = StructuralRetriever(data, encoder)
    else:
        r = A1Retriever(data) if encoder is None else VNextRetriever(data, encoder, variant=method)
    startup = time.perf_counter() - start
    rows = []
    for case in fixture["cases"]:
        start = time.perf_counter()
        found = r.search(case["query"], category=Category.OTHER)
        elapsed = time.perf_counter() - start
        trace = (
            asdict(r.explain(case["query"], category=Category.OTHER).trace)
            if method == "S"
            else legacy_trace(r, case["query"])
        )
        again = r.search(case["query"], category=Category.OTHER)
        assert found == again
        for hit in found.hits:
            verify_citation(
                cite(r, hit.chunk_id, category=Category.OTHER), data, r, category=Category.OTHER
            )
            assert hit.doc_ref in found.catalog.authoritative_refs
        units = (
            tuple(
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
            if case["positive"]
            else ()
        )
        metrics = asdict(score(units, tuple(h.chunk_id for h in found.hits))) if units else None
        candidate_recall = (
            sum(covered(u, frozenset(trace["generated"])) for u in units) / len(units)
            if units
            else None
        )
        retained_recall = (
            sum(covered(u, frozenset(trace["post_decision"])) for u in units) / len(units)
            if units
            else None
        )
        losses = []
        for u in units:
            if covered(u, frozenset(h.chunk_id for h in found.hits)):
                continue
            stage = (
                "reranking"
                if covered(u, frozenset(trace["post_decision"]))
                else "support_decision"
                if covered(u, frozenset(trace["generated"]))
                else "candidate_generation"
            )
            losses.append(
                {
                    "unit": u.unit_id,
                    "stage": stage,
                    "source_reasons": {
                        i: trace["excluded"].get(i, stage)
                        for bundle in u.alternatives
                        for i in bundle
                    },
                }
            )
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "positive": case["positive"],
                "language": case["language"],
                "multi": len(units) > 1,
                "accepted": bool(found.hits),
                "metrics": metrics,
                "candidate_recall": candidate_recall,
                "post_decision_recall": retained_recall,
                "candidate_K": len(trace["generated"]),
                "trace": trace,
                "hits": [asdict(h) for h in found.hits],
                "status": found.status,
                "method": found.method,
                "losses": losses,
                "seconds": elapsed,
                "deterministic": True,
            }
        )
    positives = [x for x in rows if x["positive"]]
    negatives = [x for x in rows if not x["positive"]]

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "denominator": len(group),
            "candidate_recall": statistics.mean(x["candidate_recall"] for x in group),
            "post_decision_recall": statistics.mean(x["post_decision_recall"] for x in group),
            "recall": statistics.mean(x["metrics"]["recall"] for x in group),
            "mrr": statistics.mean(x["metrics"]["reciprocal_rank"] for x in group),
            "complete": statistics.mean(not x["metrics"]["missing_units"] for x in group),
            "false_rejections": sum(not x["accepted"] for x in group),
        }

    controls = []
    controls.append(
        {
            "id": "conflict",
            "passed": r.search("retention", category=Category.EXPORT, snapshot="conflict").status
            == "policy_conflict",
        }
    )
    faq = next(c for c in r.index.chunks if c.kind == DocKind.FAQ)
    try:
        verify_citation(cite(r, faq.chunk_id), data, r)
        rejected = False
    except ValueError:
        rejected = True
    controls.append({"id": "FAQ_authority", "passed": rejected})
    durations = sorted(x["seconds"] for x in rows)
    return {
        "scope": "PUBLIC_ENGINEERING_ONLY",
        "corpus_builder_version": CORPUS_VERSION,
        "corpus_builder_sha256": hashlib.sha256(
            Path(__file__).with_name("structural_corpus.py").read_bytes()
        ).hexdigest(),
        "split": split,
        "method": method,
        "data_sha256": HASHES[split],
        "source_sha256": runtime_source_fingerprint(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_identity": encoder.identity if encoder else None,
        "aggregate": aggregate(positives),
        "languages": {
            name: aggregate([x for x in positives if x["language"] in languages])
            for name, languages in {
                "English": {"en", "paraphrase", "long", "multi"},
                "Chinese": {"zh"},
                "mixed": {"mixed"},
                "multi_evidence": {"mixed", "multi", "long"},
            }.items()
        },
        "negative": {
            "denominator": len(negatives),
            "false_acceptances": sum(x["accepted"] for x in negatives),
        },
        "controls": controls,
        "startup_s": startup,
        "query_p95_s": durations[math.ceil(0.95 * len(durations)) - 1],
        "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "rows": rows,
        "formal_executions": 0,
        "external_inference_API_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "challenge"))
    parser.add_argument("method", choices=("A1", "D", "H", "S"))
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    if args.split == "challenge":
        assert args.decision is not None
        lock = json.loads(args.decision.read_text())
        assert lock["scope"] == "FROZEN_PUBLIC_STRUCTURAL_DEV"
        assert lock["helper_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert lock["source_sha256"] == runtime_source_fingerprint()
        assert (
            lock["corpus_builder_sha256"]
            == hashlib.sha256(
                Path(__file__).with_name("structural_corpus.py").read_bytes()
            ).hexdigest()
        )
        assert lock["data_sha256"] == HASHES
    result = evaluate(args.split, args.method, args.model_directory)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(
        json.dumps(
            {k: result[k] for k in ("method", "aggregate", "languages", "negative", "controls")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
