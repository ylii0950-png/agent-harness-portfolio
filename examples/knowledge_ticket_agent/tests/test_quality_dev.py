"""Independent public expectations, fixed comparison and evidence provenance attacks."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.agent_tools import encoded
from knowledge_ticket_agent.business_evaluation import public_manifest
from knowledge_ticket_agent.evidence import covered, score
from knowledge_ticket_agent.quality_evidence import AnswerSpec, acquire, assess_parent_answer
from knowledge_ticket_agent.quality_retrieval import QualityRetriever
from knowledge_ticket_agent.retrieval import Retriever, SearchResult
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.rules import evaluate_rules
from knowledge_ticket_agent.schemas import BusinessDecision, Category, Dataset

from .quality_public_oracle import PUBLIC_QUALITY_CASES, PublicQualityCase


@pytest.fixture(scope="module")
def data() -> Dataset:
    return load_dataset()


def test_fixed_candidates_and_public_m5p_units(data: Dataset) -> None:
    manifest = public_manifest(data)
    methods = {
        "BM25": Retriever(data),
        "Keyword": Retriever(data),
        "A1": StrategyRetriever(data, config=strategy("A1")),
        "Q1": QualityRetriever(data, candidate="Q1"),
        "Q2": QualityRetriever(data, candidate="Q2"),
    }
    rows: dict[str, Any] = {}
    for name, retriever in methods.items():
        cases: list[dict[str, Any]] = []
        for case in manifest.cases[:12]:
            result = retriever.search(
                case.task.query,
                category=Category(case.oracle.category),
                snapshot=case.task.snapshot,
                method="keyword" if name == "Keyword" else "bm25",
            )
            metric = score(case.retrieval_units, tuple(h.chunk_id for h in result.hits))
            cases.append(
                {
                    "case_id": case.case_id,
                    "metrics": asdict(metric),
                    "top3": [asdict(h) for h in result.hits],
                    "diversity": len({h.doc_ref for h in result.hits}),
                }
            )
        rows[name] = {
            "cases": cases,
            "summary": {
                field: sum(c["metrics"][field] for c in cases) / 12
                for field in ("hit", "recall", "reciprocal_rank", "raw_chunk_recall")
            },
        }
    assert rows["BM25"]["summary"]["recall"] == pytest.approx(41 / 72)
    assert rows["A1"]["summary"]["recall"] == pytest.approx(17 / 24)
    assert rows["Q1"]["summary"]["recall"] == rows["Q2"]["summary"]["recall"]
    assert rows["Q2"]["summary"]["recall"] > rows["BM25"]["summary"]["recall"]
    # A real regression must remain visible; no assertion of universal dominance.
    assert rows["Q2"]["summary"]["reciprocal_rank"] < rows["A1"]["summary"]["reciprocal_rank"]
    _save("initial", rows)


def _save(key: str, value: object) -> None:
    target = os.getenv("M2Q_PUBLIC_REPORT")
    if target:
        p = Path(target)
        old = json.loads(p.read_text()) if p.exists() else {}
        old[key] = value
        p.write_text(json.dumps(old, indent=2) + "\n")


def test_public_multistep_measurement(data: Dataset) -> None:
    rows: list[dict[str, Any]] = []
    for case in public_manifest(data).cases[:12]:
        variants: dict[str, Any] = {}
        for label, ret, expansion in (
            ("BM25_initial_only", Retriever(data), False),
            ("BM25_catalog_expansion", Retriever(data), True),
            ("Q2_catalog_expansion", QualityRetriever(data, candidate="Q2"), True),
        ):
            run = acquire(
                data,
                case.task.ticket_id,
                case.task.query,
                retriever=ret,
                snapshot=case.task.snapshot,
                expand=expansion,
            )
            obtained = frozenset(run.app.parent_chunks())
            required = frozenset(run.app.required_chunks())
            rec = sum(covered(u, obtained) for u in case.oracle.required_units) / len(
                case.oracle.required_units
            )
            records = run.app.registry.records
            variants[label] = {
                "stop_reason": run.result.stop_reason.value,
                "models": run.result.model_rounds,
                "tools": run.result.tool_calls,
                "search_calls": sum(r.call.name == "kb_search" for r in records),
                "full_get_calls": sum(r.call.name == "kb_get" for r in records),
                "REC_public_units": rec,
                "M3_catalog_chunk_coverage": len(obtained & required) / len(required),
                "missing_independent_catalog_chunks": sorted(required - obtained),
                "actual_authoritative_sections": sorted(
                    {(c.doc_ref, c.section_id) for c in ret.index.chunks if c.chunk_id in obtained}
                ),
                "commits": len(run.app.workflow.store.snapshot.commits),
                "parent_only": True,
                "backend_search_calls": len(run.app.backend),
            }
            assert run.result.model_rounds <= 12 and run.result.tool_calls <= 28
            assert (
                variants[label]["commits"] == 0
                and run.app.initial == run.app.workflow.store.snapshot
            )
        rows.append({"case_id": case.case_id, "variants": variants})
    _save("acquisition", rows)
    # Public expectations, not an assumption that heading expansion can never miss.
    assert sum(r["variants"]["Q2_catalog_expansion"]["REC_public_units"] for r in rows) >= sum(
        r["variants"]["BM25_initial_only"]["REC_public_units"] for r in rows
    )


@pytest.mark.parametrize("case", PUBLIC_QUALITY_CASES, ids=lambda c: c.case_id)
def test_independent_public_answerability(data: Dataset, case: PublicQualityCase) -> None:
    representative = {
        Category.BILLING: "KT-D01",
        Category.ACCESS: "KT-D08",
        Category.EXPORT: "KT-D10",
    }

    class WithheldSource(Retriever):
        def search(
            self,
            query: str,
            *,
            category: Category | None = None,
            snapshot: str = "normal",
            top_k: int = 3,
            method: str = "bm25",
        ) -> SearchResult:
            result = super().search(
                query, category=category, snapshot=snapshot, top_k=5, method=method
            )
            allowed = (
                ()
                if case.source_filter == "expired_only"
                else tuple(h for h in result.hits if h.doc_ref.startswith("FAQ-"))
            )
            return replace(result, status="ok" if allowed else "no_match", hits=allowed[:top_k])

    retriever = (
        WithheldSource(data)
        if case.source_filter != "all"
        else QualityRetriever(data, candidate="Q2")
    )
    run = acquire(
        data,
        representative[case.category],
        case.query,
        retriever=retriever,
        expand=case.source_filter == "all",
    )
    initial = acquire(
        data,
        representative[case.category],
        case.query,
        retriever=WithheldSource(data) if case.source_filter != "all" else Retriever(data),
        expand=False,
    )
    for observation_run in (initial,):
        if case.evidence_origin == "child":
            observation_run.app.registry.records = [
                replace(r, depth=1) for r in observation_run.app.registry.records
            ]
            observation_run.app.trace.rows = [
                (e, {**f, "depth": 1}) for e, f in observation_run.app.trace.rows
            ]
        elif case.evidence_origin == "backend":
            observation_run.app.registry.records = []
            observation_run.app.trace.rows = []
    before = assess_parent_answer(
        initial.app, AnswerSpec(case.category, case.intent, case.required_sections)
    )
    if case.evidence_origin == "child":
        # A coherent original Parent run exists, but only Child gets are offered as evidence.
        # They must not receive Parent credit, even if their canonical source is correct.
        run.app.registry.records = [replace(r, depth=1) for r in run.app.registry.records]
        run.app.trace.rows = [(e, {**f, "depth": 1}) for e, f in run.app.trace.rows]
    elif case.evidence_origin == "backend":
        run.app.registry.records = []
        run.app.trace.rows = []
    result = assess_parent_answer(
        run.app, AnswerSpec(case.category, case.intent, case.required_sections)
    )
    assert result.status == case.expected_status, (case.case_id, result)
    assert not run.app.workflow.store.snapshot.commits
    if result.status == "answerable":
        assert result.supported_citation_ids and not result.missing_sections
    fact_decision = None
    if case.trusted_data_variant != "unchanged":
        assert case.trusted_ticket_id is not None
        tickets = tuple(
            replace(
                t,
                facts=replace(
                    t.facts,
                    **(
                        {"invoice_id": None}
                        if case.trusted_data_variant == "missing_invoice"
                        else {"trusted_identity_ref": "UNKNOWN-PUBLIC-IDENTITY"}
                    ),
                ),
            )
            if t.ticket_id == case.trusted_ticket_id
            else t
            for t in data.tickets
        )
        factual = evaluate_rules(replace(data, tickets=tickets), case.trusted_ticket_id)
        assert factual.decision is BusinessDecision.LOAD_ERROR
        fact_decision = asdict(factual)
    _save(
        case.case_id,
        {
            "expected": case.expected_status,
            "before_same_gate_initial_BM25": asdict(before),
            "actual": asdict(result),
            "reason": case.reason,
            "evidence_origin": case.evidence_origin,
            "source_filter": case.source_filter,
            "actual_M3_missing_fact_decision": fact_decision,
        },
    )


@pytest.mark.parametrize("mutation", ["args", "order", "trace", "citation", "root", "missing"])
def test_tampered_observation_never_answers(data: Dataset, mutation: str) -> None:
    run = acquire(
        data,
        "KT-D01",
        "first purchase credit restrictions",
        retriever=QualityRetriever(data, candidate="Q2"),
        expand=True,
    )
    spec = AnswerSpec(
        Category.BILLING,
        "policy_explanation",
        (("POL-BILL-01@1.0", "eligibility"), ("POL-BILL-02@1.0", "restrictions")),
    )
    assert assess_parent_answer(run.app, spec).status == "answerable"
    rows = run.app.registry.records
    index = next(i for i, r in enumerate(rows) if r.call.name == "kb_get")
    row = rows[index]
    if mutation == "args":
        rows[index] = replace(
            row,
            call=replace(row.call, arguments=encoded({"chunk_id": "stale", "doc_ref": "stale"})),
        )
    elif mutation == "order":
        rows.reverse()
    elif mutation == "trace":
        run.app.trace.rows = []
    elif mutation == "citation":
        rows[index] = replace(row, citation=None)
    elif mutation == "root":
        rows[index] = replace(row, run_id="other-root")
    else:
        del rows[index]
    assert assess_parent_answer(run.app, spec).status != "answerable"


def test_full_source_coverage_does_not_expose_mutation_or_expired_get(data: Dataset) -> None:
    from knowledge_ticket_agent.agent_mock import scripted_calls, source_script
    from knowledge_ticket_agent.agent_tools import DemoApplication

    from agent_harness_lab.providers import MockProvider
    from agent_harness_lab.runtime import Agent, RunLimits

    app = DemoApplication(data, "KT-D01")
    searches, gets = source_script(app)
    expired = next(c for c in app.retriever.index.chunks if c.doc_ref == "POL-BILL-01@0.9")
    provider = MockProvider(
        (
            scripted_calls("search", searches),
            scripted_calls("get", gets),
            scripted_calls(
                "forbidden",
                (
                    ("ticket_transition", {"ticket_id": "KT-D01", "destination": "deleted"}),
                    ("kb_get", {"chunk_id": expired.chunk_id, "doc_ref": expired.doc_ref}),
                ),
            ),
            "blocked",
        )
    )
    with app.registry.restrict_tools(("kb_search", "kb_get")):
        Agent(provider, app.registry, limits=RunLimits(5, 28), trace=app.trace).run(
            "New public forbidden action probe"
        )
    assert set(app.required_chunks()) <= set(app.parent_chunks())
    rejected = app.registry.records[-2:]
    assert all(not r.result.ok and r.handler_entries == 0 for r in rejected[:1])
    assert not rejected[-1].result.ok
    assert not app.workflow.store.snapshot.commits and app.initial == app.workflow.store.snapshot
