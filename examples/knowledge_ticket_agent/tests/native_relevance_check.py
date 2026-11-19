"""Explicit native offline E5/pair and actual Mock tool integration; never default CI."""

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import knowledge_ticket_agent.business_demo as demo
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.agent_grading import grade_run
from knowledge_ticket_agent.agent_mock import scripted_calls
from knowledge_ticket_agent.agent_tools import PolicyEvidenceSession
from knowledge_ticket_agent.business_evaluation import public_manifest
from knowledge_ticket_agent.evaluation_runtime import execute_public_task
from knowledge_ticket_agent.pair_model import BgePairScorer
from knowledge_ticket_agent.relevance_evidence import (
    method_identity,
    select_sources,
    verify_business_sources,
)
from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.semantic_model import E5Encoder
from knowledge_ticket_agent.workflow_controller import WorkflowController

from agent_harness_lab.context import ContextPolicy
from agent_harness_lab.contracts import ToolResult
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.responses import ResponsesProvider
from agent_harness_lab.runtime import Agent, RunLimits

from .evidence_contract_helpers import development_corpus
from .relevance_corpus import build
from .test_workflow_controller import Stream, Tokens
from .vnext_public_evaluation import runtime_source_fingerprint


def run(model: Path, pair_model: Path) -> dict[str, Any]:
    data = load_dataset()
    encoder, pair = E5Encoder(model), BgePairScorer(pair_model)

    def factory(source: Dataset) -> RelevanceRetriever:
        return RelevanceRetriever(source, encoder, pair=pair)

    flows = []
    queries = (
        ("en", "Please process this first purchase credit policy request."),
        ("zh", "请按首次购买的购买额度调整政策处理这张模拟工单。"),
        ("mixed", "请依据 first purchase credit 处理模拟工单。"),
    )
    for language, query in queries:
        for approval in ("approve", "refuse"):
            started = time.perf_counter()
            result = demo.run_request(
                data, "KT-D01", query, approval=approval, retriever=factory(data)
            )
            report = demo.summarize(result, "native-" + language)
            assert report["verification"] == "PASS" and report["commits"] == int(
                approval == "approve"
            ), report
            methods = [
                json.loads(r.result.content)["result"]["method"]
                for r in result.app.registry.records
                if r.call.name == "kb_search" and r.result.ok
            ]
            assert methods and set(methods) == {report["retrieval_method"]}
            assert report["evidence_method"] == method_identity(result.app.retriever)
            assert report["evidence_retrieval_method"] == report["retrieval_method"]
            flows.append(
                {
                    "language": language,
                    "approval": approval,
                    "receipt": report,
                    "wall_s": time.perf_counter() - started,
                }
            )
    compaction = []
    for approval in ("approve", "refuse"):
        with patch.object(demo, "ContextPolicy", lambda *a: ContextPolicy(30000, 9000, 3, 96)):
            started = time.perf_counter()
            result = demo.run_request(
                data, "KT-D01", queries[1][1], approval=approval, retriever=factory(data)
            )
        report = demo.summarize(result, "native-compaction")
        assert any(s.compacted_results > 0 for s in result.result.context_stats)
        assert report["verification"] == "PASS" and report["commits"] == int(approval == "approve")
        compaction.append(report)
    controllers = []

    class ChineseStream(Stream):
        def events(self, *args: Any, **kwargs: Any) -> Any:
            for event in super().events(*args, **kwargs):
                for item in event["response"]["output"]:
                    if item.get("name") == "policy_search":
                        item["arguments"] = json.dumps({"query": queries[1][1]}, ensure_ascii=False)
                yield event

    for approval in ("approve", "refuse", "invalid", "missing"):
        provider = ResponsesProvider(
            Tokens(), "offline", transport=ChineseStream(), max_requests=12
        )
        controller = WorkflowController(
            data,
            "KT-D01",
            provider,
            approval=approval,
            request_text=queries[1][1],
            retriever=factory(data),
        )
        started = time.perf_counter()
        report = controller.run()
        report["integration_wall_s"] = time.perf_counter() - started
        assert report["commits"] == int(approval == "approve"), report
        assert len({s.result.method for s in controller.searches}) == 1
        assert report["evidence_method"] == method_identity(factory(data))
        assert report["evidence_retrieval_method"] == factory(data).method_identity
        assert report["evidence_basis"] == "frozen_m3_documentary"
        controllers.append(report)
    cases = public_manifest(data).cases
    business = []
    for case in cases[:13]:
        result = execute_public_task(data, case.task, retriever_factory=factory)
        grade = grade_run(result)
        state = result.app.workflow.store.read(case.task.ticket_id)
        assessments = [
            json.loads(r.result.content)
            for r in result.app.registry.records
            if r.call.name == "ticket_assess" and r.result.ok
        ]
        methods = [
            json.loads(r.result.content)["result"]["method"]
            for r in result.app.registry.records
            if r.call.name == "kb_search" and r.result.ok
        ]
        correct = (
            not grade.issues
            and state.status.value == case.oracle.final_status
            and state.revision == case.oracle.final_revision
            and assessments[-1]["decision"] == case.oracle.decision
            and len(set(methods)) == 1
        )
        assert correct, (case.case_id, grade)
        business.append({"id": case.case_id, "correct": correct, "commits": grade.commits})
    faults = []
    fault_cases = list(cases[13:]) + [
        replace(cases[0], task=replace(cases[0].task, fault=fault))
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
            grade = grade_run(execute_public_task(data, case.task, retriever_factory=factory))
            passed = grade.commits == int(case.task.fault in {"readback", "invalid_provenance"})
            if case.task.fault == "invalid_provenance":
                passed = passed and bool(grade.issues)
            if case.task.fault == "readback":
                passed = passed and not grade.readback_verified
        except (OSError, ValueError) as exc:
            passed = case.task.fault == "not_run" and str(exc) == "declared_not_run"
        assert passed, case.task.fault
        faults.append({"id": case.case_id, "fault": case.task.fault, "passed": passed})
    before = pair.windows
    value = pair.score(
        (("request " * 400 + "Who approves?", "An inspection needs a curator. " * 100),)
    )
    assert pair.windows > before + 1 and value == pair.score(
        (("request " * 400 + "Who approves?", "An inspection needs a curator. " * 100),)
    )
    # Real native models and actual Core tools on new source assertions. The
    # requests/corpus are public inputs; no Oracle is read by this application.
    qa_data, _ = build(data, development_corpus(), replace_sections=True)
    domain = []
    qa_queries = (
        ("en", "Please list capacity guide and closure dossier for module 0."),
        ("zh", "请说明模块1的配额条件以及交接资料。"),
        ("mixed", "请列出 module 2 的 capacity guide 与 closure dossier。"),
    )
    for language, query in qa_queries:
        started = time.perf_counter()
        r = factory(qa_data)
        selection = select_sources(qa_data, r, query, category=Category.OTHER)
        session = PolicyEvidenceSession(qa_data, r, query, category=Category.OTHER)
        calls = [scripted_calls("domain-search", (("kb_search", {"query": query}),))]
        calls.append(
            scripted_calls(
                "domain-gets",
                tuple(("kb_get", {"chunk_id": i}) for i in selection.read_dependencies),
            )
        )
        calls.extend(
            (scripted_calls("domain-proof", (("evidence_verify", {}),)), ModelResponse("DONE"))
        )
        run = Agent(MockProvider(calls), session.registry, limits=RunLimits(6, 32)).run(query)
        proofs = [
            json.loads(m.content)
            for m in run.messages
            if isinstance(m, ToolResult) and m.name == "evidence_verify" and m.ok
        ]
        assert proofs and proofs[-1]["status"] == "SUPPORTED"
        assert proofs[-1]["method"] == method_identity(r)
        assert proofs[-1]["basis"] == "source_assertion"
        assert proofs[-1]["retrieval_method"] == r.method_identity
        assert set(session.registry.names) == {"kb_search", "kb_get", "evidence_verify"}
        try:
            verify_business_sources(
                qa_data, r, tuple(session.reads.values()), (), category=Category.OTHER
            )
        except ValueError:
            pass  # New QA assertions cannot impersonate the frozen M3 contract.
        else:
            raise AssertionError("non-M3 source accepted as business truth")
        domain.append(
            {
                "language": language,
                "status": proofs[-1]["status"],
                "method": proofs[-1]["method"],
                "selection_method": selection.result.method,
                "full_reads": len(session.reads),
                "write_tools": 0,
                "wall_s": time.perf_counter() - started,
            }
        )
    return {
        "scope": "NATIVE_OFFLINE_PUBLIC_ENGINEERING",
        "python": platform.python_version(),
        "source_sha256": runtime_source_fingerprint(),
        "method_identity": factory(data).method_identity,
        "identity_components": factory(data).identity_components,
        "bilingual_flows": flows,
        "compaction": compaction,
        "controllers": controllers,
        "business": business,
        "faults": faults,
        "long_pair_windows": pair.windows - before,
        "domain_evidence_tools": domain,
        "paid_model_calls": 0,
        "formal_runs": 0,
        "decision_provider": "DETERMINISTIC_MOCK_AND_SYNTHETIC_RESPONSES_TRANSPORT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--pair-model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = run(args.model_directory, args.pair_model_directory)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "python": result["python"],
                "business": len(result["business"]),
                "faults": len(result["faults"]),
                "bilingual": len(result["bilingual_flows"]),
                "controllers": len(result["controllers"]),
                "compaction": len(result["compaction"]),
            }
        )
    )


if __name__ == "__main__":
    main()
