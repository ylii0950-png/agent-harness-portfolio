"""Fictional business CLI: scripted Mock default, explicit supervised real mode."""

import argparse
import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from agent_harness_lab.context import ContextPolicy
from agent_harness_lab.contracts import ToolDefinition
from agent_harness_lab.execution import TotalLimits
from agent_harness_lab.providers import ConversationItem, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.agent_grading import _events, grade_run
from knowledge_ticket_agent.agent_mock import MockRun, scripted_calls
from knowledge_ticket_agent.agent_tools import DemoApplication, verified_parent_citations
from knowledge_ticket_agent.approvals import documentary_requirements
from knowledge_ticket_agent.bilingual import (
    BilingualRetriever,
    RequestContext,
    contextualize,
    error_message,
    output_language,
    presentation,
)
from knowledge_ticket_agent.evaluation_runtime import FeedbackMockProvider, ProviderInput
from knowledge_ticket_agent.quality_evidence import (
    AnswerDecision,
    AnswerSpec,
    EvidenceNavigator,
    assess_parent_answer,
    policy_navigation,
)
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.retrieval_profiles import A1Retriever as _BusinessRetriever
from knowledge_ticket_agent.schemas import Dataset, Split
from knowledge_ticket_agent.strict import require


class BusinessProvider(FeedbackMockProvider):
    """Existing Todo/assessment/ASK/write/readback flow with shared evidence navigation."""

    def __init__(
        self,
        app: DemoApplication,
        query: str,
        followups: int,
        *,
        request: RequestContext | None = None,
    ) -> None:
        super().__init__(ProviderInput(app.ticket_id, ()))
        self.app = app
        self.request = request
        from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever

        prospective = isinstance(app.retriever, RelevanceRetriever)
        self.navigator = EvidenceNavigator(
            query,
            policy_navigation(app.data, app.category, app.snapshot)[:followups],
            # Trusted M3 documentary prerequisites, never evaluation Oracle units.
            allowed=frozenset(app.required_chunks()) if prospective else None,
            batch_followups=3 if prospective else 1,
        )
        self.documentary: AnswerDecision | None = None

    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        # Cache only actual full results before any Core context compaction.
        for row in self._payloads(messages, "kb_get"):
            c = row.get("citation")
            if type(c) is dict and type(c.get("chunk_id")) is str:
                self._citations.add(c["chunk_id"])
        if self.request is not None and self.request.status != "supported":
            if self._phase == "sources":
                self._phase = "request_guard"
                MockProvider.complete(self, messages, tools=tools)
                return scripted_calls(
                    "guard",
                    (
                        (
                            "todo_write",
                            {"operation": "update", "task_id": "evidence", "status": "completed"},
                        ),
                        (
                            "todo_write",
                            {"operation": "update", "task_id": "outcome", "status": "in_progress"},
                        ),
                    ),
                )
            if self._phase == "request_guard":
                self._phase = "final"
                MockProvider.complete(self, messages, tools=tools)
                return self._read()
            return super().complete(messages, tools=tools)
        if self._phase in {"sources", "get"}:
            response = self.navigator.step(messages)
            if response.tool_calls:
                MockProvider.complete(self, messages, tools=tools)
                return response
            # Acquired results already came through actual Registry kb_gets.
            # Reuse the existing flow's assessment phase without re-fetching them.
            self._phase = "get"
            self._get_cursor = 64
        if self._phase == "assessed":
            requirements = documentary_requirements(
                self.app.data, self.app.category, self.app.snapshot
            )
            spec = AnswerSpec(
                self.app.category,
                "policy_explanation",
                tuple((c.doc_ref, c.section_id) for r in requirements for c in r.components),
                self.app.snapshot,
            )
            self.documentary = assess_parent_answer(self.app, spec)
            if self.documentary.status != "answerable":
                MockProvider.complete(self, messages, tools=tools)
                self._phase = "final"
                return self._read()
        return super().complete(messages, tools=tools)


def run_ticket(
    data: Dataset,
    ticket_id: str,
    *,
    approval: str = "refuse",
    followups: int = 6,
    snapshot: str = "normal",
    actor_id: str = "SYN-STAFF-02",
    request: RequestContext | None = None,
    retriever: Retriever | None = None,
) -> MockRun:
    """Fresh in-memory run, trusted Mock approval fixture, unchanged exact WRITE/ASK."""
    require(approval in {"approve", "refuse"}, "demo_approval")
    require(type(followups) is int and 0 <= followups <= 8, "demo_followup_limit")
    tickets = tuple(t for t in data.tickets if t.ticket_id == ticket_id and t.split is Split.DEV)
    require(len(tickets) == 1, "demo_dev_ticket")
    app = DemoApplication(data, ticket_id, approval=approval, snapshot=snapshot, actor_id=actor_id)
    if retriever is not None:
        require(retriever.index == app.retriever.index, "business_retrieval_corpus")
        app.retriever = retriever
    else:
        app.retriever = (
            BilingualRetriever(data) if request is not None else _BusinessRetriever(data)
        )
    provider = BusinessProvider(app, tickets[0].body, followups, request=request)
    try:
        guard = (
            app.registry.restrict_tools(
                ("ticket_read", "todo_write", "todo_read", "kb_search", "kb_get")
            )
            if request is not None and request.status != "supported"
            else nullcontext()
        )
        with guard:
            return _run_business_agent(app, provider, ticket_id)
    finally:
        app.close()


def _run_business_agent(
    app: DemoApplication, provider: BusinessProvider, ticket_id: str
) -> MockRun:
    result = Agent(
        provider,
        app.registry,
        limits=RunLimits(12, 28),
        total_limits=TotalLimits(16, 28, 1),
        trace=app.trace,
        context_policy=ContextPolicy(160000, 120000, 3, 96),
    ).run(f"Handle fictional ticket {ticket_id}. Never infer approval or external facts.")
    return MockRun(app, result, provider, ())


def run_request(
    data: Dataset,
    ticket_id: str,
    text: str,
    *,
    language: str = "auto",
    approval: str = "refuse",
    followups: int = 6,
    snapshot: str = "normal",
    actor_id: str = "SYN-STAFF-02",
    retriever: Retriever | None = None,
) -> MockRun:
    adapted, request = contextualize(data, ticket_id, text, language)
    return run_ticket(
        adapted,
        ticket_id,
        approval=approval,
        followups=followups,
        snapshot=snapshot,
        actor_id=actor_id,
        request=request,
        retriever=retriever,
    )


def summarize(run: MockRun, label: str) -> dict[str, Any]:
    grade = grade_run(run)
    provider = run.provider
    assert isinstance(provider, BusinessProvider)
    citations = verified_parent_citations(run.app)
    assessment = provider._payloads(run.result.messages, "ticket_assess")
    approval = "not_requested"
    for row in run.app.registry.records:
        if row.call.name == "ticket_transition":
            events = _events(run.app, row, "tool.call")
            if events:
                approval = str(events[0][1].get("permission_reason"))
    initial = run.app.initial
    assert initial is not None
    before = next(t for t in initial.tickets if t.ticket_id == run.app.ticket_id)
    after = run.app.workflow.store.read(run.app.ticket_id)
    report: dict[str, Any] = {
        "scenario": label,
        "execution": "DETERMINISTIC_MOCK",
        "ticket": run.app.ticket_id,
        "decision": assessment[-1].get("decision") if assessment else "unavailable",
        "reason": approval
        if grade.outcome == "permission_denial"
        else (assessment[-1].get("reason") if assessment else run.result.stop_reason.value),
        "outcome": grade.outcome,
        "issues": list(grade.issues),
        "evidence": asdict(provider.documentary) if provider.documentary else None,
        "citations": sorted({f"{c.doc_ref}#{c.section_id}" for c in citations}),
        "activity": {
            "initial_retriever": getattr(run.app.retriever, "method_identity", "A1"),
            "search_calls": grade.agent_search_calls,
            "followup_searches": max(0, grade.agent_search_calls - 1),
            "full_get_calls": grade.agent_get_calls,
            "models": grade.parent_model_calls,
            "tools": grade.parent_tool_calls,
            "parent_chunks": grade.agent_full_source_chunks,
            "required_chunks": grade.required_policy_chunks,
            "child_models": grade.child_model_calls,
            "backend_search_calls": grade.backend_search_calls,
        },
        "approval": approval,
        "handler_entries": grade.handler_entries,
        "before": asdict(before),
        "after": asdict(after),
        "commits": grade.commits,
        "postwrite_readback_verified": grade.readback_verified,
        "unchanged_state_verified": initial == run.app.workflow.store.snapshot,
        "verification": "PASS"
        if not grade.issues
        and grade.outcome
        in {"business_success", "expected_denial", "permission_denial", "read_only_success"}
        else "FAILED",
    }

    if provider.request is not None:
        if assessment:
            from knowledge_ticket_agent.rules import evaluate_rules

            evaluated = evaluate_rules(run.app.data, run.app.ticket_id, snapshot=run.app.snapshot)
            if evaluated.decision.value == report["decision"]:
                report["rule_reasons"] = list(evaluated.reasons)
        report["request"] = asdict(provider.request)
        report["application_decision"] = provider.request.status
        report["display"] = presentation(report, provider.request.language)
        if provider.request.status != "supported":
            report["display"]["message"] = error_message(
                provider.request.status, provider.request.language
            )
    searches = provider._payloads(run.result.messages, "kb_search")
    if searches:
        first = searches[0].get("result")
        if type(first) is dict:
            report["retrieval_method"] = first.get("method")
    from knowledge_ticket_agent.relevance_evidence import method_audit

    report.update(method_audit(run.app.retriever, run.app.snapshot))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fictional ticket Agent: Mock default; explicit supervised OAuth mode"
    )
    parser.add_argument(
        "--scenario",
        choices=("all", "valid", "insufficient", "denied", "conflict", "missing-facts"),
        default="all",
    )
    parser.add_argument("--ticket", help="Run one public DEV ticket instead of the named scenarios")
    parser.add_argument(
        "--approval",
        choices=("approve", "refuse"),
        default=None,
        help="Trusted simulated approval fixture (custom ticket default: refuse)",
    )
    parser.add_argument(
        "--followups", type=int, default=None, help="Override targeted follow-up cap,0..8"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Mock observations or canonical supervised application receipt",
    )
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--snapshot", choices=("normal", "conflict"), default=None)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--subscription-only", action="store_true")
    parser.add_argument(
        "--request-text", help="Fictional customer request; trusted --ticket context required"
    )
    parser.add_argument("--language", choices=("auto", "en", "zh-CN"), default="auto")
    parser.add_argument("--actor", default="SYN-STAFF-02", help="Trusted fictional actor context")
    parser.add_argument("--retrieval-profile", choices=("a1", "vnext"), default="a1")
    parser.add_argument("--retrieval-variant", choices=("D", "H", "S", "R0", "R1"))
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--pair-model-directory", type=Path)
    args = parser.parse_args(argv)
    variant = args.retrieval_variant or "H"
    if args.retrieval_profile == "a1" and any(
        value is not None
        for value in (args.retrieval_variant, args.model_directory, args.pair_model_directory)
    ):
        parser.error("semantic_options_require_vnext_profile")
    if args.pair_model_directory is not None and variant != "R1":
        parser.error("pair_model_directory_requires_R1")
    encoder = None
    if args.retrieval_profile == "vnext":
        if args.mode != "mock" or args.model_directory is None:
            parser.error("vnext_requires_offline_mock_and_model_directory")
        from knowledge_ticket_agent.semantic_model import E5Encoder

        encoder = E5Encoder(args.model_directory)
    pair = None
    if args.retrieval_profile == "vnext" and variant == "R1":
        if args.pair_model_directory is None:
            parser.error("R1_requires_explicit_pair_model_directory")
        from knowledge_ticket_agent.pair_model import BgePairScorer

        pair = BgePairScorer(args.pair_model_directory)
    localized = args.request_text is not None or args.language != "auto"
    if args.request_text is not None and not args.ticket:
        parser.error(
            error_message("demo_dev_ticket", output_language(args.request_text, args.language))
        )
    if args.mode == "real":
        import os

        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            parser.error("real_mode_disabled_in_ci")
        if not args.ticket or not args.subscription_only:
            parser.error("real_requires_ticket_and_subscription_confirmation")
        if args.scenario != "all" or args.followups is not None or args.snapshot is not None:
            parser.error("scenario_and_followups_are_mock_only")
        from knowledge_ticket_agent.workflow_live import real_cli

        if args.actor != "SYN-STAFF-02":
            parser.error("real_actor_override_not_supported")
        real_cli(
            args.ticket,
            args.model,
            args.approval or "refuse",
            structured=args.json,
            request_text=args.request_text,
            language=args.language,
        )
        return
    scenarios = {
        "valid": ("KT-D01", "approve", 6, "normal"),
        "insufficient": ("KT-D01", "approve", 0, "normal"),
        "denied": ("KT-D08", "refuse", 6, "normal"),
        "conflict": ("KT-D12", "approve", 8, "conflict"),
        "missing-facts": ("KT-D01", "approve", 6, "normal"),
    }
    selected = list(scenarios)[:3] if args.scenario == "all" else [args.scenario]
    if args.ticket:
        selected = ["custom"]
    data = load_dataset()
    reports = []
    for label in selected:
        ticket, approval, followups, snapshot = (
            (
                args.ticket,
                args.approval or "refuse",
                6 if args.followups is None else args.followups,
                "normal",
            )
            if label == "custom"
            else scenarios[label]
        )
        if args.approval is not None:
            approval = args.approval
        if args.followups is not None:
            followups = args.followups
        if args.snapshot is not None:
            snapshot = args.snapshot
        try:
            scenario_data = data
            if label == "missing-facts":
                # Public in-memory fault fixture; packaged facts and policies stay intact.
                scenario_data = replace(
                    data,
                    tickets=tuple(
                        replace(t, facts=replace(t.facts, invoice_id=None))
                        if t.ticket_id == ticket
                        else t
                        for t in data.tickets
                    ),
                )
            selected_retriever: Retriever | None = None
            if encoder is not None:
                from knowledge_ticket_agent.retrieval_vnext import VNextRetriever

                if variant in {"R0", "R1"}:
                    from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever

                    selected_retriever = RelevanceRetriever(
                        scenario_data, encoder, variant=variant, pair=pair
                    )
                elif variant == "S":
                    from knowledge_ticket_agent.retrieval_structural import StructuralRetriever

                    selected_retriever = StructuralRetriever(scenario_data, encoder)
                else:
                    selected_retriever = VNextRetriever(scenario_data, encoder, variant=variant)
            if localized:
                text = args.request_text
                if text is None:
                    target = next((t for t in scenario_data.tickets if t.ticket_id == ticket), None)
                    require(target is not None, "demo_dev_ticket")
                    assert target is not None
                    text = target.body
                run = run_request(
                    scenario_data,
                    ticket,
                    text,
                    language=args.language,
                    approval=approval,
                    followups=followups,
                    snapshot=snapshot,
                    actor_id=args.actor,
                    retriever=selected_retriever,
                )
            else:
                run = run_ticket(
                    scenario_data,
                    ticket,
                    approval=approval,
                    followups=followups,
                    snapshot=snapshot,
                    actor_id=args.actor,
                    retriever=selected_retriever,
                )
            report = summarize(run, label)
        except ValueError as error:
            locale = output_language(args.request_text or "", args.language)
            parser.error(error_message(str(error), locale) if localized else str(error))
        if args.json:
            report["trace"] = run.app.trace.rows
            report["tool_records"] = [asdict(r) for r in run.app.registry.records]
            report["business_audit"] = [asdict(e) for e in run.app.workflow.audit.events]
            reports.append(report)
        elif localized and report["request"]["language"] == "zh-CN":
            print(f"\n[{label}] {ticket} | 本地模拟 Agent")
            print("决策：" + report["display"]["decision"])
            print("依据：" + report["display"]["reason"])
            print("来源：" + ", ".join(report["citations"]))
            print(
                f"ASK: {report['approval']} | {report['before']['status']} "
                f"r{report['before']['revision']} -> {report['after']['status']} "
                f"r{report['after']['revision']}"
            )
            print(report["display"]["message"])
            print("核验：" + report["display"]["verification"])
        else:
            activity = report["activity"]
            assert isinstance(activity, dict)
            print(f"\n[{label}] {ticket} | DETERMINISTIC MOCK")
            print(
                f"Decision: {report['decision']} | Outcome: {report['outcome']} | "
                f"Reason: {report['reason']}"
            )
            print(
                f"{'A1' if args.retrieval_profile == 'a1' else 'vNext'} + targeted evidence: "
                f"{activity['search_calls']} searches "
                f"({activity['followup_searches']} follow-up), "
                f"{activity['full_get_calls']} full reads; "
                f"{activity['parent_chunks']}/{activity['required_chunks']} Parent chunks"
            )
            print("Sources: " + ", ".join(report["citations"]))
            print(
                f"ASK: {report['approval']} | State: {report['before']['status']} "
                f"r{report['before']['revision']} -> {report['after']['status']} "
                f"r{report['after']['revision']}"
            )
            checked = (
                "committed state read back"
                if report["postwrite_readback_verified"]
                else "state verified unchanged"
                if report["unchanged_state_verified"]
                else "state could not be verified"
            )
            print(f"Verification: {report['verification']} | {checked}")
        if report["verification"] != "PASS":
            if args.json:
                print(json.dumps(reports, indent=2, ensure_ascii=False))
            raise SystemExit(1)
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
