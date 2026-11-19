"""Bounded read-only acquisition and specified-explanation abstention.

Trusted AnswerSpec is application configuration, never inferred from a question
or model argument. It promises documentary coverage, not semantic entailment.
No business fact or execution/approval assertion is established by this adapter.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, cast

from agent_harness_lab.contracts import ToolDefinition, ToolResult
from agent_harness_lab.execution import TotalLimits
from agent_harness_lab.providers import ConversationItem, MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, RunLimits
from knowledge_ticket_agent import relevance_evidence
from knowledge_ticket_agent.agent_grading import _canonical_get, _events
from knowledge_ticket_agent.agent_mock import MockRun, scripted_calls
from knowledge_ticket_agent.agent_tools import SCHEMA, DemoApplication, call_reference
from knowledge_ticket_agent.approvals import documentary_requirements
from knowledge_ticket_agent.citations import Citation
from knowledge_ticket_agent.evidence_adapter import (
    EvidenceRequest,
    EvidenceRequirement,
    Outcome,
    SectionComponent,
    assess_evidence,
    scope_for,
)
from knowledge_ticket_agent.evidence_contract import Verdict
from knowledge_ticket_agent.retrieval import Catalog, Retriever, SearchHit, SearchResult
from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import FixtureError, decode, parse_json, require

ANSWER_VERSION = "lqd-specified-answer-1.0.0"


@dataclass(frozen=True)
class AnswerSpec:
    """Trusted finite explanation contract; not an arbitrary claim to be proved."""

    category: Category
    intent: str
    sections: tuple[tuple[str, str], ...]
    snapshot: str = "normal"


@dataclass(frozen=True)
class AnswerDecision:
    status: str
    reason: str
    covered_sections: tuple[tuple[str, str], ...] = ()
    missing_sections: tuple[tuple[str, str], ...] = ()
    supported_citation_ids: tuple[str, ...] = ()
    version: str = ANSWER_VERSION


def _search_result(raw: object) -> SearchResult:
    # The original general fixture decoder intentionally lacks float/fixed-pair
    # support. Parse this existing public tool schema locally, not by weakening it.
    require(type(raw) is dict, "search_result")
    value = cast(dict[str, Any], raw)
    require(set(value) == {f.name for f in fields(SearchResult)}, "search_result")
    require(type(value["hits"]) is list and len(value["hits"]) <= 5, "search_hits")
    hits = []
    for hit in value["hits"]:
        require(type(hit) is dict and set(hit) == {f.name for f in fields(SearchHit)}, "search_hit")
        hits.append(SearchHit(**hit))
    c = value["catalog"]
    require(type(c) is dict and set(c) == {f.name for f in fields(Catalog)}, "search_catalog")
    catalog = Catalog(
        **cast(
            dict[str, Any],
            {k: tuple(tuple(i) if type(i) is list else i for i in v) for k, v in c.items()},
        )
    )
    return SearchResult(
        value["status"],
        value["corpus_version"],
        value["index_sha256"],
        value["method"],
        tuple(hits),
        catalog,
    )


def assess_parent_answer(app: DemoApplication, spec: AnswerSpec) -> AnswerDecision:
    """No truth flags, query text, Gold, backend observations or Child credit."""
    if type(spec) is not AnswerSpec or spec.intent != "policy_explanation":
        return AnswerDecision("not_assessable", "personal_fact_permission_or_unknown_intent")
    try:
        require(
            spec.category is app.category
            and spec.snapshot == app.snapshot
            and type(spec.sections) is tuple
            and 0 < len(spec.sections) <= 16
            and len(set(spec.sections)) == len(spec.sections),
            "answer_scope",
        )
        scope = scope_for(app.data, category=app.category, snapshot=app.snapshot)
        selection = app.retriever.selection(category=app.category, snapshot=app.snapshot)
        require(
            all(
                ref in selection.authoritative_refs
                and any(
                    c.doc_ref == ref and c.section_id == section for c in app.retriever.index.chunks
                )
                for ref, section in spec.sections
            ),
            "answer_inapplicable_requirement",
        )
        root = app.workflow.run_id
        require(app.workflow.store.snapshot == app.initial, "read_only_effects")
        records = app.registry.records
        require([r.ordinal for r in records] == list(range(1, len(records) + 1)), "record_order")
        parent = [r for r in records if r.depth == 0]
        require(all(r.run_id == root for r in parent), "root_scope")
        expected_refs = [call_reference(r.call.call_id) for r in parent]
        for event in ("tool.call", "tool.result"):
            require(
                [f.get("call_ref") for e, f in app.trace.rows if e == event and f.get("depth") == 0]
                == expected_refs,
                "trace_order",
            )
        citations: dict[str, Citation] = {}
        for row in parent:
            calls, returns = _events(app, row, "tool.call"), _events(app, row, "tool.result")
            require(len(calls) == len(returns) == 1 and calls[0][0] < returns[0][0], "trace_pair")
            require(
                row.result.name == row.call.name
                and row.result.call_id == row.call.call_id
                and returns[0][1].get("success") is row.result.ok
                and type(row.handler_entries) is int
                and row.handler_entries == int(calls[0][1].get("authorized") is True),
                "handler_pair",
            )
            if row.call.name == "kb_get" and row.result.ok:
                canonical = _canonical_get(app, row)
                citations[canonical.chunk_id] = canonical
        if isinstance(app.retriever, RelevanceRetriever):
            proof = relevance_evidence.verify_business_sources(
                app.data,
                app.retriever,
                tuple(citations.values()),
                spec.sections,
                category=app.category,
                snapshot=app.snapshot,
            )
            if proof.status is Verdict.NOT_ASSESSABLE:
                return AnswerDecision("not_assessable", "unresolved_source_evidence")
        searched: set[str] = set()
        usable: dict[str, tuple[SearchResult, Citation]] = {}
        for row in app.registry.records:
            if row.depth != 0 or row.run_id != root:
                continue
            require(row.result.call_id == row.call.call_id, "call_pair")
            if row.call.name == "kb_search" and row.result.ok:
                raw = parse_json(row.result.content)
                require(type(raw) is dict and raw.get("schema") == SCHEMA, "search_output")
                assert isinstance(raw, dict)
                result = _search_result(raw["result"])
                for hit in result.hits:
                    searched.add(hit.chunk_id)
            elif row.call.name == "kb_get" and row.result.ok:
                require(row.citation is not None, "get_receipt")
                assert row.citation is not None
                identifier = row.citation.chunk_id
                require(identifier in searched and identifier in citations, "get_without_search")
                # Use only a genuine preceding actual result that contains this get.
                for previous in reversed(app.registry.records[: row.ordinal - 1]):
                    if previous.depth != 0 or previous.run_id != root:
                        continue
                    if previous.call.name == "kb_search" and previous.result.ok:
                        raw = parse_json(previous.result.content)
                        require(type(raw) is dict and raw.get("schema") == SCHEMA, "search_output")
                        assert isinstance(raw, dict)
                        result = _search_result(raw["result"])
                        matched_hit = next(
                            (h for h in result.hits if h.chunk_id == identifier), None
                        )
                        if matched_hit is not None:
                            usable[identifier] = (
                                replace(result, hits=(matched_hit,)),
                                citations[identifier],
                            )
                            break
        complete, missing = [], []
        supported: set[str] = set()
        selection = app.retriever.selection(category=app.category, snapshot=app.snapshot)
        if selection.conflicts:
            return AnswerDecision("not_assessable", "unresolved_authority_conflict")
        for n, (ref, section) in enumerate(spec.sections):
            requirement = EvidenceRequirement(
                f"answer-{n}",
                ANSWER_VERSION,
                scope,
                (SectionComponent(f"component-{n}", ref, section),),
            )
            needed = {
                c.chunk_id
                for c in app.retriever.index.chunks
                if c.doc_ref == ref and c.section_id == section
            }
            obtained: set[str] = set()
            for identifier in sorted(needed & usable.keys()):
                result, citation = usable[identifier]
                # M2G validates full canonical source, scope, authority and citation.
                request = EvidenceRequest(
                    requirement.requirement_id, requirement.version, scope, result, (citation,)
                )
                assessment = assess_evidence(app.data, (requirement,), request)
                if assessment.outcome is Outcome.NOT_ASSESSABLE:
                    return AnswerDecision("not_assessable", "invalid_source_or_requirement")
                obtained.update(c.chunk_id for c in assessment.citations)
            if needed and needed <= obtained and ref in selection.authoritative_refs:
                complete.append((ref, section))
                supported.update(obtained)
            else:
                missing.append((ref, section))
        return AnswerDecision(
            "insufficient" if missing else "answerable",
            "specified_sections_missing" if missing else "specified_explanation_only",
            tuple(complete),
            tuple(missing),
            tuple(sorted(supported)),
        )
    except (FixtureError, ValueError, TypeError, KeyError, AttributeError):
        return AnswerDecision("not_assessable", "invalid_parent_evidence_or_contract")


class EvidenceNavigator:
    """Finite feedback script: initial search, actual gets, missing-section navigation."""

    def __init__(
        self,
        query: str,
        navigation: tuple[tuple[tuple[str, ...], str], ...],
        *,
        allowed: frozenset[str] | None = None,
        batch_followups: int = 1,
    ) -> None:
        self.query, self.navigation = query, navigation
        require(type(batch_followups) is int and 1 <= batch_followups <= 3, "navigation_batch")
        self.batch_followups = batch_followups
        self.allowed = allowed
        self.started = False
        self.turn = 0
        self.searched: dict[str, str] = {}
        self.obtained: set[str] = set()
        self.attempted: set[str] = set()
        self.expanded: set[int] = set()

    def step(self, messages: Sequence[ConversationItem]) -> ModelResponse:
        self.turn += 1
        if not self.started:
            self.started = True
            return scripted_calls("initial", (("kb_search", {"query": self.query}),))
        for item in messages:
            if type(item) is not ToolResult or not item.ok:
                continue
            if item.content.startswith("[Compacted low-trust"):
                # A preview is not a new observation. Retain only previously
                # observed genuine results; never parse/promote the summary.
                continue
            value = parse_json(item.content)
            if type(value) is not dict:
                continue
            if item.name == "kb_search":
                result = _search_result(value["result"])
                for hit in result.hits:
                    if hit.doc_ref in result.catalog.authoritative_refs:
                        self.searched[hit.chunk_id] = hit.doc_ref
            if item.name == "kb_get":
                citation = decode(Citation, value["citation"])
                self.obtained.add(citation.chunk_id)
        pending = sorted(
            (self.searched.keys() - self.attempted)
            & (self.allowed if self.allowed is not None else self.searched.keys())
        )
        if pending:
            chosen = pending[:3]
            self.attempted.update(chosen)
            return scripted_calls(
                f"get-{self.turn}",
                tuple(("kb_get", {"chunk_id": i, "doc_ref": self.searched[i]}) for i in chosen),
            )
        requests: list[tuple[str, dict[str, object]]] = []
        for n, (needed, query) in enumerate(self.navigation):
            if n not in self.expanded and not set(needed) <= self.obtained:
                self.expanded.add(n)
                requests.append(("kb_search", {"query": query}))
                if len(requests) == self.batch_followups:
                    break
        if requests:
            return scripted_calls(f"expand-{self.turn}", tuple(requests))
        return ModelResponse("Read-only source acquisition finished; no business claim or action.")


class _ReadProvider(MockProvider):
    def __init__(
        self,
        query: str,
        navigation: tuple[tuple[tuple[str, ...], str], ...],
        *,
        allowed: frozenset[str] | None = None,
    ) -> None:
        super().__init__([ModelResponse() for _ in range(12)])
        self.navigator = EvidenceNavigator(query, navigation, allowed=allowed)

    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        super().complete(messages, tools=tools)
        return self.navigator.step(messages)


def policy_navigation(
    data: Dataset, category: Category, snapshot: str = "normal"
) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Trusted missing-section targets shared by read-only and business Agent paths."""
    retriever = Retriever(data)
    navigation = []
    for requirement in documentary_requirements(data, category, snapshot):
        c = requirement.components[0]
        chunks = tuple(
            k.chunk_id
            for k in retriever.index.chunks
            if k.doc_ref == c.doc_ref and k.section_id == c.section_id
        )
        document = next(d for d in data.documents if d.metadata.ref == c.doc_ref)
        section = next(s for s in document.sections if s.section_id == c.section_id)
        navigation.append((chunks, (section.heading + " " + document.metadata.title)[:256]))
    return tuple(navigation)


class _TopThree(Retriever):
    def __init__(self, data: Dataset, delegate: Retriever) -> None:
        super().__init__(data)
        self.delegate = delegate

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
    ) -> SearchResult:
        result = self.delegate.search(
            query, category=category, snapshot=snapshot, top_k=min(top_k, 3), method=method
        )
        # A1 adds lexical score metadata; preserve rank/source fields while using
        # the established tool wire schema, as the business adapter already does.
        return replace(
            result,
            hits=tuple(
                SearchHit(**{f.name: getattr(hit, f.name) for f in fields(SearchHit)})
                for hit in result.hits
            ),
        )


def acquire(
    data: Dataset,
    ticket_id: str,
    query: str,
    *,
    retriever: Retriever,
    snapshot: str = "normal",
    expand: bool = False,
    sections: tuple[tuple[str, str], ...] | None = None,
) -> MockRun:
    """Existing Agent/Registry only; no writes/assess/delegation exposed or invoked.

    Expansion obligations come from the same complete M3 policy catalog, never
    evaluation units or the answer oracle. Finite caps leave misses unfilled.
    """
    app = DemoApplication(data, ticket_id, snapshot=snapshot, approval="refuse")
    require(retriever.index == app.retriever.index, "acquisition_corpus")
    app.retriever = _TopThree(data, retriever)
    navigation = policy_navigation(data, app.category, snapshot) if expand else ()
    allowed = None
    if sections is not None:
        allowed = frozenset(
            c.chunk_id for c in retriever.index.chunks if (c.doc_ref, c.section_id) in sections
        )
        navigation = tuple((ids, q) for ids, q in navigation if set(ids) <= allowed)
    provider = _ReadProvider(query, navigation, allowed=allowed)
    try:
        with app.registry.restrict_tools(("kb_search", "kb_get")):
            result = Agent(
                provider,
                app.registry,
                limits=RunLimits(12, 28),
                total_limits=TotalLimits(16, 28, 1),
                trace=app.trace,
            ).run(query)
        return MockRun(app, result, provider, ())
    finally:
        app.close()


@dataclass(frozen=True)
class PolicyResolution:
    """Versioned documentary result, separate from lexical recall/business eligibility."""

    decision: AnswerDecision
    searches: int = 0
    full_reads: int = 0
    model_rounds: int = 0
    initial_hit_ids: tuple[str, ...] = ()
    contract: str = "lqd-policy-resolution-2.0.0"
    approval_calls: int = 0
    commits: int = 0
    unchanged_state_verified: bool = True


def resolve_policy(
    data: Dataset, ticket_id: str, query: str, spec: AnswerSpec, *, retriever: Retriever
) -> PolicyResolution:
    """Trusted application selects obligations; untrusted question never grants facts.

    Invalid/expired/FAQ requirements are rejected even when there are no reads.
    Only specified applicable sections receive navigation, rather than every
    policy in a category. A lexical hit or zero hits alone never means answerable.
    """
    if type(spec) is not AnswerSpec or spec.intent != "policy_explanation":
        return PolicyResolution(AnswerDecision("not_assessable", "unsupported_intent_or_facts"))
    try:
        require(retriever.index == Retriever(data).index, "resolution_corpus")
        from knowledge_ticket_agent.rules import evaluate_rules
        from knowledge_ticket_agent.schemas import Split

        require(
            any(t.ticket_id == ticket_id and t.split is Split.DEV for t in data.tickets),
            "resolution_dev_context",
        )
        require(
            evaluate_rules(data, ticket_id, snapshot=spec.snapshot).category is spec.category,
            "resolution_context_category",
        )
        selection = retriever.selection(category=spec.category, snapshot=spec.snapshot)
        require(not selection.conflicts, "resolution_conflict")
        require(
            type(spec.sections) is tuple
            and 0 < len(spec.sections) <= 16
            and len(set(spec.sections)) == len(spec.sections),
            "resolution_sections",
        )
        for ref, section in spec.sections:
            require(ref in selection.authoritative_refs, "resolution_authority")
            require(
                any(c.doc_ref == ref and c.section_id == section for c in retriever.index.chunks),
                "resolution_section",
            )
        run = acquire(
            data,
            ticket_id,
            query,
            retriever=retriever,
            snapshot=spec.snapshot,
            expand=True,
            sections=spec.sections,
        )
        decision = assess_parent_answer(run.app, spec)
        first = next(
            (r for r in run.app.registry.records if r.call.name == "kb_search" and r.result.ok),
            None,
        )
        initial: tuple[str, ...] = ()
        if first is not None:
            value = parse_json(first.result.content)
            assert isinstance(value, dict)
            initial = tuple(h.chunk_id for h in _search_result(value["result"]).hits)
        return PolicyResolution(
            decision,
            sum(r.call.name == "kb_search" for r in run.app.registry.records),
            sum(r.call.name == "kb_get" and r.result.ok for r in run.app.registry.records),
            run.result.model_rounds,
            initial,
            approval_calls=len(run.app.approval_calls),
            commits=len(run.app.workflow.store.snapshot.commits),
            unchanged_state_verified=run.app.initial == run.app.workflow.store.snapshot,
        )
    except (FixtureError, ValueError, TypeError, KeyError, AttributeError):
        return PolicyResolution(
            AnswerDecision("not_assessable", "invalid_or_inapplicable_contract")
        )
