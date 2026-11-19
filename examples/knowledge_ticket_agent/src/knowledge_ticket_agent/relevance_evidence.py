"""Exact finite-domain proof over canonical policy claims. No score-based support."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace

from knowledge_ticket_agent.citations import Citation, verify_citation
from knowledge_ticket_agent.evidence_contract import (
    VERSION,
    Conflict,
    Constraint,
    EvidenceBinding,
    EvidenceVerdict,
    FactRequirement,
    Modality,
    Operator,
    QueryObligation,
    QueryObligations,
    Scope,
    SourceClaim,
    Verdict,
    compatible,
    implies,
    intersect_scope,
    negate,
    normalized,
    scope_entails,
    scope_relation,
)
from knowledge_ticket_agent.policy_assertions import (
    ClaimSet,
    compile_catalog,
    compile_claims,
    ranking_passages,
)
from knowledge_ticket_agent.policy_query import plan_obligations, ranking_request
from knowledge_ticket_agent.retrieval import Retriever, SearchResult
from knowledge_ticket_agent.retrieval_relevance import (
    VIEW_VERSION,
    RelevanceAudit,
    RelevanceRetriever,
    RelevanceTrace,
    RetrievalView,
)
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import require

SELECTION_VERSION = "lqd-obligation-source-selection-2.0.0"


@dataclass(frozen=True)
class PolicySourceSelection:
    result: SearchResult
    read_dependencies: tuple[str, ...]
    ranking: RelevanceAudit
    metadata_added: tuple[str, ...]


def source_method_identity(retriever: RelevanceRetriever) -> str:
    return (
        SELECTION_VERSION
        + ":"
        + hashlib.sha256((retriever.method_identity + VIEW_VERSION).encode()).hexdigest()
    )


def select_sources(
    data: Dataset,
    retriever: RelevanceRetriever,
    query: str,
    *,
    category: Category | None = None,
    snapshot: str = "normal",
) -> PolicySourceSelection:
    """Plan source reads from query-free metadata, never unseen claim values.

    The raw R1 path remains unchanged. Policy fact-key locations define the
    finite source view; the same E5/BGE code ranks its compact assertion text.
    Greedy key coverage avoids
    schema/background sections consuming the three evidence slots. This is a
    read plan, not a claim that these sources actually satisfy the request.
    Every potentially overlapping counterclaim remains a required full read.
    """
    catalog = compile_catalog(data, retriever, category=category, snapshot=snapshot)
    plan = plan_obligations(query, catalog)
    coverage: dict[str, set[tuple[int, str]]] = {}
    dependencies: set[str] = set()
    if not plan.unknown and not catalog.unknown:
        for n, obligation in enumerate(plan.obligations):
            if obligation.covered is not True:
                continue
            if (
                obligation.anchor is not None
                and scope_entails(obligation.scope, obligation.anchor) is not True
            ):
                continue
            keys = {normalized(r.key) for b in obligation.alternatives for r in b}
            for location in catalog.claim_locations:
                if location.key not in keys:
                    continue
                if scope_relation(obligation.scope, location.scope) is not False:
                    dependencies.add(location.chunk_id)
                if scope_entails(obligation.scope, location.scope) is True:
                    coverage.setdefault(location.chunk_id, set()).add((n, location.key))
    identifiers = tuple(sorted(dependencies))
    require(len(identifiers) <= 32, "source_read_plan_capacity")
    if not identifiers:
        authority = retriever.selection(category=category, snapshot=snapshot)
        result = SearchResult(
            "not_assessable"
            if plan.unknown or catalog.unknown or any(o.covered is None for o in plan.obligations)
            else "no_match",
            retriever.index.corpus_version,
            retriever.index.sha256,
            source_method_identity(retriever),
            (),
            authority,
        )
        audit = RelevanceAudit(
            result,
            RelevanceTrace(
                (), (), (), (), (), (), {}, (), (), {}, "no_identified_source_obligations"
            ),
        )
        return PolicySourceSelection(result, (), audit, ())
    view = RetrievalView(
        ranking_request(plan),
        ranking_passages(
            data, retriever, catalog, identifiers, category=category, snapshot=snapshot
        ),
    )
    ranking = retriever.explain(query, category=category, snapshot=snapshot, view=view)
    scores = ranking.trace.raw_scores
    order = list(
        dict.fromkeys(
            (
                *[h.chunk_id for h in ranking.result.hits],
                *sorted(scores, key=lambda identifier: (-scores[identifier], identifier)),
                *sorted(coverage),
            )
        )
    )
    priorities = {identifier: n for n, identifier in enumerate(order)}
    selected: list[str] = []
    uncovered = set().union(*coverage.values()) if coverage else set()
    while coverage and len(selected) < 3:
        remaining = [i for i in coverage if i not in selected]
        if not remaining:
            break
        best = min(remaining, key=lambda i: (-len(coverage[i] & uncovered), priorities[i]))
        selected.append(best)
        uncovered.difference_update(coverage[best])
    if not coverage:
        selected = [h.chunk_id for h in ranking.result.hits]
    hits = tuple(
        retriever._hit(
            retriever.get_chunk(i, category=category, snapshot=snapshot),
            next((h.score for h in ranking.result.hits if h.chunk_id == i), 0.0),
        )
        for i in selected
    )
    result = replace(
        ranking.result,
        hits=hits,
        method=source_method_identity(retriever),
        status="policy_conflict"
        if ranking.result.catalog.conflicts
        else "ok"
        if hits
        else "no_match",
    )
    return PolicySourceSelection(
        result,
        tuple(sorted(dependencies)),
        ranking,
        tuple(sorted(dependencies - set(ranking.trace.generated))),
    )


def canonical_sources(
    data: Dataset,
    retriever: Retriever,
    citations: tuple[Citation, ...],
    *,
    category: Category | None = None,
    snapshot: str = "normal",
) -> None:
    require(type(citations) is tuple and len(citations) <= 32, "evidence_capacity")
    authority = retriever.selection(category=category, snapshot=snapshot).authoritative_refs
    for citation in citations:
        require(citation.doc_ref in authority, "non_authoritative_source")
        verify_citation(citation, data, retriever, category=category, snapshot=snapshot)
        c = retriever.get_chunk(citation.chunk_id, category=category, snapshot=snapshot)
        require(
            citation.quote == c.text
            and citation.start_char == c.start_char
            and citation.end_char == c.end_char,
            "full_evidence_required",
        )


def method_identity(retriever: Retriever, snapshot: str = "normal") -> str:
    payload = (
        VERSION,
        SELECTION_VERSION,
        VIEW_VERSION,
        getattr(retriever, "method_identity", "legacy"),
        retriever.index.sha256,
        snapshot,
    )
    return VERSION + ":" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def contradictions(
    claims: tuple[SourceClaim, ...],
    functional: frozenset[str],
    requests: tuple[Scope, ...] = (),
) -> tuple[tuple[Conflict, ...], tuple[str, ...]]:
    conflicts: list[Conflict] = []
    unknown = []
    by_key: dict[str, list[SourceClaim]] = {}
    for claim in claims:
        by_key.setdefault(claim.key, []).append(claim)
    for key, group in by_key.items():
        for n, left in enumerate(group):
            for right in group[n + 1 :]:
                relation = scope_relation(left.scope, right.scope)
                if relation is None:
                    unknown.append("unknown_condition_relationship")
                if relation is not True:
                    continue
                positive, prohibited = (
                    (right, left) if left.modality is Modality.FORBIDDEN else (left, right)
                )
                if (
                    prohibited.modality is not Modality.FORBIDDEN
                    or positive.modality is Modality.FORBIDDEN
                ):
                    continue
                a = (Constraint(key, positive.operator, positive.value),)
                b = (Constraint(key, prohibited.operator, prohibited.value),)
                # A required bound may be tightened by a prohibition without a
                # conflict. Explicit permission instead promises its whole set.
                opposed = (
                    compatible(a, b) if positive.modality is Modality.PERMITTED else implies(a, b)
                )
                if opposed is None:
                    unknown.append("unknown_value_relationship")
                elif opposed:
                    conflicts.append(Conflict(positive, prohibited, "opposite_applicable_claims"))
        if key not in functional:
            continue
        # Partition only at actual applicability boundaries. Joint constraints
        # catch contradictions that no individual pair can demonstrate.
        regions = set(requests or tuple(c.scope for c in group))
        for claim in group:
            for region in sorted(regions, key=repr):
                relation = scope_relation(region, claim.scope)
                if relation is None:
                    unknown.append("unknown_condition_relationship")
                intersection = intersect_scope(region, claim.scope)
                if intersection is not None:
                    regions.add(intersection)
            if len(regions) > 128:
                unknown.append("condition_partition_capacity")
                break
        else:
            for region in sorted(regions, key=repr):
                active = tuple(
                    c
                    for c in group
                    if c.modality is not Modality.PERMITTED
                    and scope_entails(region, c.scope) is True
                )
                constraints = tuple(
                    Constraint(
                        key,
                        negate(c.operator) if c.modality is Modality.FORBIDDEN else c.operator,
                        c.value,
                    )
                    for c in active
                )
                possible = compatible(constraints, ())
                if possible is None:
                    unknown.append("unknown_value_relationship")
                elif not possible and len(active) >= 2:
                    conflicts.append(
                        Conflict(
                            active[0],
                            active[1],
                            "inconsistent_joint_functional_constraints",
                            active[2:],
                        )
                    )
    return tuple(dict.fromkeys(conflicts)), tuple(dict.fromkeys(unknown))


def _matches(req: FactRequirement, claim: SourceClaim, scope: Scope) -> bool | None:
    if normalized(req.key) != claim.key:
        return False
    applies = scope_entails(scope, claim.scope)
    if applies is not True:
        return applies
    if req.modality is not None:
        if req.modality is not claim.modality:
            return False
    elif claim.modality is Modality.FORBIDDEN:
        return False
    if req.value is None:
        if req.operator is not None:
            return req.operator is claim.operator
        # An exclusion does not identify a required nominal value (e.g. a
        # signatory). Numeric bounds do describe a quantity constraint.
        return claim.value.kind == "number" or claim.operator is Operator.EQ
    supplied = (Constraint(req.key, claim.operator, claim.value),)
    requested = (Constraint(req.key, req.operator or Operator.EQ, req.value),)
    return (
        implies(requested, supplied)
        if claim.modality in {Modality.FORBIDDEN, Modality.PERMITTED}
        else implies(supplied, requested)
    )


def _bind(
    req: FactRequirement, scope: Scope, claims: tuple[SourceClaim, ...]
) -> tuple[tuple[SourceClaim, ...], bool]:
    matched, eligible = [], []
    uncertain = False
    for claim in claims:
        if normalized(req.key) != claim.key:
            continue
        applies = scope_entails(scope, claim.scope)
        if applies is None:
            uncertain = True
        if applies is True and (
            claim.modality is req.modality
            or (req.modality is None and claim.modality is not Modality.FORBIDDEN)
        ):
            eligible.append(claim)
        match = _matches(req, claim, scope)
        if match is None:
            uncertain = True
        elif match:
            matched.append(claim)
    if matched:
        return tuple(matched), uncertain
    if (
        req.functional
        and req.value is not None
        and eligible
        and req.modality in {Modality.FACT, Modality.REQUIRED}
    ):
        together = tuple(Constraint(req.key, c.operator, c.value) for c in eligible)
        possible = compatible(together, ())
        if possible is True:
            result = implies(
                together, (Constraint(req.key, req.operator or Operator.EQ, req.value),)
            )
            if result is True:
                return tuple(eligible), uncertain
            uncertain |= result is None
        else:
            uncertain |= possible is None
    return (), uncertain


def _prove_verified(
    plan: QueryObligations, sources: ClaimSet, *, method: str, retrieval_method: str
) -> EvidenceVerdict:
    require(
        type(plan) is QueryObligations
        and type(plan.obligations) is tuple
        and len(plan.obligations) <= 16,
        "proof_contract",
    )
    global_unknown = tuple(dict.fromkeys((*plan.unknown, *sources.unknown)))
    if not plan.obligations:
        return EvidenceVerdict(
            Verdict.NOT_ASSESSABLE,
            (),
            (),
            (),
            (*global_unknown, "no_query_obligations"),
            method,
            retrieval_method,
        )
    branches = plan.alternatives or (tuple(range(len(plan.obligations))),)
    require(
        len(branches) <= 64
        and all(
            b and all(type(i) is int and 0 <= i < len(plan.obligations) for i in b)
            for b in branches
        ),
        "query_expression",
    )
    outcomes = []
    for branch in branches:
        # One option per source-declared OR bundle. Expand a bounded DNF without
        # losing AND requirements or allowing unparsed query text to disappear.
        combinations: list[tuple[tuple[int, int], ...]] = [()]
        for index in branch:
            obligation = plan.obligations[index]
            require(
                0 < len(obligation.alternatives) <= 8
                and all(0 < len(b) <= 16 for b in obligation.alternatives),
                "obligation_contract",
            )
            combinations = [
                (*previous, (index, n))
                for previous in combinations
                for n in range(len(obligation.alternatives))
            ]
            require(len(combinations) <= 128, "proof_alternative_capacity")
        for combination in combinations:
            bindings: list[EvidenceBinding] = []
            missing: list[tuple[str, tuple[str, ...]]] = []
            unknown = list(global_unknown)
            wanted = tuple(
                (plan.obligations[i], n, plan.obligations[i].alternatives[n])
                for i, n in combination
            )
            keys = frozenset(normalized(req.key) for _, _, bundle in wanted for req in bundle)
            scopes = tuple(
                o.scope
                for o, _, _ in wanted
                if o.covered is True
                and (o.anchor is None or scope_entails(o.scope, o.anchor) is True)
            )
            relevant = tuple(
                c
                for c in sources.claims
                if c.key in keys
                and any(scope_relation(scope, c.scope) is not False for scope in scopes)
            )
            functional = frozenset(
                normalized(req.key) for _, _, bundle in wanted for req in bundle if req.functional
            )
            conflicts, uncertain = contradictions(relevant, functional, scopes)
            unknown.extend(uncertain)
            for obligation, n, bundle in wanted:
                if obligation.covered is not True:
                    missing.append((obligation.obligation_id, ("policy_scope_coverage",)))
                    if obligation.covered is None:
                        unknown.append("unknown_policy_scope_coverage")
                    continue
                absent = []
                for req in bundle:
                    anchored = (
                        True
                        if obligation.anchor is None
                        else scope_entails(obligation.scope, obligation.anchor)
                    )
                    found, uncertain_binding = (
                        _bind(req, obligation.scope, relevant)
                        if anchored is True
                        else ((), anchored is None)
                    )
                    if uncertain_binding:
                        unknown.append(obligation.obligation_id + ":unknown_condition_or_value")
                    if found:
                        qualifiers = tuple(
                            c
                            for c in relevant
                            if c.key == normalized(req.key)
                            and c not in found
                            and scope_entails(obligation.scope, c.scope) is True
                        )
                        bindings.append(
                            EvidenceBinding(obligation.obligation_id, n, req, found, qualifiers)
                        )
                    else:
                        absent.append(req.key)
                if absent:
                    missing.append((obligation.obligation_id, tuple(absent)))
            unread = tuple(
                sorted(
                    {
                        loc.chunk_id
                        for loc in sources.unread
                        if loc.key in keys
                        and any(scope_relation(scope, loc.scope) is not False for scope in scopes)
                    }
                )
            )
            if unread:
                missing.append(("source_closure", unread))
            status = (
                Verdict.NOT_ASSESSABLE
                if unknown or conflicts
                else Verdict.INSUFFICIENT
                if missing
                else Verdict.SUPPORTED
            )
            outcome = EvidenceVerdict(
                status,
                tuple(missing),
                tuple(bindings),
                conflicts,
                tuple(dict.fromkeys(unknown)),
                method,
                retrieval_method,
            )
            if status is Verdict.SUPPORTED:
                return outcome
            outcomes.append(outcome)
    # No alternative proved true: uncertainty in an unresolved alternative
    # cannot become known false because another alternative is incomplete.
    return min(
        outcomes,
        key=lambda v: (
            v.status is not Verdict.NOT_ASSESSABLE,
            sum(len(keys) for _, keys in v.missing),
        ),
    )


def verify_support(
    data: Dataset,
    retriever: Retriever,
    query: str,
    citations: tuple[Citation, ...],
    *,
    category: Category | None = None,
    snapshot: str = "normal",
) -> EvidenceVerdict:
    """Only verified full reads supply facts. Query constraints never become claims."""
    canonical_sources(data, retriever, citations, category=category, snapshot=snapshot)
    catalog = compile_catalog(data, retriever, category=category, snapshot=snapshot)
    plan = plan_obligations(query, catalog)
    require(plan.query == query, "proof_query_binding")
    claims = compile_claims(
        data, retriever, citations, catalog, category=category, snapshot=snapshot
    )
    read_ids = {c.chunk_id for c in citations}
    unread = tuple(loc for loc in catalog.claim_locations if loc.chunk_id not in read_ids)
    claims = ClaimSet(claims.claims, claims.unknown, unread)
    if retriever.selection(category=category, snapshot=snapshot).conflicts:
        claims = ClaimSet(
            claims.claims, (*claims.unknown, "unresolved_authority_conflict"), claims.unread
        )
    return _prove_verified(
        plan,
        claims,
        method=method_identity(retriever, snapshot),
        retrieval_method=getattr(retriever, "method_identity", "legacy"),
    )


def verify_business_sources(
    data: Dataset,
    retriever: Retriever,
    citations: tuple[Citation, ...],
    requirements: Iterable[tuple[str, str]],
    *,
    category: Category,
    snapshot: str = "normal",
) -> EvidenceVerdict:
    """Trusted frozen M3 documentary contract, not arbitrary semantic answerability.

    Only the original source-locked M3 rules may project whole-section metadata.
    They remain the authority for customer/invoice/identity facts. Query-based
    proof cannot consume this internal documentary projection.
    """
    from knowledge_ticket_agent.evidence_contract import Operator, Value
    from knowledge_ticket_agent.policy_assertions import span_for
    from knowledge_ticket_agent.rules import validate_rule_contract

    validate_rule_contract(data)
    canonical_sources(data, retriever, citations, category=category, snapshot=snapshot)
    scope = Scope(
        "trusted_m3", "documentary_read", category.value, "frozen_policy", data.manifest.as_of
    )
    obligations = tuple(
        QueryObligation(
            ref + "#" + section,
            scope,
            ((FactRequirement(ref + "#" + section, modality=Modality.FACT),),),
            (0, 0),
        )
        for ref, section in requirements
    )
    claims = []
    for citation in citations:
        span = span_for(data, citation, citation.start_char, citation.end_char, snapshot)
        claims.append(
            SourceClaim(
                scope,
                normalized(citation.doc_ref + "#" + citation.section_id),
                Operator.EQ,
                Value("text", span.quote_sha256),
                Modality.FACT,
                span,
                span,
                "frozen_m3_documentary",
            )
        )
    return _prove_verified(
        QueryObligations("TRUSTED_M3_DOCUMENTARY", obligations),
        ClaimSet(tuple(claims), ()),
        method=method_identity(retriever, snapshot),
        retrieval_method=getattr(retriever, "method_identity", "legacy"),
    )


def source_consistent(
    data: Dataset,
    retriever: Retriever,
    citations: tuple[Citation, ...],
    *,
    category: Category | None = None,
    snapshot: str = "normal",
) -> bool:
    if not isinstance(retriever, RelevanceRetriever):
        return True
    require(category is not None, "business_category")
    assert category is not None
    from knowledge_ticket_agent.approvals import documentary_requirements

    required = tuple(
        (c.doc_ref, c.section_id)
        for r in documentary_requirements(data, category, snapshot)
        for c in r.components
    )
    return (
        verify_business_sources(
            data, retriever, citations, required, category=category, snapshot=snapshot
        ).status
        is Verdict.SUPPORTED
    )


def method_audit(retriever: Retriever, snapshot: str = "normal") -> dict[str, str]:
    if not isinstance(retriever, RelevanceRetriever):
        return {}
    return {
        "evidence_method": method_identity(retriever, snapshot),
        "evidence_retrieval_method": retriever.method_identity,
        "evidence_basis": "frozen_m3_documentary",
    }
