"""Read-only completeness of trusted specified sections, never business permission.

The application supplies the Dataset and immutable requirements as trusted
configuration. Never register requirements from a question, tool argument or
evaluation annotation. This Python API is not a same-user trust boundary.
"""

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from knowledge_ticket_agent.chunking import MAX_CHUNK_BYTES
from knowledge_ticket_agent.citations import Citation, cite, verify_citation
from knowledge_ticket_agent.evidence import EvidenceUnit, covered
from knowledge_ticket_agent.retrieval import Retriever, SearchHit, SearchResult
from knowledge_ticket_agent.retrieval_strategies import StrategyHit
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind
from knowledge_ticket_agent.strict import FixtureError, require


class Outcome(StrEnum):
    SUFFICIENT = "SUFFICIENT_FOR_SPECIFIED_EVIDENCE"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
    CONFLICT = "CONFLICT"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class RequirementMode(StrEnum):
    SOURCE_SECTIONS = "source_sections_only"
    BUSINESS_APPLICABILITY = "business_applicability"


class Reason(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_REQUIREMENTS = "invalid_trusted_requirements"
    UNKNOWN_REQUIREMENT = "unknown_requirement_or_intent"
    REQUIREMENT_VERSION = "requirement_version_mismatch"
    SCOPE = "scope_mismatch"
    SOURCE_BINDING = "source_binding_invalid"
    INVALID_RESULT = "retrieval_result_invalid"
    INAPPLICABLE_SOURCE = "source_not_applicable"
    DUPLICATE_SOURCE = "duplicate_evidence"
    INVALID_CITATION = "citation_invalid_or_incomplete"
    CONFLICT = "unresolved_authority_conflict"
    BUSINESS_FACTS = "business_applicability_not_assessed"
    MISSING = "specified_components_missing"
    FAQ_ONLY = "faq_is_navigation_only"
    COMPLETE = "specified_source_sections_complete"


@dataclass(frozen=True)
class EvidenceScope:
    corpus_version: str
    manifest_sha256: str
    index_sha256: str
    as_of: str
    category: Category
    snapshot: str


@dataclass(frozen=True)
class SectionComponent:
    component_id: str
    doc_ref: str
    section_id: str


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    version: str
    scope: EvidenceScope
    components: tuple[SectionComponent, ...]
    mode: RequirementMode = RequirementMode.SOURCE_SECTIONS


@dataclass(frozen=True)
class EvidenceRequest:
    requirement_id: str
    requirement_version: str
    scope: EvidenceScope
    result: SearchResult
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class EvidenceAssessment:
    outcome: Outcome
    reasons: tuple[Reason, ...]
    covered_components: tuple[str, ...] = ()
    missing_components: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    navigation_citations: tuple[Citation, ...] = ()


def scope_for(data: Dataset, *, category: Category, snapshot: str = "normal") -> EvidenceScope:
    """Trusted configuration helper; not a declaration of business applicability."""
    retriever = Retriever(data)
    retriever.selection(category=category, snapshot=snapshot)
    return EvidenceScope(
        data.manifest.corpus_version,
        data.manifest_sha256,
        retriever.index.sha256,
        data.manifest.as_of,
        category,
        snapshot,
    )


def _label(value: object) -> bool:
    return type(value) is str and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value))


def _scope(value: EvidenceScope) -> bool:
    return (
        type(value) is EvidenceScope
        and isinstance(value.category, Category)
        and type(value.snapshot) is str
        and 0 < len(value.snapshot) <= 64
        and all(
            type(v) is str and 0 < len(v) <= 128
            for v in (value.corpus_version, value.as_of, value.manifest_sha256, value.index_sha256)
        )
    )


def _requirements(items: tuple[EvidenceRequirement, ...]) -> None:
    require(type(items) is tuple and len(items) <= 16, "adapter_requirements")
    identifiers = []
    for item in items:
        require(
            type(item) is EvidenceRequirement
            and _label(item.requirement_id)
            and _label(item.version)
            and _scope(item.scope)
            and isinstance(item.mode, RequirementMode),
            "adapter_requirements",
        )
        require(
            type(item.components) is tuple and 0 < len(item.components) <= 8, "adapter_components"
        )
        keys = []
        sources = []
        for component in item.components:
            require(
                type(component) is SectionComponent
                and _label(component.component_id)
                and _label(component.section_id)
                and type(component.doc_ref) is str
                and 0 < len(component.doc_ref) <= 128,
                "adapter_components",
            )
            keys.append(component.component_id)
            sources.append((component.doc_ref, component.section_id))
        require(
            len(set(keys)) == len(keys) and len(set(sources)) == len(sources), "adapter_components"
        )
        identifiers.append(item.requirement_id)
    require(len(set(identifiers)) == len(identifiers), "adapter_requirements")


def _hits(result: SearchResult, retriever: Retriever, request: EvidenceRequest) -> None:
    require(
        type(result) is SearchResult and type(result.hits) is tuple and len(result.hits) <= 5,
        "adapter_result",
    )
    selection = retriever.selection(
        category=request.scope.category, snapshot=request.scope.snapshot
    )
    require(
        result.corpus_version == request.scope.corpus_version
        and result.index_sha256 == request.scope.index_sha256
        and result.catalog == selection
        and type(result.method) is str
        and 0 < len(result.method) <= 256,
        "adapter_result",
    )
    require(all(type(h) in {SearchHit, StrategyHit} for h in result.hits), "adapter_result")
    require(
        all(type(h.chunk_id) is str and 0 < len(h.chunk_id) <= 256 for h in result.hits),
        "adapter_result",
    )
    require(len({h.chunk_id for h in result.hits}) == len(result.hits), "adapter_duplicate")
    expected_status = (
        "policy_conflict" if selection.conflicts else ("ok" if result.hits else "no_match")
    )
    require(result.status == expected_status, "adapter_result")
    for hit in result.hits:
        require(
            type(hit.score) is float and math.isfinite(hit.score) and hit.score > 0,
            "adapter_result",
        )
        chunk = retriever.get_chunk(
            hit.chunk_id, category=request.scope.category, snapshot=request.scope.snapshot
        )
        for field in (
            "doc_ref",
            "section_id",
            "source_path",
            "start_char",
            "end_char",
            "start_line",
            "end_line",
        ):
            require(
                type(getattr(hit, field)) is type(getattr(chunk, field))
                and getattr(hit, field) == getattr(chunk, field),
                "adapter_result",
            )
        require(
            type(hit.truncated) is bool
            and hit.truncated == (len(chunk.text) > 600)
            and type(hit.excerpt) is str
            and len(hit.excerpt) <= 600
            and hit.excerpt == chunk.text[:600]
            and hit.trust == "low_trust_source_data",
            "adapter_result",
        )


def assess_evidence(
    data: Dataset, trusted_requirements: tuple[EvidenceRequirement, ...], request: EvidenceRequest
) -> EvidenceAssessment:
    """Pure source-section check; no search, filesystem, fact evaluator or action.

    Configuration must be supplied by the trusted caller, separate from request
    parsing. Positive results assert only the configured documentary coverage.
    """
    failure = Reason.INVALID_REQUIREMENTS
    try:
        _requirements(trusted_requirements)
        failure = Reason.INVALID_REQUEST
        require(
            type(request) is EvidenceRequest
            and _label(request.requirement_id)
            and _label(request.requirement_version)
            and _scope(request.scope),
            "adapter_request",
        )
        requirement = next(
            (r for r in trusted_requirements if r.requirement_id == request.requirement_id), None
        )
        if requirement is None:
            return EvidenceAssessment(Outcome.NOT_ASSESSABLE, (Reason.UNKNOWN_REQUIREMENT,))
        if request.requirement_version != requirement.version:
            return EvidenceAssessment(Outcome.NOT_ASSESSABLE, (Reason.REQUIREMENT_VERSION,))
        if request.scope != requirement.scope:
            return EvidenceAssessment(Outcome.NOT_ASSESSABLE, (Reason.SCOPE,))
        failure = Reason.SOURCE_BINDING
        require(type(data) is Dataset, "adapter_data")
        retriever = Retriever(data)
        require(
            scope_for(
                data, category=requirement.scope.category, snapshot=requirement.scope.snapshot
            )
            == requirement.scope,
            "adapter_scope",
        )
        selection = retriever.selection(
            category=request.scope.category, snapshot=request.scope.snapshot
        )
        failure = Reason.INVALID_REQUIREMENTS
        units = []
        for component in requirement.components:
            chunks = tuple(
                c
                for c in retriever.index.chunks
                if c.doc_ref == component.doc_ref and c.section_id == component.section_id
            )
            require(
                bool(chunks) and all(c.kind != DocKind.FAQ for c in chunks), "adapter_component"
            )
            if component.doc_ref not in selection.authoritative_refs:
                return EvidenceAssessment(Outcome.NOT_ASSESSABLE, (Reason.INAPPLICABLE_SOURCE,))
            units.append(EvidenceUnit(component.component_id, (tuple(c.chunk_id for c in chunks),)))
        failure = Reason.INVALID_RESULT
        _hits(request.result, retriever, request)
        failure = Reason.INVALID_CITATION
        require(
            type(request.citations) is tuple
            and len(request.citations) <= 5
            and all(type(c) is Citation for c in request.citations),
            "adapter_citations",
        )
        require(all(type(c.chunk_id) is str for c in request.citations), "adapter_citations")
        require(
            all(
                len(c.chunk_id) <= 256
                and type(c.quote) is str
                and len(c.quote) <= MAX_CHUNK_BYTES
                and len(c.quote.encode()) <= MAX_CHUNK_BYTES
                for c in request.citations
            ),
            "adapter_citations",
        )
        if len({c.chunk_id for c in request.citations}) != len(request.citations):
            return EvidenceAssessment(Outcome.NOT_ASSESSABLE, (Reason.DUPLICATE_SOURCE,))
        supplied = {c.chunk_id: c for c in request.citations}
        require(supplied.keys() == {h.chunk_id for h in request.result.hits}, "adapter_citations")
        authority = []
        navigation = []
        relevant = {c for u in units for bundle in u.alternatives for c in bundle}
        for hit in sorted(request.result.hits, key=lambda h: h.chunk_id):
            full = cite(
                retriever,
                hit.chunk_id,
                category=request.scope.category,
                snapshot=request.scope.snapshot,
            )
            verify_citation(
                supplied[hit.chunk_id],
                data,
                retriever,
                category=request.scope.category,
                snapshot=request.scope.snapshot,
                require_authority=False,
            )
            require(supplied[hit.chunk_id] == full, "adapter_full_citation")
            if hit.doc_ref not in selection.authoritative_refs:
                navigation.append(full)
            elif hit.chunk_id in relevant:
                authority.append(full)
        obtained = frozenset(c.chunk_id for c in authority)
        matched = tuple(sorted(u.unit_id for u in units if covered(u, obtained)))
        missing = tuple(sorted(u.unit_id for u in units if not covered(u, obtained)))
        reasons: tuple[Reason, ...]
        if selection.conflicts:
            outcome, reasons = Outcome.CONFLICT, (Reason.CONFLICT,)
        elif requirement.mode == RequirementMode.BUSINESS_APPLICABILITY:
            outcome, reasons = Outcome.NOT_ASSESSABLE, (Reason.BUSINESS_FACTS,)
        elif missing:
            outcome, reasons = Outcome.INSUFFICIENT, (Reason.MISSING,)
            if navigation and len(navigation) == len(request.result.hits):
                reasons += (Reason.FAQ_ONLY,)
        else:
            outcome, reasons = Outcome.SUFFICIENT, (Reason.COMPLETE,)
        return EvidenceAssessment(
            outcome, reasons, matched, missing, tuple(authority), tuple(navigation)
        )
    except (FixtureError, TypeError, ValueError, AttributeError, KeyError) as exc:
        if isinstance(exc, FixtureError):
            if exc.code == "adapter_duplicate":
                failure = Reason.DUPLICATE_SOURCE
            elif exc.code == "inapplicable_chunk":
                failure = Reason.INAPPLICABLE_SOURCE
        return EvidenceAssessment(Outcome.NOT_ASSESSABLE, (failure,))
