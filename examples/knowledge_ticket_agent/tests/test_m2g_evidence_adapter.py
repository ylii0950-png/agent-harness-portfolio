"""Public section-reading exercises, not frozen evaluation regression fixtures."""

import ast
import builtins
import copy
import hashlib
import json
import os
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.citations import cite
from knowledge_ticket_agent.evidence_adapter import (
    EvidenceRequest,
    EvidenceRequirement,
    Outcome,
    Reason,
    RequirementMode,
    SectionComponent,
    assess_evidence,
    scope_for,
)
from knowledge_ticket_agent.retrieval import Retriever, SearchHit, SearchResult
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category, Dataset


def declaration(
    data: Dataset,
    *,
    snapshot: str = "normal",
    components: tuple[SectionComponent, ...] | None = None,
) -> EvidenceRequirement:
    # Explicit public documentary task: reading two published access sections.
    return EvidenceRequirement(
        "public-access-reading",
        "sections-1.0.0",
        scope_for(data, category=Category.ACCESS, snapshot=snapshot),
        components
        if components is not None
        else (
            SectionComponent("identity-text", "POL-ACCESS-01@1.0", "verification"),
            SectionComponent("safety-text", "POL-ACCESS-02@1.0", "security"),
        ),
    )


def collection(
    data: Dataset, requirement: EvidenceRequirement, refs: tuple[str, ...]
) -> EvidenceRequest:
    """Scripted public source set, no query or simulated ranking-quality claim."""
    r = Retriever(data)
    selection = r.selection(
        category=requirement.scope.category, snapshot=requirement.scope.snapshot
    )
    hits = []
    for ref in refs:
        for c in r.index.chunks:
            if c.doc_ref == ref:
                hits.append(
                    SearchHit(
                        c.chunk_id,
                        c.doc_ref,
                        c.section_id,
                        c.source_path,
                        c.start_char,
                        c.end_char,
                        c.start_line,
                        c.end_line,
                        1.0,
                        c.text[:600],
                        len(c.text) > 600,
                    )
                )
    result = SearchResult(
        "policy_conflict" if selection.conflicts else ("ok" if hits else "no_match"),
        r.index.corpus_version,
        r.index.sha256,
        "public-scripted-source-set",
        tuple(hits),
        selection,
    )
    # Inapplicable-source tests supply a canonical-looking citation; get_chunk
    # must still reject that source inside the adapter's actual declared scope.
    citations = []
    for hit in hits:
        try:
            citations.append(
                cite(
                    r,
                    hit.chunk_id,
                    category=requirement.scope.category,
                    snapshot=requirement.scope.snapshot,
                )
            )
        except ValueError:
            from knowledge_ticket_agent.citations import Citation

            c = next(c for c in r.index.chunks if c.chunk_id == hit.chunk_id)
            citations.append(
                Citation(
                    c.chunk_id,
                    r.index.sha256,
                    c.doc_ref,
                    c.section_id,
                    c.start_char,
                    c.end_char,
                    c.start_line,
                    c.end_line,
                    c.text,
                )
            )
    return EvidenceRequest(
        requirement.requirement_id, requirement.version, requirement.scope, result, tuple(citations)
    )


@pytest.fixture
def complete(dataset: Dataset) -> tuple[EvidenceRequirement, EvidenceRequest]:
    requirement = declaration(dataset)
    return requirement, collection(dataset, requirement, ("POL-ACCESS-01@1.0", "POL-ACCESS-02@1.0"))


def test_complete_sections_are_only_specified_evidence(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest]
) -> None:
    requirement, request = complete
    result = assess_evidence(dataset, (requirement,), request)
    assert result.outcome == Outcome.SUFFICIENT
    assert result.reasons == (Reason.COMPLETE,)
    assert result.covered_components == ("identity-text", "safety-text")
    assert result.missing_components == () and len(result.citations) == 2
    assert not result.navigation_citations
    assert set(asdict(result)) == {
        "outcome",
        "reasons",
        "covered_components",
        "missing_components",
        "citations",
        "navigation_citations",
    }


@pytest.mark.parametrize(
    "refs,covered,missing,faq_only",
    [
        ((), (), ("identity-text", "safety-text"), False),
        (("POL-ACCESS-01@1.0",), ("identity-text",), ("safety-text",), False),
        (("FAQ-ACCESS-01@1.0",), (), ("identity-text", "safety-text"), True),
        (("POL-ACCESS-01@1.0", "FAQ-ACCESS-01@1.0"), ("identity-text",), ("safety-text",), False),
    ],
)
def test_incomplete_and_navigation_do_not_fill_requirements(
    dataset: Dataset,
    refs: tuple[str, ...],
    covered: tuple[str, ...],
    missing: tuple[str, ...],
    faq_only: bool,
) -> None:
    requirement = declaration(dataset)
    request = collection(dataset, requirement, refs)
    result = assess_evidence(dataset, (requirement,), request)
    assert result.outcome == Outcome.INSUFFICIENT
    assert result.covered_components == covered and result.missing_components == missing
    assert result.reasons == ((Reason.MISSING, Reason.FAQ_ONLY) if faq_only else (Reason.MISSING,))


def test_policy_plus_navigation_preserves_separate_purpose(dataset: Dataset) -> None:
    requirement = declaration(dataset)
    request = collection(
        dataset, requirement, ("POL-ACCESS-01@1.0", "POL-ACCESS-02@1.0", "FAQ-ACCESS-01@1.0")
    )
    result = assess_evidence(dataset, (requirement,), request)
    assert result.outcome == Outcome.SUFFICIENT
    assert len(result.citations) == 2 and len(result.navigation_citations) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("snapshot", "conflict"),
        ("category", Category.BILLING),
        ("index_sha256", "0" * 64),
        ("corpus_version", "old"),
        ("manifest_sha256", "0" * 64),
        ("as_of", "2000-01-01T00:00:00Z"),
    ],
)
def test_declared_scope_mismatch(
    dataset: Dataset,
    complete: tuple[EvidenceRequirement, EvidenceRequest],
    field: str,
    value: Any,
) -> None:
    requirement, request = complete
    bad = replace(request, scope=replace(request.scope, **{field: value}))
    result = assess_evidence(dataset, (requirement,), bad)
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (Reason.SCOPE,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("corpus_version", "old"),
        ("index_sha256", "0" * 64),
        ("status", "ok-but-unverified"),
        ("method", ""),
    ],
)
def test_search_metadata_mismatch(
    dataset: Dataset,
    complete: tuple[EvidenceRequirement, EvidenceRequest],
    field: str,
    value: Any,
) -> None:
    requirement, request = complete
    result = assess_evidence(
        dataset, (requirement,), replace(request, result=replace(request.result, **{field: value}))
    )
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (Reason.INVALID_RESULT,)


@pytest.mark.parametrize(
    "refs", [("POL-BILL-01@0.9",), ("POL-BILL-02@1.0",), ("CONFLICT-RET-A@1.0",)]
)
def test_inapplicable_retrieved_sources_fail_closed(
    dataset: Dataset, refs: tuple[str, ...]
) -> None:
    requirement = declaration(dataset)
    result = assess_evidence(dataset, (requirement,), collection(dataset, requirement, refs))
    assert result.outcome == Outcome.NOT_ASSESSABLE
    assert result.reasons == (Reason.INAPPLICABLE_SOURCE,)


def test_unresolved_conflict_blocks_even_complete_sources(dataset: Dataset) -> None:
    requirement = EvidenceRequirement(
        "public-conflict-reading",
        "sections-1.0.0",
        scope_for(dataset, category=Category.EXPORT, snapshot="conflict"),
        (
            SectionComponent("a", "CONFLICT-RET-A@1.0", "retention"),
            SectionComponent("b", "CONFLICT-RET-B@1.0", "retention"),
        ),
    )
    result = assess_evidence(
        dataset,
        (requirement,),
        collection(dataset, requirement, ("CONFLICT-RET-A@1.0", "CONFLICT-RET-B@1.0")),
    )
    assert result.outcome == Outcome.CONFLICT and result.reasons == (Reason.CONFLICT,)
    assert result.missing_components == ()


@pytest.mark.parametrize(
    "identifier,version,reason",
    [
        ("unsupported-reading", "sections-1.0.0", Reason.UNKNOWN_REQUIREMENT),
        ("public-access-reading", "sections-9.0.0", Reason.REQUIREMENT_VERSION),
    ],
)
def test_unknown_intent_or_version(
    dataset: Dataset,
    complete: tuple[EvidenceRequirement, EvidenceRequest],
    identifier: str,
    version: str,
    reason: Reason,
) -> None:
    requirement, request = complete
    result = assess_evidence(
        dataset,
        (requirement,),
        replace(request, requirement_id=identifier, requirement_version=version),
    )
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (reason,)


def test_business_applicability_abstains_even_with_all_sources(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest]
) -> None:
    requirement, request = complete
    requirement = replace(requirement, mode=RequirementMode.BUSINESS_APPLICABILITY)
    result = assess_evidence(dataset, (requirement,), request)
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (Reason.BUSINESS_FACTS,)
    assert not result.missing_components  # Source presence is not fact verification.
    assert "facts" not in EvidenceRequest.__dataclass_fields__


@pytest.mark.parametrize(
    "mutation",
    ["quote", "span", "lines", "index", "duplicate", "missing", "extra", "partial", "source"],
)
def test_forged_duplicate_or_incomplete_citations(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest], mutation: str
) -> None:
    requirement, request = complete
    citations = list(request.citations)
    first = citations[0]
    if mutation == "duplicate":
        citations.append(first)
    elif mutation == "missing":
        citations.pop()
    elif mutation == "extra":
        extra = collection(dataset, requirement, ("FAQ-ACCESS-01@1.0",))
        citations.append(extra.citations[0])
    elif mutation == "quote":
        citations[0] = replace(first, quote="invented source")
    elif mutation == "span":
        citations[0] = replace(first, end_char=first.end_char + 1)
    elif mutation == "lines":
        citations[0] = replace(first, start_line=True)
    elif mutation == "index":
        citations[0] = replace(first, index_sha256="0" * 64)
    elif mutation == "source":
        citations[0] = replace(first, doc_ref="POL-ACCESS-02@1.0")
    else:
        citations[0] = replace(
            first, end_char=first.start_char + 5, quote=first.quote[:5], end_line=first.start_line
        )
    result = assess_evidence(dataset, (requirement,), replace(request, citations=tuple(citations)))
    assert result.outcome == Outcome.NOT_ASSESSABLE
    assert result.reasons == (
        (Reason.DUPLICATE_SOURCE,) if mutation == "duplicate" else (Reason.INVALID_CITATION,)
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0, True, 10**1000])
def test_malformed_scores_have_stable_errors(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest], value: object
) -> None:
    requirement, request = complete
    first = replace(request.result.hits[0], score=value)  # type: ignore[arg-type]
    result = assess_evidence(
        dataset,
        (requirement,),
        replace(request, result=replace(request.result, hits=(first, *request.result.hits[1:]))),
    )
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (Reason.INVALID_RESULT,)


def test_duplicate_results_rejected(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest]
) -> None:
    requirement, request = complete
    result = assess_evidence(
        dataset,
        (requirement,),
        replace(
            request,
            result=replace(request.result, hits=(*request.result.hits, request.result.hits[0])),
        ),
    )
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (Reason.DUPLICATE_SOURCE,)


@pytest.mark.parametrize(
    "mutation",
    ["empty", "duplicate", "unknown_section", "faq_requirement", "wrong_binding", "unknown_mode"],
)
def test_bad_trusted_declarations_do_not_claim_sufficient(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest], mutation: str
) -> None:
    requirement, request = complete
    if mutation == "empty":
        requirement = replace(requirement, components=())
    elif mutation == "duplicate":
        requirement = replace(requirement, components=(requirement.components[0],) * 2)
    elif mutation == "unknown_section":
        requirement = replace(
            requirement, components=(SectionComponent("x", "POL-ACCESS-01@1.0", "x"),)
        )
    elif mutation == "faq_requirement":
        requirement = replace(
            requirement, components=(SectionComponent("x", "FAQ-ACCESS-01@1.0", "navigation"),)
        )
    elif mutation == "wrong_binding":
        scope = replace(requirement.scope, manifest_sha256="0" * 64)
        requirement, request = replace(requirement, scope=scope), replace(request, scope=scope)
    else:
        requirement = replace(requirement, mode="invented")  # type: ignore[arg-type]
    result = assess_evidence(dataset, (requirement,), request)
    assert result.outcome == Outcome.NOT_ASSESSABLE
    assert result.reasons == (
        (Reason.SOURCE_BINDING,) if mutation == "wrong_binding" else (Reason.INVALID_REQUIREMENTS,)
    )


def test_public_dev_search_interface_without_ranking_changes(dataset: Dataset) -> None:
    # A public DEV ticket is used only as an integration input; not scored/tuned.
    ticket = next(t for t in dataset.tickets if t.ticket_id == "KT-D01")
    scope = scope_for(dataset, category=Category.BILLING)
    requirement = EvidenceRequirement(
        "public-billing-reading",
        "sections-1.0.0",
        scope,
        (SectionComponent("restrictions-text", "POL-BILL-02@1.0", "restrictions"),),
    )
    for retriever in (Retriever(dataset), StrategyRetriever(dataset, config=strategy("A1"))):
        result = retriever.search(ticket.body, category=Category.BILLING)
        request = EvidenceRequest(
            requirement.requirement_id,
            requirement.version,
            scope,
            result,
            tuple(cite(retriever, h.chunk_id, category=Category.BILLING) for h in result.hits),
        )
        first = assess_evidence(dataset, (requirement,), request)
        assert first == assess_evidence(dataset, (requirement,), request)
        assert first.outcome in {Outcome.SUFFICIENT, Outcome.INSUFFICIENT}
        # No ranking metric or target pass rate is computed/selected here.
        assert first.reasons in {(Reason.COMPLETE,), (Reason.MISSING,)}


def test_no_io_ranking_permission_or_input_mutation(
    dataset: Dataset,
    complete: tuple[EvidenceRequirement, EvidenceRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement, request = complete
    before = copy.deepcopy((dataset, requirement, request))

    def denied(*args: object, **kwargs: object) -> Any:
        raise AssertionError("Adapter must not perform IO, ranking or authorization")

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", denied)
        m.setattr(Path, "open", denied)
        m.setattr(os, "open", denied)
        m.setattr(Retriever, "search", denied)
        m.setattr(StrategyRetriever, "search", denied)
        m.setattr("knowledge_ticket_agent.retrieval.lexical_rank", denied)
        m.setattr("agent_harness_lab.permissions.PermissionEngine.authorize", denied)
        m.setattr("agent_harness_lab.tools.ToolRegistry.invoke", denied)
        one = assess_evidence(dataset, (requirement,), request)
        two = assess_evidence(dataset, (requirement,), request)
    assert one == two and one.outcome == Outcome.SUFFICIENT
    assert (dataset, requirement, request) == before
    with pytest.raises(FrozenInstanceError):
        one.outcome = Outcome.CONFLICT  # type: ignore[misc]


def test_adapter_import_boundary() -> None:
    from knowledge_ticket_agent import evidence_adapter

    assert evidence_adapter.__file__ is not None
    tree = ast.parse(Path(evidence_adapter.__file__).read_text())
    forbidden = {
        "os",
        "pathlib",
        "socket",
        "subprocess",
        "gold",
        "formal_ledger",
        "formal_operator",
        "formal_scoring",
        "evaluation_custody",
    }
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules.update(a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names)
    assert all(name is None or name.split(".")[-1] not in forbidden for name in modules)
    assert "query" not in EvidenceRequest.__dataclass_fields__


def test_long_independently_authored_section_requires_every_part(copied: Path) -> None:
    # New public test prose in a temporary corpus copy; original resources unchanged.
    path = copied / "documents/POL-ACCESS-01-1.0.md"
    text = path.read_text()
    body = " ".join(f"public_archive_token_{i}" for i in range(420))
    path.write_text(text.rstrip() + " " + body + "\n")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for resource in manifest["resources"]:
        if resource["path"] == "documents/POL-ACCESS-01-1.0.md":
            raw = path.read_bytes()
            resource.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest_path.write_text(json.dumps(manifest) + "\n")
    data = load_dataset(copied)
    requirement = declaration(
        data, components=(SectionComponent("archive-text", "POL-ACCESS-01@1.0", "verification"),)
    )
    request = collection(data, requirement, ("POL-ACCESS-01@1.0",))
    assert len(request.result.hits) >= 3 and any(h.truncated for h in request.result.hits)
    assert assess_evidence(data, (requirement,), request).outcome == Outcome.SUFFICIENT
    partial = replace(
        request,
        result=replace(request.result, hits=request.result.hits[:-1]),
        citations=request.citations[:-1],
    )
    result = assess_evidence(data, (requirement,), partial)
    assert result.outcome == Outcome.INSUFFICIENT
    assert result.missing_components == ("archive-text",)
    assert any(len(c.quote) > 600 for c in result.citations)


def test_lexical_relevance_does_not_establish_specified_sections(dataset: Dataset) -> None:
    requirement = declaration(dataset)
    retriever = Retriever(dataset)
    # Independently chosen public routing-section term; no held-out input.
    found = retriever.search("routing", category=Category.ACCESS)
    assert found.hits and all(h.score > 0 for h in found.hits)
    request = EvidenceRequest(
        requirement.requirement_id,
        requirement.version,
        requirement.scope,
        found,
        tuple(cite(retriever, h.chunk_id, category=Category.ACCESS) for h in found.hits),
    )
    result = assess_evidence(dataset, (requirement,), request)
    assert result.outcome == Outcome.INSUFFICIENT and result.reasons == (Reason.MISSING,)
    assert result.covered_components == ()
    assert result.missing_components == ("identity-text", "safety-text")


@pytest.mark.parametrize(
    "mutation",
    [
        "catalog",
        "hit_source",
        "excerpt",
        "chunk_length",
        "hit_list",
        "citations_list",
        "large_quote",
        "duplicate_config",
    ],
)
def test_malformed_contracts_fail_conservatively(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest], mutation: str
) -> None:
    requirement, request = complete
    requirements: tuple[EvidenceRequirement, ...] = (requirement,)
    expected = Reason.INVALID_RESULT
    if mutation == "catalog":
        request = replace(
            request,
            result=replace(
                request.result, catalog=replace(request.result.catalog, applicable_refs=())
            ),
        )
    elif mutation in {"hit_source", "excerpt", "chunk_length"}:
        field, value = {
            "hit_source": ("doc_ref", "forged@1"),
            "excerpt": ("excerpt", "forged"),
            "chunk_length": ("chunk_id", "z" * 257),
        }[mutation]
        fields: dict[str, Any] = {field: value}
        first = replace(request.result.hits[0], **fields)
        request = replace(
            request, result=replace(request.result, hits=(first, *request.result.hits[1:]))
        )
    elif mutation == "hit_list":
        request = replace(request, result=replace(request.result, hits=list(request.result.hits)))  # type: ignore[arg-type]
    elif mutation == "citations_list":
        expected = Reason.INVALID_CITATION
        request = replace(request, citations=list(request.citations))  # type: ignore[arg-type]
    elif mutation == "large_quote":
        expected = Reason.INVALID_CITATION
        request = replace(
            request,
            citations=(replace(request.citations[0], quote="z" * 9000), *request.citations[1:]),
        )
    else:
        expected = Reason.INVALID_REQUIREMENTS
        requirements = (requirement, requirement)
    result = assess_evidence(dataset, requirements, request)
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (expected,)


def test_corrupt_dataset_is_not_assessable(
    dataset: Dataset, complete: tuple[EvidenceRequirement, EvidenceRequest]
) -> None:
    requirement, request = complete
    damaged = replace(
        dataset,
        documents=(
            replace(dataset.documents[0], text=dataset.documents[0].text + " corruption"),
            *dataset.documents[1:],
        ),
    )
    result = assess_evidence(damaged, (requirement,), request)
    assert result.outcome == Outcome.NOT_ASSESSABLE and result.reasons == (Reason.SOURCE_BINDING,)
