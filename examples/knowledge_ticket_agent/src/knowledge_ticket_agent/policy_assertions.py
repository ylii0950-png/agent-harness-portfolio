"""Finite policy assertion language: source metadata first, query-independent claims.

Normative source lines are strict JSON declarations. Unparsed normative prose is
Unknown, not evidence. No NLP/model score, test fixture or answer table is loaded.
"""

import hashlib
import re
import unicodedata
import weakref
from collections import OrderedDict
from dataclasses import dataclass, replace
from dataclasses import field as cache_field

from knowledge_ticket_agent.citations import Citation, cite, verify_citation
from knowledge_ticket_agent.evidence_contract import (
    FactRequirement,
    Modality,
    Operator,
    Scope,
    SourceClaim,
    SourceSpan,
    Value,
    compatible,
    normalized,
    unit_scale,
)
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import decode, parse_json, require


@dataclass(frozen=True)
class TermDeclaration:
    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class UnitDeclaration:
    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Term:
    declaration: TermDeclaration
    span: SourceSpan


@dataclass(frozen=True)
class Unit:
    declaration: UnitDeclaration
    span: SourceSpan


@dataclass(frozen=True)
class ContextDeclaration:
    name: str
    scope: Scope


@dataclass(frozen=True)
class SchemaDeclaration:
    context: str
    name: str
    phrases: tuple[str, ...]
    alternatives: tuple[tuple[FactRequirement, ...], ...]


@dataclass(frozen=True)
class ClaimDeclaration:
    context: str
    key: str
    operator: Operator
    value: Value
    modality: Modality


@dataclass(frozen=True)
class Context:
    name: str
    scope: Scope
    span: SourceSpan


@dataclass(frozen=True)
class PolicySchema:
    declaration: SchemaDeclaration
    context: Context
    span: SourceSpan


@dataclass(frozen=True)
class ClaimLocation:
    scope: Scope
    key: str
    chunk_id: str


@dataclass(frozen=True)
class PolicyCatalog:
    contexts: tuple[Context, ...]
    terms: tuple[Term, ...]
    schemas: tuple[PolicySchema, ...]
    claim_locations: tuple[ClaimLocation, ...]
    index_sha256: str
    snapshot: str
    unknown: tuple[str, ...]
    units: tuple[Unit, ...] = ()


@dataclass(frozen=True)
class ClaimSet:
    claims: tuple[SourceClaim, ...]
    unknown: tuple[str, ...]
    unread: tuple[ClaimLocation, ...] = ()


@dataclass
class _SourceCache:
    data: Dataset
    catalogs: OrderedDict[tuple[Category | None, str], PolicyCatalog] = cache_field(
        default_factory=OrderedDict
    )
    claims: OrderedDict[tuple[Category | None, str, Citation], ClaimSet] = cache_field(
        default_factory=OrderedDict
    )
    passages: OrderedDict[tuple[Category | None, str, str], str] = cache_field(
        default_factory=OrderedDict
    )


_CACHES: weakref.WeakKeyDictionary[Retriever, _SourceCache] = weakref.WeakKeyDictionary()


def _cache(data: Dataset, retriever: Retriever) -> _SourceCache:
    cached = _CACHES.get(retriever)
    if cached is None or cached.data is not data:
        cached = _SourceCache(data)
        _CACHES[retriever] = cached
    return cached


def value_for(
    value: Value, units: tuple[Unit, ...], terms: tuple[Term, ...], origin: SourceSpan
) -> Value:
    if value.kind == "text":
        require(not value.unit, "text_unit")
        return Value("text", term(value.text, terms, origin))
    unit = " ".join(unicodedata.normalize("NFKC", value.unit).split())
    matches = {
        u.declaration.canonical
        for u in units
        if u.span.doc_ref == origin.doc_ref
        and unit in {u.declaration.canonical, *u.declaration.aliases}
    }
    require(len(matches) <= 1, "ambiguous_policy_unit")
    return Value(value.kind, value.text, next(iter(matches), value.unit)).canonical()


def span_for(data: Dataset, citation: Citation, start: int, end: int, snapshot: str) -> SourceSpan:
    policy = next(p for p in data.documents if p.metadata.ref == citation.doc_ref)
    quote = policy.text[start:end]
    require(citation.start_char <= start < end <= citation.end_char, "claim_span")
    return SourceSpan(
        citation.chunk_id,
        citation.index_sha256,
        citation.doc_ref,
        hashlib.sha256(policy.text.encode()).hexdigest(),
        start,
        end,
        quote,
        hashlib.sha256(quote.encode()).hexdigest(),
        snapshot,
        data.manifest.as_of,
        policy.metadata.kind.value,
    )


def lines(data: Dataset, citation: Citation, snapshot: str) -> tuple[tuple[str, SourceSpan], ...]:
    result = []
    start = citation.start_char
    for line in citation.quote.splitlines(keepends=True):
        text = line.strip()
        if text:
            result.append(
                (text, span_for(data, citation, start, start + len(line.rstrip("\r\n")), snapshot))
            )
        start += len(line)
    return tuple(result)


def term(text: str, terms: tuple[Term, ...], origin: SourceSpan) -> str:
    word = normalized(text)
    candidates = {
        normalized(t.declaration.canonical)
        for t in terms
        if t.span.doc_ref == origin.doc_ref
        and word
        in {normalized(t.declaration.canonical), *(normalized(a) for a in t.declaration.aliases)}
    }
    require(len(candidates) <= 1, "ambiguous_policy_term")
    return next(iter(candidates), word)


def checked_scope(
    scope: Scope, terms: tuple[Term, ...], origin: SourceSpan, units: tuple[Unit, ...] = ()
) -> Scope:
    require(type(scope) is Scope and len(scope.conditions) <= 16, "claim_scope")
    names = tuple(
        term(v, terms, origin)
        for v in (scope.subject, scope.action, scope.object, scope.scenario, scope.time)
    )
    require(all(v not in {"?", "unknown"} for v in names), "unknown_source_scope")
    for c in scope.conditions:
        normalized(c.field)
        require(type(c.operator) is Operator, "condition_operator")
        value_for(c.value, units, terms, origin)
    from knowledge_ticket_agent.evidence_contract import Constraint

    conditions = tuple(
        Constraint(
            term(c.field, terms, origin),
            c.operator,
            value_for(c.value, units, terms, origin),
        )
        for c in scope.conditions
    )
    require(compatible(conditions, ()) is True, "impossible_or_unknown_source_context")
    return Scope(
        names[0],
        names[1],
        names[2],
        names[3],
        names[4],
        tuple(sorted(conditions, key=lambda c: (c.field, c.operator, c.value.text, c.value.unit))),
    )


def resolve(reference: str, contexts: tuple[Context, ...], origin: SourceSpan) -> Context | None:
    reference = normalized(reference)
    # A scoped identifier is explicit; deictic references require a unique same-
    # document antecedent. They never erase conditions, subjects or time fields.
    name = re.sub(r"^(?:during\s+)?(?:the\s+)?same\s+|^(?:同一次|上述|同一场)", "", reference)
    deictic = reference in {"same", "同一次", "上述情况", "上述条件"}
    needs_antecedent = deictic or name != reference
    candidates = [
        c
        for c in contexts
        if c.span.doc_ref == origin.doc_ref
        and (
            deictic
            or c.name == reference
            or (name != reference and (c.name == name or c.name.endswith(" " + name)))
        )
    ]
    preceding = [c for c in candidates if c.span.start <= origin.start]
    if needs_antecedent:
        candidates = preceding
    elif preceding:
        candidates = preceding
    unique = {c.scope for c in candidates}
    return candidates[0] if len(unique) == 1 else None


def compile_catalog(
    data: Dataset,
    retriever: Retriever,
    *,
    category: Category | None = None,
    snapshot: str = "normal",
) -> PolicyCatalog:
    cached = _cache(data, retriever)
    key = (category, snapshot)
    if key not in cached.catalogs:
        cached.catalogs[key] = _compile_catalog(
            data, retriever, category=category, snapshot=snapshot
        )
        if len(cached.catalogs) > 16:
            cached.catalogs.popitem(last=False)
    cached.catalogs.move_to_end(key)
    return cached.catalogs[key]


def _compile_catalog(
    data: Dataset, retriever: Retriever, *, category: Category | None, snapshot: str
) -> PolicyCatalog:
    """Only source schema/context metadata. Never reads @claim values into a query plan."""
    scope = retriever.selection(category=category, snapshot=snapshot)
    contexts, pending, unknown = [], [], []
    raw_contexts = []
    raw_locations = []
    terms = []
    units = []
    for chunk in retriever.index.chunks:
        if chunk.doc_ref not in scope.authoritative_refs:
            continue
        citation = cite(retriever, chunk.chunk_id, category=category, snapshot=snapshot)
        verify_citation(citation, data, retriever, category=category, snapshot=snapshot)
        for text, span in lines(data, citation, snapshot):
            try:
                if text.startswith("@term "):
                    t = decode(TermDeclaration, parse_json(text[6:]))
                    require(len(t.aliases) <= 16, "term_capacity")
                    normalized(t.canonical)
                    for alias in t.aliases:
                        normalized(alias)
                    terms.append(Term(t, span))
                elif text.startswith("@unit "):
                    u = decode(UnitDeclaration, parse_json(text[6:]))
                    require(0 < len(u.aliases) <= 16, "unit_capacity")
                    normalized(u.canonical)
                    for alias in u.aliases:
                        normalized(alias)
                        canonical_unit, factor = unit_scale(u.canonical)
                        alias_unit, alias_factor = unit_scale(alias)
                        if not alias_unit.startswith("unit:"):
                            require(
                                (canonical_unit, factor) == (alias_unit, alias_factor),
                                "unit_alias_changes_scale",
                            )
                    units.append(Unit(u, span))
                elif text.startswith("@context "):
                    c = decode(ContextDeclaration, parse_json(text[9:]))
                    raw_contexts.append((c, span))
                elif text.startswith("@claim "):
                    # Index only read dependencies; never expose unseen values
                    # as SourceClaims or answers to query planning.
                    header = decode(ClaimDeclaration, parse_json(text[7:]))
                    raw_locations.append((header.context, header.key, span))
                elif text.startswith("@schema "):
                    declaration = decode(SchemaDeclaration, parse_json(text[8:]))
                    normalized(declaration.name)
                    for phrase in declaration.phrases:
                        normalized(phrase)
                    for bundle in declaration.alternatives:
                        for requirement in bundle:
                            normalized(requirement.key)
                    require(
                        0 < len(declaration.phrases) <= 16
                        and 0 < len(declaration.alternatives) <= 8
                        and all(0 < len(b) <= 16 for b in declaration.alternatives),
                        "schema_capacity",
                    )
                    require(
                        all(r.value is None for b in declaration.alternatives for r in b),
                        "schema_contains_answer_value",
                    )
                    pending.append((declaration, span))
            except (ValueError, TypeError, KeyError):
                unknown.append(span.chunk_id + ":invalid_metadata")
    for c, span in raw_contexts:
        try:
            contexts.append(
                Context(
                    normalized(c.name),
                    checked_scope(c.scope, tuple(terms), span, tuple(units)),
                    span,
                )
            )
        except ValueError:
            unknown.append(span.chunk_id + ":invalid_context")
    schemas = []
    for declaration, span in pending:
        context = resolve(declaration.context, tuple(contexts), span)
        if context is None:
            unknown.append(span.chunk_id + ":unknown_schema_context")
        else:
            declaration = replace(
                declaration,
                alternatives=tuple(
                    tuple(replace(r, key=term(r.key, tuple(terms), span)) for r in bundle)
                    for bundle in declaration.alternatives
                ),
            )
            schemas.append(PolicySchema(declaration, context, span))
    locations = []
    for reference, key, span in raw_locations:
        context = resolve(reference, tuple(contexts), span)
        if context is None:
            unknown.append(span.chunk_id + ":unknown_claim_context")
        else:
            locations.append(
                ClaimLocation(context.scope, term(key, tuple(terms), span), span.chunk_id)
            )
    require(len(contexts) <= 512 and len(schemas) <= 256, "catalog_capacity")
    return PolicyCatalog(
        tuple(contexts),
        tuple(terms),
        tuple(schemas),
        tuple(locations),
        retriever.index.sha256,
        snapshot,
        tuple(unknown),
        tuple(units),
    )


def compile_claims(
    data: Dataset,
    retriever: Retriever,
    citations: tuple[Citation, ...],
    catalog: PolicyCatalog,
    *,
    category: Category | None = None,
    snapshot: str = "normal",
) -> ClaimSet:
    claims: list[SourceClaim] = []
    unknown = list(catalog.unknown)
    require(
        catalog.index_sha256 == retriever.index.sha256 and catalog.snapshot == snapshot,
        "catalog_identity",
    )
    require(
        catalog is compile_catalog(data, retriever, category=category, snapshot=snapshot),
        "catalog_source_binding",
    )
    for citation in citations:
        verify_citation(citation, data, retriever, category=category, snapshot=snapshot)
        chunk = retriever.get_chunk(citation.chunk_id, category=category, snapshot=snapshot)
        require(
            citation.quote == chunk.text
            and citation.start_char == chunk.start_char
            and citation.end_char == chunk.end_char,
            "full_evidence_required",
        )
        cached = _cache(data, retriever)
        key = (category, snapshot, citation)
        if key in cached.claims:
            existing = cached.claims[key]
            cached.claims.move_to_end(key)
            claims.extend(existing.claims)
            unknown.extend(existing.unknown)
            continue
        first_claim, first_unknown = len(claims), len(unknown)
        for text, span in lines(data, citation, snapshot):
            try:
                if text.startswith("@claim "):
                    c = decode(ClaimDeclaration, parse_json(text[7:]))
                    context = resolve(c.context, catalog.contexts, span)
                    if context is None:
                        unknown.append(span.chunk_id + ":unknown_claim_context")
                        continue
                    require(
                        c.value.kind == "number" or c.operator in {Operator.EQ, Operator.NE},
                        "unordered_text_claim",
                    )
                    claims.append(
                        SourceClaim(
                            context.scope,
                            term(c.key, catalog.terms, span),
                            c.operator,
                            value_for(c.value, catalog.units, catalog.terms, span),
                            c.modality,
                            span,
                            context.span,
                            "source_assertion",
                            tuple(t.span for t in catalog.terms if t.span.doc_ref == span.doc_ref)
                            + tuple(
                                u.span for u in catalog.units if u.span.doc_ref == span.doc_ref
                            ),
                        )
                    )
                elif text.startswith(("@context ", "@schema ", "@term ", "@unit ", "## ")):
                    continue
                elif text.startswith("@note ") and not re.search(
                    r"\b(?:must|shall|may|requires?|required|prohibited)\b|必须|不得|禁止",
                    text,
                    flags=re.I,
                ):
                    continue
                else:
                    unknown.append(span.chunk_id + ":unparsed_source_statement")
            except (ValueError, TypeError, KeyError):
                unknown.append(span.chunk_id + ":invalid_claim")
        cached.claims[key] = ClaimSet(tuple(claims[first_claim:]), tuple(unknown[first_unknown:]))
        if len(cached.claims) > 512:
            cached.claims.popitem(last=False)
    require(len(claims) <= 512, "claim_capacity")
    return ClaimSet(tuple(claims), tuple(dict.fromkeys(unknown)))


def ranking_passages(
    data: Dataset,
    retriever: Retriever,
    catalog: PolicyCatalog,
    identifiers: tuple[str, ...],
    *,
    category: Category | None,
    snapshot: str,
) -> tuple[tuple[str, str], ...]:
    """Lossless assertion rendering for ranking only; never a full-read receipt.

    Original claims, values, operators and applicability remain in the text.
    Source schema JSON/alias lists are indexing metadata, not factual candidates.
    Unparsed body text is retained and still fails closed at proof construction.
    """
    cached = _cache(data, retriever)
    result = []
    for identifier in identifiers:
        key = (category, snapshot, identifier)
        if key not in cached.passages:
            chunk = retriever.get_chunk(identifier, category=category, snapshot=snapshot)
            require(
                chunk.doc_ref
                in retriever.selection(category=category, snapshot=snapshot).authoritative_refs,
                "ranking_view_authority",
            )
            parts = [chunk.title, chunk.heading]
            scopes = tuple(
                dict.fromkeys(
                    loc.scope for loc in catalog.claim_locations if loc.chunk_id == identifier
                )
            )
            for scope in scopes:
                parts.append(
                    " ".join(
                        (scope.subject, scope.action, scope.object, scope.scenario, scope.time)
                    )
                )
                parts.extend(
                    f"{c.field} {c.operator.value} {c.value.text} {c.value.unit}"
                    for c in scope.conditions
                )
            for line in chunk.text.splitlines():
                line = line.strip()
                if line.startswith("@claim "):
                    raw = decode(ClaimDeclaration, parse_json(line[7:]))
                    parts.append(
                        f"{raw.modality.value} {raw.key} {raw.operator.value} "
                        f"{raw.value.text} {raw.value.unit}"
                    )
                elif not line.startswith(("@context ", "@schema ", "@term ", "@unit ", "## ")):
                    parts.append(line)
            cached.passages[key] = "\n".join(parts)
            if len(cached.passages) > 512:
                cached.passages.popitem(last=False)
        cached.passages.move_to_end(key)
        result.append((identifier, cached.passages[key]))
    return tuple(result)
