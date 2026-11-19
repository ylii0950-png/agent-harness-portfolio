"""Source-vocabulary request compiler. Produces obligations, never policy facts."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, replace

from knowledge_ticket_agent.evidence_contract import (
    Operator,
    QueryObligation,
    QueryObligations,
    Scope,
    Value,
    compatible,
    intersect_scope,
    normalized,
    scope_covered,
)
from knowledge_ticket_agent.policy_assertions import PolicyCatalog, PolicySchema, value_for
from knowledge_ticket_agent.strict import require

# These are public interface concept labels, not policy answers. Publishers can
# define *any* concept through schema labels and source-owned @term aliases.
REQUEST_LABELS = {
    "startup": (
        "startup conditions",
        "start conditions",
        "conditions for starting",
        "starting requirements",
        "启动条件",
        "开始条件",
    ),
    "completion": (
        "completion records",
        "closeout records",
        "records for completion",
        "完成记录",
        "完成后记录",
    ),
    "review": (
        "review role",
        "who signs",
        "who reviews",
        "reviewer",
        "复核签字主体",
        "由谁复核",
        "谁签字",
        "复核人",
    ),
}
_FUNCTIONS = re.compile(
    r"\b(?:please|tell|me|explain|describe|show|list|what|which|are|is|the|a|an|for|of|"
    r"to|need|know|how many)\b"
    r"|请|说明|告诉我|列出|展示|是什么|有哪些|需要|的",
    re.I,
)
_CONNECTORS = re.compile(r"\band\b|\balso\b|\bor\b|以及|或者|并且|和|与|及|或|[()（）,，;；]", re.I)


@dataclass(frozen=True)
class RequestLexeme:
    schema: PolicySchema
    entities: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class QueryPredicate:
    field: str
    operator: Operator
    value: Value
    contextual: bool
    span: tuple[int, int]


def _aliases(canonical: str, catalog: PolicyCatalog, ref: str) -> tuple[str, ...]:
    words = {normalized(canonical)}
    for t in catalog.terms:
        if t.span.doc_ref == ref and normalized(t.declaration.canonical) == normalized(canonical):
            words.update(normalized(a) for a in t.declaration.aliases)
    return tuple(sorted(words, key=lambda w: (-len(w), w)))


def _occurrences(text: str, label: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        m.span()
        for m in re.finditer(
            r"(?<![a-z0-9_])" + re.escape(label) + r"(?![a-z0-9_])", text, flags=re.I
        )
    )


def _erase(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _request_lexicon(catalog: PolicyCatalog) -> tuple[RequestLexeme, ...]:
    result = []
    for schema in catalog.schemas:
        entities = _aliases(schema.context.scope.object, catalog, schema.span.doc_ref)
        labels = set(_aliases(schema.declaration.name, catalog, schema.span.doc_ref))
        labels.update(REQUEST_LABELS.get(normalized(schema.declaration.name), ()))
        for phrase in schema.declaration.phrases:
            phrase = normalized(phrase)
            spans = [s for alias in entities for s in _occurrences(phrase, alias)]
            label = _erase(phrase, spans).strip(" 的:-")
            if label:
                labels.add(label)
        result.append(
            RequestLexeme(schema, entities, tuple(sorted(labels, key=lambda w: (-len(w), w))))
        )
    return tuple(result)


def _predicates(
    q: str, catalog: PolicyCatalog
) -> tuple[tuple[QueryPredicate, ...], list[tuple[int, int]], list[str]]:
    """Explicit comparisons are untrusted request constraints, never source facts."""
    from knowledge_ticket_agent.evidence_contract import negate

    fields = {
        r.key for schema in catalog.schemas for b in schema.declaration.alternatives for r in b
    }
    fields.update(c.field for context in catalog.contexts for c in context.scope.conditions)
    fields.update("scope." + k for k in ("subject", "action", "object", "scenario", "time"))
    candidates: dict[str, set[str]] = {}
    for field in fields:
        candidates.setdefault(normalized(field), set()).add(field)
        for schema in catalog.schemas:
            for alias in _aliases(field, catalog, schema.span.doc_ref):
                candidates.setdefault(alias, set()).add(field)
    labels = {
        alias: next(iter(options)) for alias, options in candidates.items() if len(options) == 1
    }
    operators = {
        "==": Operator.EQ,
        "=": Operator.EQ,
        "!=": Operator.NE,
        "≠": Operator.NE,
        "<=": Operator.LE,
        "≤": Operator.LE,
        ">=": Operator.GE,
        "≥": Operator.GE,
        "<": Operator.LT,
        ">": Operator.GT,
        "at least": Operator.GE,
        "at most": Operator.LE,
        "less than": Operator.LT,
        "greater than": Operator.GT,
        "equal to": Operator.EQ,
        "not equal to": Operator.NE,
        "等于": Operator.EQ,
        "不等于": Operator.NE,
        "不超过": Operator.LE,
        "不少于": Operator.GE,
        "小于": Operator.LT,
        "大于": Operator.GT,
    }
    operator_pattern = "|".join(re.escape(op) for op in sorted(operators, key=len, reverse=True))
    predicates, spans, unknown = [], [], []
    occupied: list[tuple[int, int]] = []
    number = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.I)
    for label in sorted(labels, key=lambda w: (-len(w), w)):
        for start, end in _occurrences(q, label):
            if any(start < b and end > a for a, b in occupied):
                continue
            found = re.match(
                r"\s*(not\s+|并非|不是)?(" + operator_pattern + r")\s*", q[end:], flags=re.I
            )
            if found is None:
                continue
            operator = operators[found[2].casefold()]
            if found[1]:
                operator = negate(operator)
            value_start = end + found.end()
            numeric = number.match(q, value_start)
            quoted = re.match(r"""["']([^"']+)["']""", q[value_start:])
            if numeric:
                finish = numeric.end()
                unit = ""
                tail = q[finish:].lstrip()
                declared_units = {
                    name
                    for u in catalog.units
                    for name in (u.declaration.canonical, *u.declaration.aliases)
                }
                declared = next(
                    (
                        name
                        for name in sorted(declared_units, key=lambda u: (-len(u), u))
                        if _occurrences(tail, name) and _occurrences(tail, name)[0][0] == 0
                    ),
                    None,
                )
                unit_match = re.match(
                    r"\s*([a-zµμ°%][a-z0-9_µμ°%-]*(?:[./:][a-z0-9_µμ°%-]+)*|[\u3400-\u9fff]+)",
                    q[finish:],
                    flags=re.I,
                )
                if declared is not None:
                    unit = tail[: len(declared)]
                    finish += len(q[finish:]) - len(tail) + len(declared)
                elif unit_match and unit_match[1].casefold() not in {
                    "and",
                    "or",
                    "not",
                    "where",
                    "if",
                    "when",
                    "以及",
                    "和",
                    "或",
                    "与",
                }:
                    unit = unit_match[1]
                    unit = re.split("以及|或者|并且|和|与|及|或", unit)[0]
                    finish += unit_match.start(1) + len(unit)
                value = Value("number", numeric[0], unit)
            elif quoted:
                value = Value("text", quoted[1])
                finish = value_start + quoted.end()
            else:
                unknown.append("unparsed_query_value")
                continue
            # A marked applicability clause ends at a semicolon or parenthesis.
            boundary = max((q.rfind(c, 0, start) for c in ";；()（）"), default=-1)
            marker = re.search(
                r"\b(?:where|when|if)\b|如果|当", q[boundary + 1 : start], flags=re.I
            )
            contextual = marker is not None
            if marker is not None:
                marker_start = boundary + 1 + marker.start()
                spans.append((marker_start, boundary + 1 + marker.end()))
                if re.search(r"\bor\b|或者|或", q[marker_start:start]):
                    unknown.append("unsupported_condition_disjunction")
            predicates.append(
                QueryPredicate(labels[label], operator, value, contextual, (start, finish))
            )
            spans.append((start, finish))
            occupied.append((start, finish))
    return tuple(predicates), spans, unknown


def _expression(tokens: list[str]) -> tuple[tuple[int, ...], ...]:
    """Bounded Boolean grammar, AND before OR, with explicit grouping."""
    position = 0

    def atom() -> list[tuple[int, ...]]:
        nonlocal position
        require(position < len(tokens), "query_expression")
        token = tokens[position]
        position += 1
        if token == "(":
            value = expression()
            require(position < len(tokens) and tokens[position] == ")", "query_expression")
            position += 1
            return value
        require(token.isdigit(), "query_expression")
        return [(int(token),)]

    def conjunction() -> list[tuple[int, ...]]:
        nonlocal position
        result = atom()
        while position < len(tokens) and tokens[position] not in {"OR", ")"}:
            if tokens[position] == "AND":
                position += 1
            other = atom()
            result = [tuple(dict.fromkeys((*a, *b))) for a in result for b in other]
            require(len(result) <= 64, "query_expression_capacity")
        return result

    def expression() -> list[tuple[int, ...]]:
        nonlocal position
        result = conjunction()
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            result.extend(conjunction())
            require(len(result) <= 64, "query_expression_capacity")
        return result

    result = expression()
    require(position == len(tokens), "query_expression")
    return tuple(result)


def _entity_group(
    q: str, left: int, right: int, entities: list[tuple[int, int, str]]
) -> tuple[tuple[str, ...], tuple[str, ...], list[tuple[int, int]], bool]:
    ordered = sorted(set(entities))

    def distance(e: tuple[int, int, str]) -> int:
        return max(left - e[1], e[0] - right, 0)

    nearest = min(distance(e) for e in ordered)
    seeds = [i for i, entity in enumerate(ordered) if distance(entity) == nearest]
    first = last = seeds[0]

    def connected(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
        if a[1] > b[0]:
            return False
        between = q[a[1] : b[0]]
        return bool(_CONNECTORS.search(between)) and not any(
            c.isalnum() for c in _FUNCTIONS.sub(" ", _CONNECTORS.sub(" ", between))
        )

    while first and connected(ordered[first - 1], ordered[first]):
        first -= 1
    while last + 1 < len(ordered) and connected(ordered[last], ordered[last + 1]):
        last += 1
    ranges, operators = [], []
    chosen_objects = {e[2] for e in ordered[first : last + 1]}
    ambiguous = any(ordered[seed][2] not in chosen_objects for seed in seeds)
    for a, b in zip(ordered[first:last], ordered[first + 1 : last + 1], strict=True):
        between = q[a[1] : b[0]]
        ambiguous |= any(c in between for c in "()（）")
        operators.append("OR" if re.search(r"\bor\b|或者|或", between, flags=re.I) else "AND")
        ranges.append((a[1], b[0]))
    return tuple(e[2] for e in ordered[first : last + 1]), tuple(operators), ranges, ambiguous


def plan_obligations(query: str, catalog: PolicyCatalog) -> QueryObligations:
    require(type(query) is str and len(query) <= 4000 and "\x00" not in query, "query")
    query.encode("utf-8")
    q = " ".join(unicodedata.normalize("NFKC", query).replace("\n", ";").split())
    lexicon = _request_lexicon(catalog)
    predicates, spans, unknown = _predicates(q, catalog)
    entities = [
        (
            start,
            end + (2 if q[end : end + 2].casefold() in {"'s", "’s"} else 0),
            item.schema.context.scope.object,
        )
        for item in lexicon
        for alias in item.entities
        for start, end in _occurrences(q, alias)
        if not any(start < p.span[1] and end > p.span[0] for p in predicates)
    ]
    # Prefer complete entity labels over nested substrings, keeping genuine
    # ambiguous identical labels visible rather than choosing one by score.
    entities = [
        e
        for e in set(entities)
        if not any(a <= e[0] and b >= e[1] and b - a > e[1] - e[0] for a, b, _ in entities)
    ]
    spans.extend((a, b) for a, b, _ in entities)
    masked = _erase(q, [(a, b) for a, b, _ in entities] + [p.span for p in predicates])
    mentions = [
        (a, b, item)
        for item in lexicon
        for label in item.labels
        for a, b in _occurrences(masked, label)
    ]
    mentions = [
        m
        for m in mentions
        if not any(a <= m[0] and b >= m[1] and b - a > m[1] - m[0] for a, b, _ in mentions)
    ]
    grouped: dict[tuple[int, int], list[RequestLexeme]] = {}
    for a, b, item in mentions:
        grouped.setdefault((a, b), []).append(item)
    obligations: list[QueryObligation] = []
    nodes: dict[int, str] = {}
    used_predicates: set[int] = set()
    used_entities: set[str] = set()
    for (a, b), items in sorted(grouped.items()):
        clause_start = max(q.rfind(";", 0, a), q.rfind("；", 0, a)) + 1
        endings = [p for p in (q.find(";", b), q.find("；", b)) if p >= 0]
        clause_end = min(endings) if endings else len(q)
        local_entities = [e for e in entities if clause_start <= e[0] < clause_end]
        available = local_entities or (entities if len({e[2] for e in entities}) == 1 else [])
        if not available:
            unknown.append("ambiguous_or_missing_query_entity")
            continue
        objects, operators, ranges, ambiguous = _entity_group(q, a, b, available)
        spans.extend(ranges)
        if ambiguous:
            unknown.append("ambiguous_query_entity")
        active_predicates = tuple(
            (n, p) for n, p in enumerate(predicates) if clause_start <= p.span[0] < clause_end
        )
        if active_predicates and len({e[2] for e in local_entities}) > 1 and len(objects) == 1:
            unknown.append("ambiguous_predicate_scope_use_separate_clauses")
        group_nodes = []
        for entity in objects:
            entity_nodes = []
            entity_items = [item for item in items if item.schema.context.scope.object == entity]
            if not entity_items:
                unknown.append("unknown_entity_obligation")
                continue
            used_entities.add(entity)
            if len({normalized(item.schema.declaration.name) for item in entity_items}) > 1:
                unknown.append("ambiguous_query_concept")
            candidates = []
            for item in entity_items:
                source_scope = item.schema.context.scope
                conditions = list(source_scope.conditions)
                overridden: set[str] = set()
                requested_names: dict[str, str] = {}
                names = {
                    k: getattr(source_scope, k)
                    for k in ("subject", "action", "object", "scenario", "time")
                }
                for n, predicate in active_predicates:
                    if not predicate.contextual:
                        continue
                    used_predicates.add(n)
                    value = value_for(
                        predicate.value, catalog.units, catalog.terms, item.schema.span
                    )
                    if predicate.field.startswith("scope."):
                        if predicate.operator is not Operator.EQ or value.kind != "text":
                            unknown.append("unsupported_categorical_condition")
                        else:
                            previous = requested_names.get(predicate.field)
                            if previous is not None and previous != value.text:
                                unknown.append("inconsistent_categorical_query_condition")
                            requested_names[predicate.field] = value.text
                            names[predicate.field[6:]] = value.text
                    else:
                        if predicate.field not in overridden:
                            conditions = [c for c in conditions if c.field != predicate.field]
                            overridden.add(predicate.field)
                        from knowledge_ticket_agent.evidence_contract import Constraint

                        conditions.append(Constraint(predicate.field, predicate.operator, value))
                scope = Scope(**names, conditions=tuple(conditions))
                if compatible(scope.conditions, ()) is not True:
                    unknown.append("impossible_query_conditions")
                    continue
                candidates.append((item, scope))
            planned: list[tuple[RequestLexeme, Scope, bool | None]] = []
            anchors = tuple(item.schema.context.scope for item, _ in candidates)
            for item, scope in candidates:
                coverage = scope_covered(scope, anchors)
                if coverage is not True:
                    planned.append((item, scope, coverage))
                intersection = intersect_scope(scope, item.schema.context.scope)
                if intersection is not None:
                    planned.append((item, intersection, True))
            for item, scope, covered in planned:
                alternatives = []
                for bundle in item.schema.declaration.alternatives:
                    requirements = []
                    for req in bundle:
                        matches = [
                            (n, p)
                            for n, p in active_predicates
                            if not p.contextual and normalized(p.field) == normalized(req.key)
                        ]
                        if matches:
                            for n, predicate in matches:
                                used_predicates.add(n)
                                requirements.append(
                                    replace(
                                        req,
                                        value=value_for(
                                            predicate.value,
                                            catalog.units,
                                            catalog.terms,
                                            item.schema.span,
                                        ),
                                        operator=predicate.operator,
                                    )
                                )
                        else:
                            requirements.append(req)
                    alternatives.append(tuple(requirements))
                identifier = (
                    item.schema.declaration.name
                    + ":"
                    + hashlib.sha256(
                        repr((scope, alternatives, item.schema.context.scope, covered)).encode()
                    ).hexdigest()[:12]
                )
                existing = next(
                    (i for i, o in enumerate(obligations) if o.obligation_id == identifier), None
                )
                if existing is None:
                    existing = len(obligations)
                    obligations.append(
                        QueryObligation(
                            identifier,
                            scope,
                            tuple(alternatives),
                            (a, b),
                            item.schema.context.scope,
                            covered,
                        )
                    )
                entity_nodes.append(str(existing))
            if entity_nodes:
                group_nodes.append("( " + " AND ".join(entity_nodes) + " )")
        if group_nodes:
            combined = group_nodes[0]
            for operator, node in zip(operators, group_nodes[1:], strict=False):
                combined += " " + operator + " " + node
            nodes[a] = "( " + combined + " )"
        spans.append((a, b))
    if {e[2] for e in entities} - used_entities:
        unknown.append("unbound_query_entity")
    if len(used_predicates) != len(predicates):
        unknown.append("unbound_query_predicate")
    remainder = _erase(q, spans)
    remainder = _FUNCTIONS.sub(lambda m: " " * len(m[0]), remainder)
    tokens: list[str] = []
    events = [(i, text) for i, text in nodes.items()]
    for found in _CONNECTORS.finditer(remainder):
        word = found[0].casefold()
        token = (
            "OR"
            if word in {"or", "或者", "或"}
            else "("
            if word in {"(", "（"}
            else ")"
            if word in {")", "）"}
            else "AND"
        )
        events.append((found.start(), token))
        remainder = remainder[: found.start()] + " " * len(word) + remainder[found.end() :]
    if any(c.isalnum() for c in remainder):
        unknown.append("unparsed_query_constraints:" + remainder.strip())
    for _, event in sorted(events):
        for token in event.split():
            if token == "AND" and (not tokens or tokens[-1] in {"AND", "OR", "("}):
                continue
            tokens.append(token)
    if tokens and tokens[-1] == "AND":
        tokens.pop()
    require(len(obligations) <= 16, "obligation_capacity")
    branches: tuple[tuple[int, ...], ...] = ()
    try:
        if tokens:
            branches = _expression(tokens)
    except ValueError:
        unknown.append("invalid_query_expression")
    if not obligations:
        unknown.append("unknown_query_obligation")
    return QueryObligations(query, tuple(obligations), tuple(dict.fromkeys(unknown)), branches)


def ranking_request(plan: QueryObligations) -> str:
    """Language-neutral request IR. It contains no retrieved answer values."""
    parts: list[str] = []
    for obligation in plan.obligations:
        scope = obligation.scope
        parts.extend((scope.subject, scope.action, scope.object, scope.scenario, scope.time))
        parts.extend(
            f"{c.field} {c.operator.value} {c.value.text} {c.value.unit}" for c in scope.conditions
        )
        for bundle in obligation.alternatives:
            for req in bundle:
                parts.append(f"{req.modality or 'describe'} {req.key}")
                if req.value is not None:
                    parts.append(f"{req.operator} {req.value.text} {req.value.unit}")
    # Repeated fields need one ranking representation; every obligation remains
    # independently present in the proof IR, including AND/OR and conditions.
    return " | ".join(dict.fromkeys(parts))
