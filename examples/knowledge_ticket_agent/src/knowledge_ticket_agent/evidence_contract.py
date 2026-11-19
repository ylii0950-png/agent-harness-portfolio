"""Finite source-bound proof contracts. Queries state requirements, never facts."""

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum

from knowledge_ticket_agent.strict import require

VERSION = "lqd-domain-evidence-3.0.0"


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class Operator(StrEnum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class Modality(StrEnum):
    FACT = "fact"
    REQUIRED = "required"
    PERMITTED = "permitted"
    FORBIDDEN = "forbidden"


def normalized(text: str) -> str:
    require(type(text) is str and 0 < len(text) <= 512 and "\x00" not in text, "evidence_term")
    result = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    require(bool(result), "empty_evidence_term")
    return result


@dataclass(frozen=True)
class Value:
    kind: str
    text: str
    unit: str = ""

    def canonical(self) -> "Value":
        require(self.kind in {"text", "number"}, "evidence_value_kind")
        if self.kind == "text":
            require(not self.unit, "text_unit")
            return Value("text", normalized(self.text))
        require(type(self.text) is str and len(self.text) <= 64, "evidence_number_capacity")
        try:
            number = Decimal(self.text)
        except (InvalidOperation, ValueError):
            raise ValueError("evidence_number") from None
        require(number.is_finite(), "evidence_number")
        require(
            len(number.as_tuple().digits) <= 32 and -12 <= number.adjusted() <= 17,
            "evidence_number_capacity",
        )
        canonical, factor = unit_scale(self.unit)
        with localcontext() as context:
            context.prec = 64
            converted = (number * factor).normalize()
            require(
                len(converted.as_tuple().digits) <= 32 and -12 <= converted.adjusted() <= 17,
                "evidence_number_capacity",
            )
            return Value("number", str(converted), canonical)


def unit_scale(unit: str) -> tuple[str, Decimal]:
    """Standard dimensions or an exact nominal unit, never a guessed conversion.

    A publisher's domain unit is a type: quantities with that *same* symbol can
    be compared. Different nominal symbols are incomparable unless the source
    explicitly declares aliases. Unknown conversions are not missing facts.
    """
    require(
        type(unit) is str and len(unit) <= 69 and all(c.isprintable() for c in unit),
        "evidence_unit",
    )
    name = " ".join(unicodedata.normalize("NFKC", unit).split())
    groups = (
        ("s", "1", ("s", "sec", "second", "seconds", "秒")),
        ("s", "60", ("min", "minute", "minutes", "分钟")),
        ("s", "3600", ("h", "hr", "hour", "hours", "小时")),
        ("s", "86400", ("day", "days", "天")),
        ("m", "1", ("m", "metre", "metres", "meter", "meters", "米")),
        ("m", ".01", ("cm", "centimetre", "centimetres", "centimeter", "centimeters", "厘米")),
        ("m", ".001", ("mm", "millimetre", "millimetres", "millimeter", "millimeters", "毫米")),
        ("kg", "1", ("kg", "kilogram", "kilograms", "千克", "公斤")),
        ("kg", ".001", ("g", "gram", "grams", "克")),
        ("c", "1", ("c", "C", "°c", "°C", "celsius", "摄氏度")),
        ("%", "1", ("%", "percent", "百分比")),
    )
    for canonical, factor, aliases in groups:
        if name in aliases or (len(name) > 2 and name.casefold() in aliases):
            return canonical, Decimal(factor)
    # Namespace prevents a domain label from colliding with a physical base unit.
    if name.startswith("unit:"):
        require(len(name) > 5, "evidence_unit")
        return name, Decimal(1)
    require(len(name) <= 64, "evidence_unit")
    if not name:
        return name, Decimal(1)
    return "unit:" + name, Decimal(1)


@dataclass(frozen=True)
class Constraint:
    field: str
    operator: Operator
    value: Value


@dataclass(frozen=True)
class Scope:
    subject: str
    action: str
    object: str
    scenario: str
    time: str
    conditions: tuple[Constraint, ...] = ()


@dataclass(frozen=True)
class SourceSpan:
    chunk_id: str
    index_sha256: str
    doc_ref: str
    document_sha256: str
    start: int
    end: int
    quote: str
    quote_sha256: str
    snapshot: str
    as_of: str
    authority: str


@dataclass(frozen=True)
class FactRequirement:
    """None operator requests a constraint description; EQ requests a value."""

    key: str
    value: Value | None = None
    operator: Operator | None = None
    modality: Modality | None = Modality.REQUIRED
    functional: bool = False


@dataclass(frozen=True)
class QueryObligation:
    obligation_id: str
    scope: Scope
    alternatives: tuple[tuple[FactRequirement, ...], ...]
    request_span: tuple[int, int]
    anchor: Scope | None = None
    covered: bool | None = True


@dataclass(frozen=True)
class QueryObligations:
    query: str
    obligations: tuple[QueryObligation, ...]
    unknown: tuple[str, ...] = ()
    alternatives: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class SourceClaim:
    scope: Scope
    key: str
    operator: Operator
    value: Value
    modality: Modality
    span: SourceSpan
    context_span: SourceSpan
    basis: str = "source_assertion"
    definitions: tuple[SourceSpan, ...] = ()


@dataclass(frozen=True)
class EvidenceBinding:
    obligation_id: str
    alternative: int
    requirement: FactRequirement
    claims: tuple[SourceClaim, ...]
    qualifiers: tuple[SourceClaim, ...] = ()


@dataclass(frozen=True)
class Conflict:
    left: SourceClaim
    right: SourceClaim
    reason: str
    additional: tuple[SourceClaim, ...] = ()


@dataclass(frozen=True)
class EvidenceVerdict:
    status: Verdict
    missing: tuple[tuple[str, tuple[str, ...]], ...]
    bindings: tuple[EvidenceBinding, ...]
    conflicts: tuple[Conflict, ...]
    unknown: tuple[str, ...]
    method_identity: str
    retrieval_method: str
    version: str = VERSION

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _test(value: Decimal, operator: Operator, other: Decimal) -> bool:
    return {
        Operator.EQ: value == other,
        Operator.NE: value != other,
        Operator.LT: value < other,
        Operator.LE: value <= other,
        Operator.GT: value > other,
        Operator.GE: value >= other,
    }[operator]


def compatible(left: tuple[Constraint, ...], right: tuple[Constraint, ...]) -> bool | None:
    """Whether two finite conjunctions can both apply; missing information is unknown."""
    groups: dict[str, list[Constraint]] = {}
    for c in (*left, *right):
        require(type(c.operator) is Operator, "condition_operator")
        c = Constraint(normalized(c.field), c.operator, c.value.canonical())
        groups.setdefault(c.field, []).append(c)
    for constraints in groups.values():
        kinds = {(c.value.kind, c.value.unit) for c in constraints}
        if len(kinds) != 1:
            return None
        if constraints[0].value.kind == "text":
            if any(c.operator not in {Operator.EQ, Operator.NE} for c in constraints):
                return None
            equal = {c.value.text for c in constraints if c.operator is Operator.EQ}
            excluded = {c.value.text for c in constraints if c.operator is Operator.NE}
            if len(equal) > 1 or bool(equal & excluded):
                return False
        else:
            # A finite partition at all endpoints proves satisfiability for these
            # one-dimensional comparisons, including strict bounds and holes.
            points = sorted({Decimal(c.value.text) for c in constraints})
            with localcontext() as context:
                context.prec = 64
                trials = [
                    points[0] - 1,
                    points[-1] + 1,
                    *points,
                    *((a + b) / 2 for a, b in zip(points, points[1:], strict=False)),
                ]
            if not any(
                all(_test(v, c.operator, Decimal(c.value.text)) for c in constraints)
                for v in trials
            ):
                return False
    return True


def scope_relation(left: Scope, right: Scope) -> bool | None:
    for a, b in zip(
        (left.subject, left.action, left.object, left.scenario, left.time),
        (right.subject, right.action, right.object, right.scenario, right.time),
        strict=True,
    ):
        if a == "?" or b == "?":
            return None
        if normalized(a) != normalized(b) and a != "*" and b != "*":
            return False
    return compatible(left.conditions, right.conditions)


def intersect_scope(left: Scope, right: Scope) -> Scope | None:
    """Exact intersection of canonical scopes; callers check unknown relations first."""
    if scope_relation(left, right) is not True:
        return None
    names = tuple(
        b if a == "*" else a
        for a, b in zip(
            (left.subject, left.action, left.object, left.scenario, left.time),
            (right.subject, right.action, right.object, right.scenario, right.time),
            strict=True,
        )
    )
    return Scope(
        names[0],
        names[1],
        names[2],
        names[3],
        names[4],
        tuple(dict.fromkeys((*left.conditions, *right.conditions))),
    )


def scope_covered(request: Scope, alternatives: tuple[Scope, ...]) -> bool | None:
    """Prove Q implies a union of policy scopes by bounded exact complements."""
    if any(scope_entails(request, candidate) is True for candidate in alternatives):
        return True

    def constraints(scope: Scope) -> tuple[Constraint, ...]:
        values = tuple(
            (name, getattr(scope, name))
            for name in ("subject", "action", "object", "scenario", "time")
        )
        if any(value == "?" for _, value in values):
            raise ValueError("unknown_scope")
        return tuple(
            Constraint("dimension:" + name, Operator.EQ, Value("text", value))
            for name, value in values
            if value != "*"
        ) + tuple(Constraint("condition:" + c.field, c.operator, c.value) for c in scope.conditions)

    try:
        branches = [constraints(request)]
        if compatible(branches[0], ()) is not True:
            return None
        for scope in alternatives:
            condition = constraints(scope)
            if not condition:
                return True
            following = set()
            for branch in branches:
                for c in condition:
                    counterexample = (*branch, Constraint(c.field, negate(c.operator), c.value))
                    possible = compatible(counterexample, ())
                    if possible is None:
                        return None
                    if possible:
                        following.add(counterexample)
                    if len(following) > 128:
                        return None
            branches = sorted(following, key=repr)
            if not branches:
                return True
        return False
    except ValueError:
        return None


def negate(operator: Operator) -> Operator:
    return {
        Operator.EQ: Operator.NE,
        Operator.NE: Operator.EQ,
        Operator.LT: Operator.GE,
        Operator.LE: Operator.GT,
        Operator.GT: Operator.LE,
        Operator.GE: Operator.LT,
    }[operator]


def implies(left: tuple[Constraint, ...], right: tuple[Constraint, ...]) -> bool | None:
    for c in right:
        possible = compatible(left, (Constraint(c.field, negate(c.operator), c.value),))
        if possible is None:
            return None
        if possible:
            return False
    return True


def scope_entails(request: Scope, source: Scope) -> bool | None:
    for a, b in zip(
        (request.subject, request.action, request.object, request.scenario, request.time),
        (source.subject, source.action, source.object, source.scenario, source.time),
        strict=True,
    ):
        if a == "?" or b == "?":
            return None
        if b != "*" and normalized(a) != normalized(b):
            return False
    return implies(request.conditions, source.conditions)
