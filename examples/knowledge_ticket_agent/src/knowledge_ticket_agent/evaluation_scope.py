"""Public artificial scopes and explicit release-scoped trusted operator dispatch.

Default development execution remains DEV-only. Restricted scopes require the
operator context; trusted Python guards are not same-process security isolation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from knowledge_ticket_agent.schemas import Dataset, Split, Ticket

if TYPE_CHECKING:
    from knowledge_ticket_agent.formal_scope import SealedScope
from knowledge_ticket_agent.strict import require

_ISSUER = object()
PROBE_ID = "KT-E01"
PROBE_VERSION = "m5p-public-artificial-scope-1.0.0"
_SUBJECT = "M5-P ARTIFICIAL SCOPE PROBE; NOT A SEALED CASE"
_BODY = "Public fictional probe copied from DEV facts; no sealed query or Gold is used."


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, init=False)
class ArtificialCaseScope:
    ticket_id: str
    dataset_identity: str
    version: str
    _issuer: object
    source_ticket_id: str

    def __init__(
        self,
        ticket_id: str,
        dataset_identity: str,
        *,
        _issuer: object = None,
        source_ticket_id: str = "KT-D01",
    ) -> None:
        require(
            _issuer is _ISSUER and ticket_id in {f"KT-E{n:02d}" for n in range(1, 25)},
            "scope_issuer_disabled",
        )
        require(source_ticket_id in {f"KT-D{n:02d}" for n in range(1, 13)}, "scope_public_source")
        object.__setattr__(self, "source_ticket_id", source_ticket_id)
        object.__setattr__(self, "ticket_id", ticket_id)
        object.__setattr__(self, "dataset_identity", dataset_identity)
        object.__setattr__(self, "version", PROBE_VERSION)
        object.__setattr__(self, "_issuer", _issuer)

    @property
    def identity(self) -> str:
        return digest([self.version, self.ticket_id, self.dataset_identity, self.source_ticket_id])


def _probe(source: Ticket, ticket_id: str = PROBE_ID) -> Ticket:
    return replace(source, ticket_id=ticket_id, split=Split.EVAL, subject=_SUBJECT, body=_BODY)


def validate_scope(
    data: Dataset, ticket_id: str, scope: ArtificialCaseScope | SealedScope | None
) -> bool:
    if scope is None:
        return False
    from knowledge_ticket_agent.formal_scope import SealedScope, validate

    if type(scope) is SealedScope:
        return validate(data, ticket_id, scope)
    require(type(data) is Dataset, "scope_dataset_type")
    require(type(scope) is ArtificialCaseScope and scope._issuer is _ISSUER, "scope_not_issued")
    assert isinstance(scope, ArtificialCaseScope)
    require(
        scope.version == PROBE_VERSION
        and ticket_id == scope.ticket_id
        and ticket_id in {f"KT-E{n:02d}" for n in range(1, 25)}
        and scope.source_ticket_id in {f"KT-D{n:02d}" for n in range(1, 13)},
        "scope_target",
    )
    require(digest(asdict(data)) == scope.dataset_identity, "scope_dataset")
    sources = tuple(
        t for t in data.tickets if t.ticket_id == scope.source_ticket_id and t.split is Split.DEV
    )
    targets = tuple(t for t in data.tickets if t.ticket_id == ticket_id)
    require(len(sources) == len(targets) == 1, "scope_target_count")
    source, target = sources[0], targets[0]
    require(
        digest(asdict(target)) == digest(asdict(_probe(source, ticket_id))), "scope_artificial_only"
    )
    return True


def execution_identity(data: Dataset, scope: ArtificialCaseScope | SealedScope | None) -> str:
    if scope is None:
        return digest(asdict(data))  # Exactly the original DEV proposal binding.
    validate_scope(data, scope.ticket_id, scope)
    return digest({"dataset": asdict(data), "execution_scope": scope.identity})


@dataclass(frozen=True)
class ArtificialFixture:
    data: Dataset
    scope: ArtificialCaseScope


def public_eval_probe(
    data: Dataset, *, source_ticket_id: str = "KT-D01", target_ticket_id: str = PROBE_ID
) -> ArtificialFixture:
    """Select a public DEV clone and target slot; never issue scope for original EVAL content."""
    from knowledge_ticket_agent.rules import validate_rule_contract

    validate_rule_contract(data)
    require(source_ticket_id in {f"KT-D{n:02d}" for n in range(1, 13)}, "scope_public_source")
    source = next(
        t for t in data.tickets if t.ticket_id == source_ticket_id and t.split is Split.DEV
    )
    require(target_ticket_id in {f"KT-E{n:02d}" for n in range(1, 25)}, "scope_public_target")
    target = _probe(source, target_ticket_id)
    prepared = replace(
        data, tickets=tuple(target if t.ticket_id == target_ticket_id else t for t in data.tickets)
    )
    validate_rule_contract(prepared)
    require(digest(asdict(prepared)) != digest(asdict(data)), "scope_probe_not_distinct")
    scope = ArtificialCaseScope(
        target_ticket_id,
        digest(asdict(prepared)),
        _issuer=_ISSUER,
        source_ticket_id=source_ticket_id,
    )
    validate_scope(prepared, target_ticket_id, scope)
    return ArtificialFixture(prepared, scope)
