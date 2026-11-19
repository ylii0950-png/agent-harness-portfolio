"""Bounded metadata-only business observations; never approval or commit authority."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditEvent:
    ordinal: int
    kind: str
    code: str
    ticket_id: str
    action_id: str = ""
    revision: int = 0


class BusinessAudit:
    """A run-local observation stream, distinct from the atomic commit journal."""

    def __init__(self) -> None:
        self._events: tuple[AuditEvent, ...] = ()

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return self._events

    def reserve(self, count: int) -> None:
        if len(self._events) + count > 256:
            raise ValueError("audit_capacity")

    def record(
        self, kind: str, code: str, ticket_id: str, action_id: str = "", revision: int = 0
    ) -> None:
        self.reserve(1)
        self._events += (
            AuditEvent(len(self._events) + 1, kind, code, ticket_id, action_id, revision),
        )
