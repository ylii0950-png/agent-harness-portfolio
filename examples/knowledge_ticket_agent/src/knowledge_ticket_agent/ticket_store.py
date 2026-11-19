"""Run-local simulated states and an atomic in-memory commit journal.

This is trusted application code, not a sandbox against other Python code.
No account, payment, export, filesystem or network operation is implemented.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import RLock

from knowledge_ticket_agent.schemas import Status, Ticket

FaultHook = Callable[[str], None]


@dataclass(frozen=True)
class TicketState:
    ticket_id: str
    tenant_id: str
    status: Status
    revision: int
    last_action_id: str | None


@dataclass(frozen=True)
class Commit:
    action_id: str
    binding: str
    before: TicketState
    after: TicketState


@dataclass(frozen=True)
class StoreSnapshot:
    tickets: tuple[TicketState, ...]
    commits: tuple[Commit, ...] = ()


class TicketStore:
    def __init__(self, tickets: tuple[Ticket, ...]) -> None:
        if type(tickets) is not tuple or not 0 < len(tickets) <= 12:
            raise ValueError("store_tickets")
        states = tuple(
            TicketState(t.ticket_id, t.tenant_id, t.status, t.revision, t.last_action_id)
            for t in tickets
        )
        if len({s.ticket_id for s in states}) != len(states) or any(
            s.status is not Status.OPEN
            or type(s.revision) is not int
            or s.revision != 0
            or s.last_action_id is not None
            for s in states
        ):
            raise ValueError("store_initial_state")
        self._snapshot = StoreSnapshot(states)
        self._lock = RLock()

    @property
    def snapshot(self) -> StoreSnapshot:
        with self._lock:
            return self._snapshot

    def read(self, ticket_id: str) -> TicketState:
        with self._lock:
            return next(s for s in self._snapshot.tickets if s.ticket_id == ticket_id)

    def replay(self, action_id: str, binding: str) -> Commit | None:
        with self._lock:
            return next(
                (
                    c
                    for c in self._snapshot.commits
                    if c.action_id == action_id and c.binding == binding
                ),
                None,
            )

    def _commit(
        self,
        *,
        ticket_id: str,
        tenant_id: str,
        expected_revision: int,
        destination: Status,
        action_id: str,
        binding: str,
        fault: FaultHook | None = None,
    ) -> bool:
        """Only the authorized workflow handler calls this trusted primitive."""
        with self._lock:
            before = self.read(ticket_id)
            previous = next((c for c in self._snapshot.commits if c.action_id == action_id), None)
            if previous is not None:
                if (
                    previous.binding != binding
                    or previous.before.ticket_id != ticket_id
                    or previous.before.tenant_id != tenant_id
                    or previous.before.revision != expected_revision
                    or previous.after.status is not destination
                    or before != previous.after
                ):
                    raise ValueError("idempotency_mismatch")
                return False
            if (
                type(expected_revision) is not int
                or expected_revision != 0
                or before.revision != expected_revision
                or before.status is not Status.OPEN
                or before.tenant_id != tenant_id
                or type(destination) is not Status
                or destination not in {Status.RESOLVED, Status.INFORMATION, Status.REVIEW}
            ):
                raise ValueError("transition_invalid")
            if fault:
                fault("before_commit")
            after = replace(before, status=destination, revision=1, last_action_id=action_id)
            commit = Commit(action_id, binding, before, after)
            prepared = StoreSnapshot(
                tuple(after if s.ticket_id == ticket_id else s for s in self._snapshot.tickets),
                self._snapshot.commits + (commit,),
            )
            if fault:
                fault("audit_prepared")
                fault("before_publish")
            # One reference assignment publishes both the state and its commit evidence.
            self._snapshot = prepared
            if fault:
                fault("after_commit")
            return True
