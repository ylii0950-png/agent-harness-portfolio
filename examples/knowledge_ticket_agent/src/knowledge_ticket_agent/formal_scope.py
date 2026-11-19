"""Release-scoped trusted operator context; never populated by model/tool arguments."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from knowledge_ticket_agent.evaluation_scope import digest
from knowledge_ticket_agent.schemas import Dataset
from knowledge_ticket_agent.strict import require

FORMAL_SUITE = "kt-m5-restricted-24-2.0.0"
_CONTEXT: ContextVar[tuple[str, str, str] | None] = ContextVar("m5_operator", default=None)
_ISSUER = object()


@contextmanager
def operator_scope(release: str, manifest: str, corpus: str) -> Iterator[None]:
    require(_CONTEXT.get() is None, "nested_formal_operator")
    token = _CONTEXT.set((release, manifest, corpus))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def validate_manifest_context(manifest: str, corpus: str) -> None:
    context = _CONTEXT.get()
    require(context is not None and context[1:] == (manifest, corpus), "formal_scope_not_issued")


@dataclass(frozen=True)
class SealedScope:
    ticket_id: str
    dataset_identity: str
    release_identity: str
    issuer: object

    @property
    def identity(self) -> str:
        return digest([self.release_identity, self.dataset_identity, self.ticket_id])


def issue(data: Dataset, ticket_id: str) -> SealedScope:
    from dataclasses import asdict

    context = _CONTEXT.get()
    require(context is not None and digest(asdict(data)) == context[2], "formal_dataset_binding")
    assert context is not None
    require(ticket_id in {f"KT-E{n:02d}" for n in range(1, 25)}, "formal_target")
    return SealedScope(ticket_id, context[2], context[0], _ISSUER)


def validate(data: Dataset, ticket_id: str, scope: SealedScope) -> bool:
    from dataclasses import asdict

    context = _CONTEXT.get()
    require(
        type(scope) is SealedScope
        and scope.issuer is _ISSUER
        and context is not None
        and scope.release_identity == context[0]
        and scope.dataset_identity == context[2] == digest(asdict(data))
        and scope.ticket_id == ticket_id,
        "formal_scope_binding",
    )
    return True
