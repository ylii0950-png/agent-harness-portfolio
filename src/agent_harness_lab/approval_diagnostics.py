"""Eval-only exact approval and private diagnostics, never model-visible policy."""

from dataclasses import dataclass, field
from enum import StrEnum

from agent_harness_lab.permissions import ApprovalRequest, Permission


class Mismatch(StrEnum):
    TOOL = "tool_mismatch"
    TARGET = "target_mismatch"
    CONTENT = "content_mismatch"
    OVERWRITE = "overwrite_mismatch"
    SCHEMA = "argument_schema_mismatch"
    SCOPE = "approval_scope_mismatch"
    UNAVAILABLE = "diagnostic_unavailable"


@dataclass(frozen=True, repr=False)
class ApprovalEvidence:
    request: ApprovalRequest
    approved: bool
    categories: tuple[Mismatch, ...]


@dataclass(repr=False)
class ExactWriteApproval:
    """One trusted fixture scope. Classification cannot affect the approval bool."""

    target: str
    content: str
    evidence: list[ApprovalEvidence] = field(default_factory=list, init=False)
    healthy: bool = field(default=True, init=False)

    def allowed(self, request: ApprovalRequest) -> bool:
        canonical = dict(request.arguments)
        if canonical.get("overwrite") is False:
            canonical.pop("overwrite")
        return (
            request.permission == Permission.WRITE
            and request.tool_name == "write_file"
            and canonical
            == {
                "path": self.target,
                "content": self.content,
            }
        )

    def classify(self, request: ApprovalRequest) -> tuple[Mismatch, ...]:
        args = request.arguments
        result = []
        if request.permission != Permission.WRITE:
            result.append(Mismatch.SCOPE)
        if request.tool_name != "write_file":
            result.append(Mismatch.TOOL)
        if set(args) - {"path", "content", "overwrite"} or not {"path", "content"} <= set(args):
            result.append(Mismatch.SCHEMA)
        if args.get("path") != self.target:
            result.append(Mismatch.TARGET)
        if args.get("content") != self.content:
            result.append(Mismatch.CONTENT)
        if args.get("overwrite", False) is not False:
            result.append(Mismatch.OVERWRITE)
        return tuple(result)

    def __call__(self, request: ApprovalRequest) -> bool:
        # Only this predicate decides. Errors in authorization itself fail closed.
        try:
            approved = self.allowed(request)
        except Exception:
            approved = False
        try:
            categories = self.classify(request)
            if type(categories) is not tuple or any(type(c) is not Mismatch for c in categories):
                raise ValueError("invalid diagnostic category")
            if Mismatch.UNAVAILABLE in categories:
                raise ValueError("diagnostic unavailable")
            if approved == bool(categories):
                raise ValueError("inconsistent diagnostic")
        except Exception:
            categories = (Mismatch.UNAVAILABLE,)
            self.healthy = False
        self.evidence.append(ApprovalEvidence(request, approved, categories))
        return approved

    def public_records(self) -> list[dict[str, object]]:
        # Trusted ordinals, no raw arguments/IDs, targets, content or bare payload hashes.
        return [
            {
                "approval_index": index,
                "approved": row.approved,
                "categories": [c.value for c in row.categories],
                "source": "scripted_fixture",
            }
            for index, row in enumerate(self.evidence, 1)
        ]
