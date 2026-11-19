"""Synchronous run-tree accounting and metadata correlation, not process isolation."""

from dataclasses import dataclass, field
from uuid import uuid4

from agent_harness_lab.tracing import TraceSink


@dataclass(frozen=True, slots=True)
class TotalLimits:
    max_model_calls: int = 10
    max_tool_calls: int = 20
    max_depth: int = 1

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 0
                for value in (self.max_model_calls, self.max_tool_calls, self.max_depth)
            )
            or self.max_depth > 1
        ):
            raise ValueError("nonnegative integer limits required; depth must be zero or one")


@dataclass(slots=True)
class RunBudget:
    limits: TotalLimits
    model_calls: int = 0
    tool_calls: int = 0
    _providers: list[object] = field(default_factory=list, repr=False)

    def claim_provider(self, provider: object) -> None:
        # Hold references for this run only, preventing identity reuse and shared history.
        if any(provider is previous for previous in self._providers):
            raise ValueError("each agent requires a fresh provider instance")
        self._providers.append(provider)

    @property
    def models_remaining(self) -> int:
        return self.limits.max_model_calls - self.model_calls

    def tools_fit(self, count: int) -> bool:
        return count <= self.limits.max_tool_calls - self.tool_calls


@dataclass(frozen=True, slots=True)
class ExecutionFrame:
    budget: RunBudget
    run_id: str = field(default_factory=lambda: "run_" + uuid4().hex)
    parent_run_id: str | None = None
    depth: int = 0

    def child(self) -> "ExecutionFrame":
        return ExecutionFrame(self.budget, parent_run_id=self.run_id, depth=self.depth + 1)


class RunTrace:
    """Nested wrappers preserve explicit child correlation rather than overwrite it."""

    def __init__(self, sink: TraceSink, frame: ExecutionFrame) -> None:
        self._sink = sink
        self._frame = frame

    def record(self, event: str, **fields: object) -> None:
        self._sink.record(
            event,
            **{
                "run_id": self._frame.run_id,
                "parent_run_id": self._frame.parent_run_id,
                "depth": self._frame.depth,
                **fields,
            },
        )
