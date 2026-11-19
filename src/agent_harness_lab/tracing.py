"""Metadata-only JSONL traces. Prompts, content, arguments, and tokens stay out."""

import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

_FIELDS = frozenset(
    {
        "estimated_before",
        "estimated_after",
        "compacted_results",
        "summary_chars",
        "reason",
        "round",
        "tool_attempt",
        "tool_name",
        "call_ref",
        "success",
        "error_code",
        "error_kind",
        "status",
        "request_id",
        "duration_ms",
        "stop_reason",
        "model_rounds",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "event_type",
        "item_count",
        "model",
        "task_id",
        "result",
        "content_bytes",
        "capability",
        "permission_decision",
        "authorized",
        "permission_reason",
        "task_count",
        "pending_count",
        "in_progress_count",
        "completed_count",
        "todo_ref",
        "previous_status",
        "todo_status",
        "operation",
        "run_id",
        "parent_run_id",
        "depth",
        "skill_ref",
        "total_model_calls",
        "total_tool_calls",
    }
)
_EVENTS = frozenset(
    {
        "context_budget_checked",
        "compaction_started",
        "compaction_completed",
        "compaction_failed",
        "run.started",
        "model.request",
        "model.event",
        "model.response",
        "model.error",
        "tool.call",
        "tool.result",
        "run.stopped",
        "eval.result",
        "todo.created",
        "todo.updated",
        "todo.queried",
        "todo.error",
        "subagent_started",
        "subagent_completed",
        "subagent_failed",
        "subagent_limit_reached",
        "skill_discovered",
        "skill_loaded",
        "skill_load_denied",
        "skill_load_failed",
    }
)


class TraceSink(Protocol):
    def record(self, event: str, **fields: object) -> None: ...


class JSONLTrace:
    """New files only, owner-only mode; I/O errors propagate to stop execution."""

    def __init__(self, path: Path) -> None:
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self._started = time.monotonic()
        self._sequence = 0
        self._bytes = 0

    def record(self, event: str, **fields: object) -> None:
        if event not in _EVENTS or fields.keys() - _FIELDS:
            raise ValueError("trace event/field is outside the metadata allowlist")
        safe: dict[str, object] = {}
        for name, value in fields.items():
            if value is None or type(value) in (int, float, bool):
                safe[name] = value
            elif isinstance(value, str):
                if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", value) or any(
                    word in value.lower() for word in ("bearer", "token=", "sk-", "eyj")
                ):
                    safe[name] = "redacted"
                else:
                    safe[name] = value
            else:
                raise ValueError("trace metadata must be scalar")
        self._sequence += 1
        row = {
            "schema_version": 1,
            "seq": self._sequence,
            "elapsed_ms": round((time.monotonic() - self._started) * 1000, 3),
            "event": event,
            **safe,
        }
        data = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        self._bytes += len(data)
        if self._bytes > 32 * 1024 * 1024:
            raise OSError("trace size limit exceeded")
        view = memoryview(data)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                raise OSError("trace write failed")
            view = view[written:]

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "JSONLTrace":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def emit(sink: TraceSink | None, event: str, fields: Mapping[str, object]) -> None:
    if sink is not None:
        sink.record(event, **fields)
