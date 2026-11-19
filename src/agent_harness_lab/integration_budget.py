"""Fail-closed, durable inference-attempt ceiling shared by Parent and Child."""

import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

from agent_harness_lab.http_transport import HTTPTransport
from agent_harness_lab.integration import FailureKind, TokenUsage, failure

_IDS = frozenset({"A", "B", "C_off", "C_on", "D", "A_R", "B_R", "D_R", "A_S"})


class RequestLedger:
    """One exclusive suite journal; never silently reset or reuse an existing file.

    A reserved attempt is charged even if dispatch fails before reaching the API.
    This conservative count may exceed server-received requests, never undercount.
    """

    def __init__(self, path: Path, limit: int = 30) -> None:
        if type(limit) is not int or not 0 <= limit <= 30:
            raise ValueError("inference budget must be an integer from zero to 30")
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.limit = limit
        self.used = 0
        self.blocked: str | None = None
        self.records: list[dict[str, Any]] = []
        try:
            self._write({"event": "budget", "limit": limit})
        except BaseException:
            self.close()
            raise

    def _write(self, record: dict[str, Any]) -> None:
        payload = memoryview((json.dumps(record, sort_keys=True) + "\n").encode())
        while payload:
            written = os.write(self._fd, payload)
            if written <= 0:
                raise OSError("budget journal write failed")
            payload = payload[written:]
        os.fsync(self._fd)

    def reserve(self, case: str, role: str) -> dict[str, Any]:
        if case not in _IDS or role not in {"parent", "child"}:
            raise ValueError("unknown fixed scenario/role")
        if (
            case in {"A_R", "B_R", "D_R", "A_S"}
            and sum(r["case"] == case for r in self.records)
            >= {"A_R": 11, "B_R": 8, "D_R": 10, "A_S": 11}[case]
        ):
            raise failure(FailureKind.REQUEST_BUDGET, "scenario_budget_exhausted")
        if self.blocked is not None or self.used >= self.limit:
            raise failure(FailureKind.REQUEST_BUDGET, "request_budget_exhausted")
        row: dict[str, Any] = {
            "event": "attempt",
            "attempt": self.used + 1,
            "case": case,
            "role": role,
            "usage": None,
            "completed": False,
        }
        self._write(row)  # No HTTP dispatch unless the charge is durable.
        self.used += 1
        self.records.append(row)
        return row

    def complete(self, row: dict[str, Any]) -> None:
        self._write({**row, "event": "outcome"})

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "RequestLedger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class BudgetTransport:
    def __init__(
        self, ledger: RequestLedger, case: str, role: str, *, transport: HTTPTransport | None = None
    ) -> None:
        self.ledger, self.case, self.role = ledger, case, role
        self.transport = transport or HTTPTransport(timeout=30)

    def events(
        self, url: str, body: dict[str, object], token: str
    ) -> Generator[dict[str, Any], None, None]:
        if (
            url != "https://api.openai.com/v1/responses"
            or body.get("store") is not False
            or body.get("stream") is not True
        ):
            raise ValueError("only stateless streaming Responses OAuth route is allowed")
        row = self.ledger.reserve(self.case, self.role)
        try:
            for event in self.transport.events(url, body, token):
                if event.get("type") == "response.completed":
                    response = event.get("response", {})
                    raw = response.get("usage") if isinstance(response, dict) else None
                    if isinstance(raw, dict):
                        try:
                            usage = TokenUsage(
                                *(
                                    raw.get(k)
                                    for k in ("input_tokens", "output_tokens", "total_tokens")
                                )
                            )
                            row["usage"] = {
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "total_tokens": usage.total_tokens,
                            }
                        except ValueError:
                            pass  # Provider independently validates/rejects malformed usage.
                    row["completed"] = True
                yield event
        finally:
            self.ledger.complete(row)
