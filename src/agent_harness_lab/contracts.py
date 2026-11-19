"""Immutable, provider-neutral tool records; no model SDK or network dependency."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum


class ParameterType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class ToolParameter:
    name: str
    kind: ParameterType
    required: bool = True


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise ValueError("invalid tool name")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must be nonempty text")
        if type(self.parameters) is not tuple:
            raise TypeError("tool parameters must be an immutable tuple")
        names = set()
        for parameter in self.parameters:
            if type(parameter) is not ToolParameter:
                raise TypeError("expected ToolParameter")
            if not isinstance(parameter.name, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}", parameter.name
            ):
                raise ValueError("invalid parameter name")
            if parameter.name in names:
                raise ValueError("duplicate parameter name")
            if (
                not isinstance(parameter.kind, ParameterType)
                or type(parameter.required) is not bool
            ):
                raise TypeError("invalid parameter specification")
            names.add(parameter.name)

    def json_schema(self) -> dict[str, object]:
        """Fresh flat-object schema. Optional arguments are omitted, not nullable."""
        return {
            "type": "object",
            "properties": {p.name: {"type": p.kind.value} for p in self.parameters},
            "required": [p.name for p in self.parameters if p.required],
            "additionalProperties": False,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str

    def validate(self) -> None:
        # Providers return untrusted data: runtime validation is intentional.
        if type(self.call_id) is not str or not self.call_id or len(self.call_id) > 128:
            raise ValueError("invalid tool call ID")
        if type(self.name) is not str or not self.name or len(self.name) > 64:
            raise ValueError("invalid tool call name")
        if type(self.arguments) is not str:
            raise ValueError("tool arguments must be JSON text")


@dataclass(frozen=True, slots=True)
class ToolError:
    code: str
    message: str
    recoverable: bool


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    content: str = ""
    error: ToolError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def output_json(self) -> str:
        """Structured text payload for a future provider adapter and the demo."""
        error = self.error
        return json.dumps(
            {
                "ok": self.ok,
                "content": self.content,
                "error": None
                if error is None
                else {
                    "code": error.code,
                    "message": error.message,
                    "recoverable": error.recoverable,
                },
            },
            sort_keys=True,
        )
