"""Small strict dataclass decoder, not an executable rules or JSON Schema engine."""

import json
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

T = TypeVar("T")


class FixtureError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)  # No paths, input values or hidden answers in diagnostics.


def require(condition: bool, code: str) -> None:
    if not condition:
        raise FixtureError(code)


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _invalid_constant(_: str) -> Any:
    raise FixtureError("invalid_json_constant")


def parse_json(text: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, FixtureError):
            raise
        raise FixtureError("invalid_json") from None


def _value(kind: Any, raw: object, depth: int) -> Any:
    require(depth <= 24, "schema_depth")
    origin, args = get_origin(kind), get_args(kind)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _value(option, raw, depth + 1)
            except FixtureError:
                pass
        raise FixtureError("schema_type")
    if kind is type(None):
        require(raw is None, "schema_type")
        return None
    if kind in (str, int, bool):
        require(type(raw) is kind, "schema_type")
        if kind is str:
            require(len(cast(str, raw)) <= 32768, "string_limit")
            require("\x00" not in cast(str, raw), "invalid_text")
            try:
                cast(str, raw).encode("utf-8")
            except UnicodeError:
                raise FixtureError("invalid_text") from None
        return raw
    if origin is tuple:
        require(type(raw) is list and len(cast(list[object], raw)) <= 256, "schema_array")
        require(len(args) == 2 and args[1] is Ellipsis, "unsupported_schema")
        return tuple(_value(args[0], item, depth + 1) for item in cast(list[object], raw))
    if isinstance(kind, type) and issubclass(kind, Enum):
        require(type(raw) is str, "schema_type")
        try:
            return kind(raw)
        except ValueError:
            raise FixtureError("invalid_enum") from None
    if isinstance(kind, type) and is_dataclass(kind):
        require(type(raw) is dict, "schema_object")
        obj = cast(dict[str, object], raw)
        expected = {f.name for f in fields(kind)}
        require(obj.keys() == expected, "schema_fields")
        hints = get_type_hints(kind)
        return kind(**{key: _value(hints[key], value, depth + 1) for key, value in obj.items()})
    raise FixtureError("unsupported_schema")


def decode(model: type[T], raw: object) -> T:
    return cast(T, _value(model, raw, 0))
