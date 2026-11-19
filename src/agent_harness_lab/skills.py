"""Metadata-only local discovery and bounded lazy reads through WorkspaceFiles."""

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum

from agent_harness_lab.workspace import WorkspaceFiles

MAX_SKILLS = 64
MAX_DOCUMENT_BYTES = 65536
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class SkillErrorCode(StrEnum):
    INVALID_CONFIG = "invalid_config"
    INVALID_NAME = "invalid_name"
    INVALID_METADATA = "invalid_metadata"
    INVALID_DOCUMENT = "invalid_document"
    DUPLICATE_NAME = "duplicate_name"
    NOT_FOUND = "not_found"
    NOT_AUTHORIZED = "not_authorized"
    TOO_MANY = "too_many"
    CHANGED = "changed"


class SkillError(Exception):
    def __init__(self, code: SkillErrorCode) -> None:
        self.code = code
        super().__init__("skill operation rejected: " + code.value)


@dataclass(frozen=True, slots=True, repr=False)
class SkillInfo:
    name: str
    description: str
    entry: str

    def __repr__(self) -> str:
        return "SkillInfo(<protected>)"


def _scalar(value: str) -> str:
    """Restricted string frontmatter, not a YAML parser or executable expression."""
    try:
        if value.startswith('"'):
            result = json.loads(value)
            if type(result) is not str:
                raise ValueError("expected string")
            return result
        if value.startswith("'"):
            if not re.fullmatch(r"'(?:[^']|'')*'", value):
                raise ValueError("invalid quoted string")
            return value[1:-1].replace("''", "'")
        if (
            not value
            or value[0] in "[{&*!|>}`"
            or value.casefold() in {"true", "false", "null", "~"}
            or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value)
        ):
            raise ValueError("unsupported metadata")
        return value
    except (ValueError, TypeError, RecursionError):
        raise SkillError(SkillErrorCode.INVALID_METADATA) from None


def _metadata(header: str) -> tuple[str, str]:
    lines = header.splitlines()
    if len(lines) < 4 or lines[0] != "---" or lines[-1] != "---":
        raise SkillError(SkillErrorCode.INVALID_METADATA)
    fields: dict[str, str] = {}
    for line in lines[1:-1]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep or key not in {"name", "description"} or key in fields:
            raise SkillError(SkillErrorCode.INVALID_METADATA)
        fields[key] = _scalar(value.strip())
    name, description = fields.get("name", ""), fields.get("description", "")
    if (
        _NAME.fullmatch(name) is None
        or not description.strip()
        or not description.isprintable()
        or len(description.encode("utf-8")) > 256
    ):
        raise SkillError(SkillErrorCode.INVALID_METADATA)
    return name, description


def _document(text: str) -> tuple[tuple[str, str], str]:
    lines = text.splitlines(keepends=True)
    for index in range(1, len(lines)):
        if lines[index].rstrip("\r\n") == "---":
            header = "".join(lines[: index + 1])
            if len(header.encode("utf-8")) > 4096:
                raise SkillError(SkillErrorCode.INVALID_METADATA)
            body = "".join(lines[index + 1 :]).strip()
            if not body:
                raise SkillError(SkillErrorCode.INVALID_DOCUMENT)
            return _metadata(header), body
    raise SkillError(SkillErrorCode.INVALID_METADATA)


class SkillRegistry:
    """One immutable metadata snapshot, no cached bodies or shared loaded state.

    Configured directories must be inside the supplied trusted workspace. All
    names map to discovered entries rather than interpolating model paths.
    Reconstruct the registry to refresh metadata; full text is read on demand.
    """

    def __init__(
        self,
        workspace: WorkspaceFiles,
        directories: Collection[str],
        *,
        allowed_names: Collection[str] | None = None,
    ) -> None:
        if isinstance(directories, (str, bytes)) or not isinstance(directories, Collection):
            raise SkillError(SkillErrorCode.INVALID_CONFIG)
        roots = tuple(directories)
        if (
            not 1 <= len(roots) <= 8
            or any(type(path) is not str or not path for path in roots)
            or len(set(roots)) != len(roots)
        ):
            raise SkillError(SkillErrorCode.INVALID_CONFIG)
        if allowed_names is not None and (
            isinstance(allowed_names, (str, bytes))
            or not isinstance(allowed_names, Collection)
            or any(type(name) is not str or _NAME.fullmatch(name) is None for name in allowed_names)
        ):
            raise SkillError(SkillErrorCode.INVALID_CONFIG)
        self._workspace = workspace
        self._directories = roots
        self._allowed = None if allowed_names is None else frozenset(allowed_names)
        self._skills: tuple[SkillInfo, ...] | None = None

    def discover(self) -> tuple[SkillInfo, ...]:
        if self._skills is None:
            proposed: dict[str, SkillInfo] = {}
            count = 0
            for directory in self._directories:
                for folder in self._workspace.list_directories(directory):
                    count += 1
                    if count > MAX_SKILLS:
                        raise SkillError(SkillErrorCode.TOO_MANY)
                    if _NAME.fullmatch(folder) is None:
                        raise SkillError(SkillErrorCode.INVALID_NAME)
                    entry = directory + "/" + folder + "/SKILL.md"
                    name, description = _metadata(self._workspace.read_frontmatter(entry))
                    if name in proposed:
                        raise SkillError(SkillErrorCode.DUPLICATE_NAME)
                    proposed[name] = SkillInfo(name, description, entry)
            self._skills = tuple(
                proposed[name]
                for name in sorted(proposed)
                if self._allowed is None or name in self._allowed
            )
        return self._skills

    def load(self, name: str) -> str:
        if type(name) is not str or _NAME.fullmatch(name) is None:
            raise SkillError(SkillErrorCode.INVALID_NAME)
        if self._allowed is not None and name not in self._allowed:
            raise SkillError(SkillErrorCode.NOT_AUTHORIZED)
        info = next((skill for skill in self.discover() if skill.name == name), None)
        if info is None:
            raise SkillError(SkillErrorCode.NOT_FOUND)
        fields, body = _document(
            self._workspace.read_text(info.entry, max_bytes=MAX_DOCUMENT_BYTES)
        )
        if fields != (info.name, info.description):
            raise SkillError(SkillErrorCode.CHANGED)
        return body
