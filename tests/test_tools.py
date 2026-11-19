import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.contracts import ParameterType, ToolCall, ToolDefinition, ToolParameter
from agent_harness_lab.tools import Argument, Permission, ToolRegistry, register_workspace_tools
from agent_harness_lab.workspace import WorkspaceFiles


@pytest.fixture
def configured() -> tuple[ToolRegistry, list[dict[str, Argument]]]:
    calls: list[dict[str, Argument]] = []
    registry = ToolRegistry()

    def handler(arguments: Mapping[str, Argument]) -> str:
        calls.append(dict(arguments))
        return "ok"

    registry.register(
        ToolDefinition(
            "check",
            "Check flat arguments.",
            (
                ToolParameter("text", ParameterType.STRING),
                ToolParameter("count", ParameterType.INTEGER),
                ToolParameter("flag", ParameterType.BOOLEAN, required=False),
            ),
        ),
        handler,
        permission=Permission.READ,
    )
    return registry, calls


def test_registration_lookup_catalog_and_optional_arguments(
    configured: tuple[ToolRegistry, list[dict[str, Argument]]],
) -> None:
    registry, calls = configured
    assert registry.names == ("check",)
    assert registry.get("missing") is None
    definition = registry.get("check")
    assert definition is not None
    schema = definition.json_schema()
    assert schema["additionalProperties"] is False and schema["required"] == ["text", "count"]
    schema["required"] = []
    assert definition.json_schema()["required"] == ["text", "count"]
    result = registry.invoke(ToolCall("one", "check", '{"text":"ok","count":2}'))
    assert result.ok and json.loads(result.output_json())["content"] == "ok"
    assert calls == [{"text": "ok", "count": 2}]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition, lambda arguments: "different", permission=Permission.WRITE)
    assert registry.get("check") == definition


@pytest.mark.parametrize(
    "arguments",
    [
        "{",
        "[]",
        "null",
        '"text"',
        '{"text":"a","text":"b","count":1}',
        '{"text":"x","count":NaN}',
        '{"text":"x","count":Infinity}',
        '{"text":"x","count":true}',
        '{"text":"x","count":1.0}',
        '{"text":null,"count":1}',
        '{"text":"x"}',
        '{"text":"x","count":1,"extra":2}',
        '{"text":"x","count":1,"flag":"true"}',
        '{"text":[],"count":1}',
        '{"text":"x","count":1,"flag":null}',
    ],
)
def test_illegal_arguments_never_reach_handler(
    configured: tuple[ToolRegistry, list[dict[str, Argument]]], arguments: str
) -> None:
    registry, calls = configured
    result = registry.invoke(ToolCall("bad", "check", arguments))
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_arguments" and result.error.recoverable
    assert calls == []


@pytest.mark.parametrize(
    "arguments",
    ['{"text":"' + ("x" * 1_048_576) + '","count":1}', "[" * 1100 + "]" * 1100, "\ud800"],
)
def test_oversized_deep_and_bad_unicode_arguments_fail_closed(
    configured: tuple[ToolRegistry, list[dict[str, Argument]]], arguments: str
) -> None:
    registry, calls = configured
    result = registry.invoke(ToolCall("bad", "check", arguments))
    assert result.error is not None and result.error.code == "invalid_arguments"
    assert calls == []


def test_permissions_snapshot_and_denial_before_handler() -> None:
    granted = {Permission.READ}
    registry = ToolRegistry(permissions=granted)
    granted.add(Permission.WRITE)
    calls: list[str] = []

    def write(arguments: Mapping[str, Argument]) -> str:
        calls.append("ran")
        return "ok"

    registry.register(ToolDefinition("write", "Write."), write, permission=Permission.WRITE)
    result = registry.invoke(ToolCall("one", "write", "{}"))
    assert result.error is not None and result.error.code == "permission_denied"
    assert calls == []


@pytest.mark.parametrize("permissions", ["write", ["write"], [1]])
def test_invalid_capability_configuration(permissions: Any) -> None:
    with pytest.raises(TypeError):
        ToolRegistry(permissions=permissions)


def test_unknown_tool_including_shell_name_is_structured_error() -> None:
    registry = ToolRegistry()
    result = registry.invoke(ToolCall("one", "bash", '{"command":"never executed"}'))
    assert result.error is not None and result.error.code == "unknown_tool"
    assert result.error.recoverable


def test_non_text_handler_output_is_fatal() -> None:
    def wrong(arguments: Mapping[str, Argument]) -> Any:
        return 3

    registry = ToolRegistry()
    registry.register(ToolDefinition("bad", "Bad output."), wrong, permission=Permission.READ)
    result = registry.invoke(ToolCall("one", "bad", "{}"))
    assert result.error is not None and not result.error.recoverable
    assert result.error.code == "invalid_tool_output"


@pytest.mark.parametrize("name", ["", "Bad", "with-dash", "a" * 65])
def test_invalid_definition_name(name: str) -> None:
    with pytest.raises(ValueError):
        ToolDefinition(name, "Description.")


def test_duplicate_parameter_and_mutable_specification_are_rejected() -> None:
    parameter = ToolParameter("text", ParameterType.STRING)
    with pytest.raises(ValueError):
        ToolDefinition("bad", "Duplicate.", (parameter, parameter))
    bad: Any = [parameter]
    with pytest.raises(TypeError):
        ToolDefinition("bad", "Mutable.", bad)


def test_workspace_wrappers_read_write_and_error_payload(tmp_path: Path) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
        register_workspace_tools(registry, workspace)
        assert registry.invoke(
            ToolCall("one", "write_file", '{"path":"file","content":"hello"}')
        ).ok
        assert registry.invoke(ToolCall("two", "read_file", '{"path":"file"}')).content == "hello"
        denied = registry.invoke(
            ToolCall("three", "write_file", '{"path":"file","content":"replaced"}')
        )
        assert denied.error is not None and denied.error.code == "workspace_already_exists"
        assert registry.invoke(
            ToolCall("four", "write_file", '{"path":"file","content":"new","overwrite":true}')
        ).ok
        assert workspace.read_text("file") == "new"
    closed = registry.invoke(ToolCall("five", "read_file", '{"path":"file"}'))
    assert (
        closed.error is not None
        and closed.error.code == "workspace_closed"
        and not closed.error.recoverable
    )


@pytest.mark.parametrize("argument", ['"true"', "1", "null"])
def test_write_flag_cannot_be_smuggled_through_json(tmp_path: Path, argument: str) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
        register_workspace_tools(registry, workspace)
        result = registry.invoke(
            ToolCall(
                "one",
                "write_file",
                '{"path":"file","content":"hello","overwrite":' + argument + "}",
            )
        )
    assert result.error is not None and result.error.code == "invalid_arguments"
    assert not (tmp_path / "file").exists()


def test_no_model_parameter_can_grant_workspace_capability(tmp_path: Path) -> None:
    with WorkspaceFiles(tmp_path, allow_write=True) as workspace:
        registry = ToolRegistry(permissions=(Permission.READ, Permission.WRITE))
        register_workspace_tools(registry, workspace)
        result = registry.invoke(
            ToolCall("one", "write_file", '{"path":"file","content":"hello","allow_write":true}')
        )
    assert result.error is not None and result.error.code == "invalid_arguments"
    assert not (tmp_path / "file").exists()
