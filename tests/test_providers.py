from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from agent_harness_lab.providers import (
    Message,
    MockProvider,
    ModelProvider,
    ModelResponse,
    ProviderExhaustedError,
)


def test_provider_protocol_and_determinism() -> None:
    messages = [Message("user", "hello")]
    first: ModelProvider = MockProvider(["one", "two"])
    second: ModelProvider = MockProvider(["one", "two"])
    assert (
        [first.complete(messages), first.complete(messages)]
        == [
            second.complete(messages),
            second.complete(messages),
        ]
        == [ModelResponse("one"), ModelResponse("two")]
    )


def test_history_and_script_are_immutable_snapshots() -> None:
    script = ["original", "second"]
    provider = MockProvider(script)
    script[0] = "changed"
    messages = [Message("user", "original")]
    assert provider.complete(messages).content == "original"
    messages.append(Message("assistant", "added later"))
    history = provider.calls
    assert history == ((Message("user", "original"),),)
    provider.complete(messages)
    assert len(history) == 1
    assert len(provider.calls) == 2


@pytest.mark.parametrize("script", [[], ["only"]])
def test_exhaustion_is_explicit_and_does_not_record_a_call(script: list[str]) -> None:
    provider = MockProvider(script)
    messages = [Message("user", "request")]
    for _ in script:
        provider.complete(messages)
    for _ in range(2):
        with pytest.raises(ProviderExhaustedError, match="exhausted"):
            provider.complete(messages)
    assert len(provider.calls) == len(script)


@pytest.mark.parametrize("messages", [None, "hello", b"hello", ["hello"], [None]])
def test_invalid_messages_do_not_consume_response(messages: Any) -> None:
    provider = MockProvider(["still available"])
    with pytest.raises(TypeError):
        provider.complete(messages)
    assert provider.calls == ()
    assert provider.complete([Message("user", "valid")]).content == "still available"


def test_empty_messages_do_not_consume_response() -> None:
    provider = MockProvider([""])
    with pytest.raises(ValueError, match="at least one"):
        provider.complete([])
    assert provider.complete([Message("user", "")]).content == ""


@pytest.mark.parametrize("script", [None, "one", b"one", [1], [None]])
def test_invalid_response_scripts(script: Any) -> None:
    with pytest.raises(TypeError):
        MockProvider(script)


@pytest.mark.parametrize("role", ["tool", "developer", "", None])
def test_invalid_message_role(role: Any) -> None:
    with pytest.raises(ValueError):
        Message(role, "content")


@pytest.mark.parametrize("content", [None, 7, b"bytes"])
def test_non_text_message_and_response(content: Any) -> None:
    with pytest.raises(TypeError):
        Message("user", content)
    with pytest.raises(TypeError):
        ModelResponse(content)


def test_frozen_values() -> None:
    message = Message("user", "original")
    response = ModelResponse("original")
    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]  # Deliberately test frozen mutation.
    with pytest.raises(FrozenInstanceError):
        response.content = "changed"  # type: ignore[misc]  # Deliberately test frozen mutation.
