"""Synchronous, fail-closed tool authorization; no filesystem or model dependency."""

import inspect
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from agent_harness_lab.contracts import ToolCall

Argument = str | int | bool


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionReason(StrEnum):
    ALLOWED = "allowed"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_REQUEST = "invalid_request"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_POLICY = "invalid_policy"
    CAPABILITY_DENIED = "capability_denied"
    POLICY_DENIED = "policy_denied"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_INVALID = "approval_invalid"
    APPROVAL_FAILED = "approval_failed"
    APPROVAL_INTERRUPTED = "approval_interrupted"
    APPROVED = "approved"


@dataclass(frozen=True, slots=True, repr=False)
class ApprovalRequest:
    """One validated call, with an immutable argument copy; never log its payload."""

    call_id: str
    tool_name: str
    permission: Permission
    arguments: Mapping[str, Argument]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def __repr__(self) -> str:
        return "ApprovalRequest(<protected>)"


class ApprovalHandler(Protocol):
    def __call__(self, request: ApprovalRequest) -> bool: ...


@dataclass(frozen=True, slots=True)
class Authorization:
    """Policy ASK is separate from the final permission to execute this call."""

    decision: Decision
    authorized: bool
    reason: PermissionReason


@dataclass(frozen=True, slots=True, init=False, repr=False)
class PermissionEngine:
    """Immutable capability ceiling + exact tool rules + per-call approval.

    Registered READ tools are allowed by default. WRITE requires explicit trusted
    capability opt-in, preserving the original Registry API. A rule may restrict
    a granted capability to DENY/ASK; it can never grant an absent capability.
    Any invalid configuration denies all operations. No wildcard/path policy,
    approval cache, automatic input(), shell, or process sandbox is provided.
    """

    _permissions: frozenset[Permission]
    _rules: Mapping[str, Decision]
    _approval: ApprovalHandler | None
    _valid: bool

    def __init__(
        self,
        *,
        permissions: Collection[Permission] = (Permission.READ,),
        rules: Mapping[str, Decision] | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        valid = True
        grants: frozenset[Permission] = frozenset()
        copied: dict[str, Decision] = {}
        try:
            if isinstance(permissions, (str, bytes)):
                raise ValueError("invalid capabilities")
            snapshot = tuple(permissions)
            if any(type(value) is not Permission for value in snapshot):
                raise ValueError("invalid capability")
            grants = frozenset(snapshot)
            if rules is not None:
                if not isinstance(rules, Mapping):
                    raise ValueError("invalid rules")
                copied = dict(rules)
                if any(
                    type(name) is not str
                    or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
                    or type(decision) is not Decision
                    for name, decision in copied.items()
                ):
                    raise ValueError("invalid rule")
            if approval_handler is not None and (
                not callable(approval_handler)
                or inspect.iscoroutinefunction(approval_handler)
                or inspect.iscoroutinefunction(approval_handler.__call__)
            ):
                raise ValueError("approval must be synchronous")
        except Exception:
            valid, grants, copied = False, frozenset(), {}
        object.__setattr__(self, "_permissions", grants)
        object.__setattr__(self, "_rules", MappingProxyType(copied))
        object.__setattr__(self, "_approval", approval_handler)
        object.__setattr__(self, "_valid", valid)

    def __repr__(self) -> str:
        return "PermissionEngine(<protected>)"

    def authorize(
        self,
        call: ToolCall,
        permission: Permission | None,
        arguments: Mapping[str, Argument] | None,
    ) -> Authorization:
        def deny(reason: PermissionReason, decision: Decision = Decision.DENY) -> Authorization:
            return Authorization(decision, False, reason)

        if not self._valid:
            return deny(PermissionReason.INVALID_POLICY)
        if type(call) is not ToolCall:
            return deny(PermissionReason.INVALID_REQUEST)
        try:
            call.validate()
        except (TypeError, ValueError):
            return deny(PermissionReason.INVALID_REQUEST)
        if permission is None:
            return deny(PermissionReason.UNKNOWN_TOOL)
        if type(permission) is not Permission:
            return deny(PermissionReason.INVALID_REQUEST)
        if permission not in self._permissions:
            return deny(PermissionReason.CAPABILITY_DENIED)
        if arguments is None:
            return deny(PermissionReason.INVALID_ARGUMENTS)
        decision = self._rules.get(call.name, Decision.ALLOW)
        if decision == Decision.DENY:
            return deny(PermissionReason.POLICY_DENIED)
        if decision == Decision.ALLOW:
            return Authorization(decision, True, PermissionReason.ALLOWED)
        if self._approval is None:
            return deny(PermissionReason.APPROVAL_UNAVAILABLE, Decision.ASK)
        try:
            request = ApprovalRequest(call.call_id, call.name, permission, arguments)
            reply: object = self._approval(request)
            # A coroutine returned by a sync wrapper must not be awaited or leaked.
            if inspect.iscoroutine(reply):
                reply.close()
            if type(reply) is not bool:
                return deny(PermissionReason.APPROVAL_INVALID, Decision.ASK)
            if reply is not True:
                return deny(PermissionReason.APPROVAL_DENIED, Decision.ASK)
        except (KeyboardInterrupt, SystemExit):
            return deny(PermissionReason.APPROVAL_INTERRUPTED, Decision.ASK)
        except Exception:
            return deny(PermissionReason.APPROVAL_FAILED, Decision.ASK)
        return Authorization(Decision.ASK, True, PermissionReason.APPROVED)
