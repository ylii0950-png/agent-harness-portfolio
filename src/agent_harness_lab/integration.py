"""Sanitized integration failures; never expose HTTP bodies or credentials."""

import re
from dataclasses import dataclass
from enum import StrEnum


class FailureKind(StrEnum):
    AUTHORIZATION = "authorization"
    QUOTA = "quota"
    API = "api"
    PROTOCOL = "protocol"
    INTERRUPTED = "interrupted"
    REQUEST_BUDGET = "request_budget"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class FailureInfo:
    kind: FailureKind
    code: str
    status: int | None = None
    request_id: str | None = None


class ProviderFailure(Exception):
    def __init__(self, info: FailureInfo) -> None:
        self.info = info
        super().__init__(f"{info.kind.value}: {info.code}")


KNOWN_CODES = frozenset(
    {
        "subscription_sharing_user_not_eligible",
        "subscription_sharing_usage_limit_exceeded",
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_unsupported_capability",
        "subscription_sharing_route_not_supported",
        "subscription_sharing_invalid_user",
        "chatpass_v2_scope_not_authorized",
        "chatpass_v2_invalid_authorization_context",
        "subscription_sharing_user_unavailable",
        "invalid_grant",
        "invalid_client",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
        "access_denied",
        "insufficient_quota",
        "rate_limit_exceeded",
    }
)


def api_failure(status: int | None, body: object, request_id: str | None = None) -> ProviderFailure:
    code = "http_error"
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        candidate = body["error"].get("code") or body["error"].get("error")
        if isinstance(candidate, str) and candidate in KNOWN_CODES:
            code = candidate
    elif (
        isinstance(body, dict)
        and isinstance(body.get("error"), str)
        and body.get("error") in KNOWN_CODES
    ):
        code = body["error"]
    kind = FailureKind.API
    if status == 401 or code in {
        "invalid_grant",
        "invalid_client",
        "access_denied",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }:
        kind = FailureKind.AUTHORIZATION
    elif status == 429 or code in {
        "insufficient_quota",
        "rate_limit_exceeded",
        "subscription_sharing_usage_limit_exceeded",
    }:
        kind = FailureKind.QUOTA
    elif code in {
        "subscription_sharing_user_not_eligible",
        "subscription_sharing_invalid_user",
        "chatpass_v2_scope_not_authorized",
        "chatpass_v2_invalid_authorization_context",
    }:
        kind = FailureKind.AUTHORIZATION
    elif status == 403:
        kind = FailureKind.AUTHORIZATION
    if request_id is None or not re.fullmatch(r"req_[A-Za-z0-9_-]{1,100}", request_id):
        request_id = None
    return ProviderFailure(FailureInfo(kind, code, status, request_id))


def failure(kind: FailureKind, code: str) -> ProviderFailure:
    return ProviderFailure(FailureInfo(kind, code))


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in (self.input_tokens, self.output_tokens, self.total_tokens)
        ):
            raise ValueError("usage must contain nonnegative integers or unknown values")
