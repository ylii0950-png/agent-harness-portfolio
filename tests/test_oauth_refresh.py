import time
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.integration import FailureInfo, FailureKind, ProviderFailure
from agent_harness_lab.oauth import SCOPES, CredentialStore, OAuthConnection, Session


class RefreshTransport:
    def __init__(self, payload: dict[str, Any] | ProviderFailure) -> None:
        self.payload = payload
        self.calls: list[bytes] = []

    def json(
        self, method: str, url: str, *, data: bytes | None = None, headers: Any = None
    ) -> dict[str, Any]:
        assert method == "POST" and url == "https://auth.openai.com/api/accounts/oauth/token"
        assert data is not None
        self.calls.append(data)
        if isinstance(self.payload, ProviderFailure):
            raise self.payload
        return self.payload


def test_refresh_rotates_tokens_atomically_and_subsequent_call_reuses_grant(tmp_path: Path) -> None:
    from typing import cast

    from agent_harness_lab.http_transport import HTTPTransport

    store = CredentialStore(tmp_path / "private")
    store.save(Session("oaiapp_test", "subject", "expired-access", "old-refresh", "id", SCOPES, 1))
    transport = RefreshTransport(
        {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "scope": " ".join(SCOPES),
            "token_type": "Bearer",
            "expires_in": 3600,
        }
    )
    connection = OAuthConnection(store, cast(HTTPTransport, transport))
    assert connection.access_token() == "new-access"
    assert connection.access_token() == "new-access" and len(transport.calls) == 1
    saved = store.load()
    assert saved is not None and saved.refresh_token == "new-refresh"
    assert b"scope=" not in transport.calls[0] and b"client_id=oaiapp_test" in transport.calls[0]


def test_unexpired_grant_never_refreshes_or_reads_external_credentials(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "private")
    store.save(
        Session(
            "oaiapp_test", "subject", "still-valid", "refresh", "id", SCOPES, time.time() + 3600
        )
    )
    assert OAuthConnection(store).access_token() == "still-valid"


def test_temporary_refresh_failure_preserves_credentials(tmp_path: Path) -> None:
    from typing import cast

    from agent_harness_lab.http_transport import HTTPTransport

    store = CredentialStore(tmp_path / "private")
    session = Session("oaiapp_test", "subject", "expired", "refresh", "id", SCOPES, 1)
    store.save(session)
    transport = RefreshTransport(
        ProviderFailure(FailureInfo(FailureKind.API, "subscription_sharing_usage_unavailable", 503))
    )
    with pytest.raises(ProviderFailure):
        OAuthConnection(store, cast(HTTPTransport, transport)).access_token()
    assert store.load() == session


def test_refresh_scope_downgrade_never_enables_inference(tmp_path: Path) -> None:
    from typing import cast

    from agent_harness_lab.http_transport import HTTPTransport

    store = CredentialStore(tmp_path / "private")
    store.save(Session("oaiapp_test", "subject", "expired", "refresh", "id", SCOPES, 1))
    transport = RefreshTransport(
        {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "scope": "openid",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
    )
    with pytest.raises(ProviderFailure):
        OAuthConnection(store, cast(HTTPTransport, transport)).access_token()


@pytest.mark.parametrize(
    "code",
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    ],
)
def test_terminal_refresh_failure_clears_tokens_but_preserves_registration(
    tmp_path: Path, code: str
) -> None:
    from typing import cast

    from agent_harness_lab.http_transport import HTTPTransport
    from agent_harness_lab.integration import api_failure

    store = CredentialStore(tmp_path / "private")
    store.save(Session("oaiapp_test", "subject", "expired", "refresh", "id", SCOPES, 1))
    error = api_failure(400, {"error": code})
    assert error.info.kind == FailureKind.AUTHORIZATION
    transport = RefreshTransport(error)
    with pytest.raises(ProviderFailure):
        OAuthConnection(store, cast(HTTPTransport, transport)).access_token()
    saved = store.load()
    assert saved is not None and saved.client_id == "oaiapp_test" and saved.subject == "subject"
    assert not saved.access_token and not saved.refresh_token and not saved.id_token
    with pytest.raises(ProviderFailure, match="plan_scope_missing"):
        OAuthConnection(store, cast(HTTPTransport, transport)).access_token()
    assert len(transport.calls) == 1


def test_scope_downgrade_persists_rotated_credentials_before_denial(tmp_path: Path) -> None:
    from typing import cast

    from agent_harness_lab.http_transport import HTTPTransport

    store = CredentialStore(tmp_path / "private")
    store.save(Session("oaiapp_test", "subject", "expired", "old-refresh", "id", SCOPES, 1))
    transport = RefreshTransport(
        {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "scope": "openid",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
    )
    with pytest.raises(ProviderFailure, match="plan_scope_missing"):
        OAuthConnection(store, cast(HTTPTransport, transport)).access_token()
    saved = store.load()
    assert (
        saved is not None
        and saved.refresh_token == "new-refresh"
        and saved.scopes == frozenset({"openid"})
    )
    with pytest.raises(ProviderFailure):
        OAuthConnection(store, cast(HTTPTransport, transport)).access_token()
    assert len(transport.calls) == 1


def test_partially_cleared_credentials_are_not_a_valid_invalidation_record(tmp_path: Path) -> None:
    import hashlib
    import json

    store = CredentialStore(tmp_path / "private")
    session = Session("oaiapp_test", "subject", "access", "refresh", "id", SCOPES, 1)
    store.save(session)
    path = store.directory / (hashlib.sha256(session.client_id.encode()).hexdigest() + ".json")
    record = json.loads(path.read_text())
    record["access_token"] = ""
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="partially missing"):
        store.load()


def test_sse_rate_limit_code_is_classified_as_quota_even_without_http_status() -> None:
    from agent_harness_lab.integration import api_failure

    result = api_failure(None, {"error": {"code": "rate_limit_exceeded"}})
    assert result.info.kind == FailureKind.QUOTA and result.info.status is None
