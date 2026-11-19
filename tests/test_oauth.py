import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from agent_harness_lab.integration import ProviderFailure
from agent_harness_lab.oauth import (
    ISSUER,
    SCOPES,
    AuthorizationAttempt,
    CredentialStore,
    Session,
    validate_tokens,
)

SignedTokens = tuple[dict[str, Any], dict[str, Any], RSAPrivateKey]


def test_pkce_dynamic_registration_and_stable_host(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "private")
    host = store.host_id()
    assert store.host_id() == host
    attempt = AuthorizationAttempt(host, "http://127.0.0.1:1234/auth/callback")
    query = parse_qs(urlsplit(attempt.url()).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["agent_name_hint"] == ["agent-harness-lab"]
    assert set(query["scope"][0].split()) == SCOPES
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(attempt.verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert query["code_challenge"] == [expected]
    assert query["code_challenge_method"] == ["S256"]
    assert attempt.state != AuthorizationAttempt(host, attempt.redirect_uri).state


@pytest.mark.parametrize(
    "values",
    [
        {"state": "wrong", "client_id": "oaiapp_new", "code": "code"},
        {"state": "good", "code": "code"},
        {"state": "good", "client_id": "dynamic_agent_client", "code": "code"},
        {"state": "good", "client_id": "oaiapp_new"},
        {"state": "good", "error": "access_denied"},
    ],
)
def test_invalid_callback_does_not_exchange_tokens(values: dict[str, str]) -> None:
    attempt = AuthorizationAttempt(
        "urn:uuid:test", "http://127.0.0.1:1/auth/callback", state="good"
    )
    with pytest.raises(ProviderFailure):
        attempt.callback(urlencode(values))


def test_callback_duplicate_state_and_returning_client_rejected() -> None:
    attempt = AuthorizationAttempt(
        "urn:uuid:test", "http://127.0.0.1:1/auth/callback", "oaiapp_original", state="good"
    )
    assert attempt.callback("state=good&code=test") == ("oaiapp_original", "test")
    with pytest.raises(ProviderFailure):
        attempt.callback("state=good&state=good&code=test")
    with pytest.raises(ProviderFailure):
        attempt.callback("state=good&code=test&client_id=oaiapp_different")
    assert "agent_name_hint" not in parse_qs(urlsplit(attempt.url()).query)


@pytest.fixture
def signed_tokens() -> SignedTokens:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk["kid"] = "test-key"
    claims = {
        "sub": "synthetic-subject",
        "iss": ISSUER,
        "aud": "oaiapp_test",
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        "nonce": "test-nonce",
    }
    token = jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test-key"})
    return (
        {
            "id_token": token,
            "access_token": "test-access",
            "refresh_token": "test-refresh",
            "scope": " ".join(SCOPES),
            "token_type": "Bearer",
            "expires_in": 3600,
        },
        {"keys": [jwk]},
        private,
    )


def test_signed_identity_and_scopes_are_validated(signed_tokens: SignedTokens) -> None:
    payload, jwks, _ = signed_tokens
    session = validate_tokens(payload, "oaiapp_test", "test-nonce", jwks)
    session.require_plan()
    assert session.subject == "synthetic-subject" and session.client_id == "oaiapp_test"
    assert "test-access" not in repr(session) and "test-refresh" not in repr(session)


@pytest.mark.parametrize(
    "change,value",
    [
        ("nonce", "wrong"),
        ("aud", "oaiapp_other"),
        ("iss", "https://invalid.example"),
        ("exp", 1),
        ("sub", ""),
    ],
)
def test_invalid_signed_claims_fail_closed(
    signed_tokens: SignedTokens, change: str, value: object
) -> None:
    payload, jwks, private = signed_tokens
    claims = jwt.decode(payload["id_token"], options={"verify_signature": False})
    claims[change] = value
    payload["id_token"] = jwt.encode(
        claims, private, algorithm="RS256", headers={"kid": "test-key"}
    )
    with pytest.raises(ProviderFailure):
        validate_tokens(payload, "oaiapp_test", "test-nonce", jwks)


def test_wrong_signing_key_and_algorithm_fail(signed_tokens: SignedTokens) -> None:
    payload, jwks, _ = signed_tokens
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    claims = jwt.decode(payload["id_token"], options={"verify_signature": False})
    payload["id_token"] = jwt.encode(claims, other, algorithm="RS256", headers={"kid": "test-key"})
    with pytest.raises(ProviderFailure):
        validate_tokens(payload, "oaiapp_test", "test-nonce", jwks)
    payload["id_token"] = jwt.encode(
        claims, "x" * 32, algorithm="HS256", headers={"kid": "test-key"}
    )
    with pytest.raises(ProviderFailure):
        validate_tokens(payload, "oaiapp_test", "test-nonce", jwks)


def test_identity_without_plan_scope_and_account_switch_fail(signed_tokens: SignedTokens) -> None:
    payload, jwks, _ = signed_tokens
    payload["scope"] = "openid profile email"
    session = validate_tokens(payload, "oaiapp_test", "test-nonce", jwks)
    with pytest.raises(ProviderFailure, match="plan_scope_missing"):
        session.require_plan()
    previous = Session("oaiapp_test", "different-subject", "old", "old", "old", SCOPES, 1)
    with pytest.raises(ProviderFailure):
        validate_tokens(payload, "oaiapp_test", "test-nonce", jwks, previous=previous)


def test_private_atomic_credentials_and_symlink_rejection(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "private")
    session = Session(
        "oaiapp_test", "subject", "test-access", "test-refresh", "test-id", SCOPES, 100
    )
    store.save(session)
    assert store.load() == session
    for path in store.directory.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    assert not list(store.directory.glob(".pending-*"))
    active = store.directory / "active.json"
    active.unlink()
    active.symlink_to(tmp_path / "outside")
    with pytest.raises((ValueError, OSError)):
        store.save(session)
    with pytest.raises((ValueError, OSError)):
        store.load()
    assert not (tmp_path / "outside").exists()


def test_per_registration_records_do_not_mix(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "private")
    first = Session("oaiapp_first", "one", "a", "b", "c", SCOPES, 100)
    second = Session("oaiapp_second", "two", "d", "e", "f", SCOPES, 100)
    store.save(first)
    store.save(second)
    assert store.load() == second
    assert len(list(store.directory.glob("*.json"))) == 3


def test_permissive_or_linked_credential_storage_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o755)
    with pytest.raises(ValueError):
        CredentialStore(directory)
    os.chmod(directory, 0o700)
    store = CredentialStore(directory)
    store.host_id()
    os.chmod(directory / "host.json", 0o644)
    with pytest.raises(ValueError):
        store.host_id()
