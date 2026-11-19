"""Independent SIWC authorization. Never inspect another application's credentials."""

import argparse
import base64
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import tempfile
import time
import uuid
import webbrowser
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from agent_harness_lab.http_transport import HTTPTransport
from agent_harness_lab.integration import FailureKind, ProviderFailure, failure

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
JWKS = ISSUER + "/.well-known/jwks.json"
SCOPES = frozenset(
    {"openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"}
)


@dataclass(frozen=True, slots=True, repr=False)
class Session:
    client_id: str
    subject: str
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    id_token: str = field(repr=False)
    scopes: frozenset[str]
    expires_at: float

    def __repr__(self) -> str:
        return "Session(<protected>)"

    def require_plan(self) -> None:
        if not {"resource.invoke", "chatgpt.tokens.use.direct"} <= self.scopes:
            raise failure(FailureKind.AUTHORIZATION, "plan_scope_missing")


class CredentialStore:
    """Application-owned directory; atomic owner-only regular files, no symlinks."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or Path.home() / ".config/agent-harness-lab"
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError("credential directory must be private and owned by the current user")

    def _read(self, name: str) -> dict[str, Any] | None:
        path = self.directory / name
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ValueError("credential file must be private, singly linked, and regular")
            with os.fdopen(fd, "rb", closefd=False) as source:
                data = source.read(65537)
            if len(data) > 65536:
                raise ValueError("credential file is too large")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("invalid credential record")
            return result
        finally:
            os.close(fd)

    def _write(self, name: str, record: dict[str, Any]) -> None:
        path = self.directory / name
        if path.is_symlink():
            raise ValueError("credential destination must not be a symlink")
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as target:
                json.dump(record, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @contextmanager
    def locked(self) -> Iterator[None]:
        fd = os.open(
            self.directory / ".session.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ValueError("session lock must be a private regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def host_id(self) -> str:
        with self.locked():
            return self._host_id()

    def _host_id(self) -> str:
        record = self._read("host.json")
        if record is None:
            record = {"host_id": "urn:uuid:" + str(uuid.uuid4())}
            self._write("host.json", record)
        value = record.get("host_id")
        if not isinstance(value, str) or not value.startswith("urn:uuid:"):
            raise ValueError("invalid stored host identifier")
        uuid.UUID(value.removeprefix("urn:uuid:"), version=4)
        return value

    def load(self) -> Session | None:
        active = self._read("active.json")
        if active is None:
            return None
        client_id = active.get("client_id")
        if not isinstance(client_id, str):
            raise ValueError("invalid active registration")
        record = self._read(hashlib.sha256(client_id.encode()).hexdigest() + ".json")
        if record is None or record.get("client_id") != client_id:
            raise ValueError("missing selected registration")
        try:
            fields = [
                record[key]
                for key in ("client_id", "subject", "access_token", "refresh_token", "id_token")
            ]
            if (
                any(not isinstance(value, str) for value in fields)
                or not fields[0]
                or not fields[1]
            ):
                raise ValueError("invalid registration")
            if (
                not isinstance(record["scopes"], list)
                or any(not isinstance(s, str) for s in record["scopes"])
                or type(record["expires_at"]) not in (int, float)
                or not math.isfinite(record["expires_at"])
            ):
                raise ValueError("invalid session fields")
            if any(not value for value in fields[2:]) and not (
                all(value == "" for value in fields[2:])
                and record["scopes"] == []
                and record["expires_at"] == 0
            ):
                raise ValueError("partially missing credentials")
            return Session(
                fields[0],
                fields[1],
                fields[2],
                fields[3],
                fields[4],
                frozenset(record["scopes"]),
                record["expires_at"],
            )
        except (KeyError, TypeError):
            raise ValueError("invalid registration") from None

    def invalidate(self, session: Session) -> None:
        # Retain the verified registration mapping; discard all unusable credentials.
        self.save(Session(session.client_id, session.subject, "", "", "", frozenset(), 0))

    def save(self, session: Session) -> None:
        self._write(
            hashlib.sha256(session.client_id.encode()).hexdigest() + ".json",
            {
                "client_id": session.client_id,
                "subject": session.subject,
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "id_token": session.id_token,
                "scopes": sorted(session.scopes),
                "expires_at": session.expires_at,
            },
        )
        self._write("active.json", {"client_id": session.client_id})


@dataclass(frozen=True, slots=True, repr=False)
class AuthorizationAttempt:
    host_id: str
    redirect_uri: str
    client_id: str = "dynamic_agent_client"
    state: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(48))

    def url(self) -> str:
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self.verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        query = {
            "client_id": self.client_id,
            "ext_agent_host_id": self.host_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(sorted(SCOPES)),
            "resource": RESOURCE,
            "state": self.state,
            "nonce": self.nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if self.client_id == "dynamic_agent_client":
            query["agent_name_hint"] = "agent-harness-lab"
        return AUTHORIZE + "?" + urlencode(query)

    def callback(self, query: str) -> tuple[str, str]:
        values = parse_qs(query, keep_blank_values=True, max_num_fields=12)
        if any(len(value) != 1 for value in values.values()):
            raise failure(FailureKind.AUTHORIZATION, "invalid_callback")
        if not hmac.compare_digest(values.get("state", [""])[0], self.state):
            raise failure(FailureKind.AUTHORIZATION, "state_mismatch")
        if "error" in values:
            raise failure(FailureKind.AUTHORIZATION, "consent_denied")
        client = values.get("client_id", [self.client_id])[0]
        if (
            not client.startswith("oaiapp_")
            or len(client) > 256
            or (self.client_id != "dynamic_agent_client" and client != self.client_id)
        ):
            raise failure(FailureKind.AUTHORIZATION, "client_mismatch")
        code = values.get("code", [""])[0]
        if not code or len(code) > 4096:
            raise failure(FailureKind.AUTHORIZATION, "missing_code")
        return client, code


def validate_tokens(
    payload: dict[str, Any],
    client_id: str,
    nonce: str,
    jwks: dict[str, Any],
    *,
    previous: Session | None = None,
) -> Session:
    try:
        import jwt
    except ImportError:
        raise failure(FailureKind.AUTHORIZATION, "oauth_extra_required") from None

    try:
        id_token = payload["id_token"]
        header = jwt.get_unverified_header(id_token)
        if header.get("alg") != "RS256":
            raise ValueError("unexpected algorithm")
        raw_keys = jwks.get("keys")
        if not isinstance(raw_keys, list) or not isinstance(header.get("kid"), str):
            raise ValueError("invalid signing keys")
        keys = [
            key for key in raw_keys if isinstance(key, dict) and key.get("kid") == header["kid"]
        ]
        if len(keys) != 1:
            raise ValueError("unknown signing key")
        signing_key = jwt.PyJWK.from_dict(keys[0], algorithm="RS256").key
        claims = jwt.decode(
            id_token,
            signing_key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "aud", "iss", "nonce"]},
        )
        if not isinstance(claims["nonce"], str) or not hmac.compare_digest(claims["nonce"], nonce):
            raise ValueError("nonce mismatch")
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise ValueError("missing identity")
        if claims.get("azp") not in (None, client_id) or (
            isinstance(claims["aud"], list)
            and len(claims["aud"]) > 1
            and claims.get("azp") != client_id
        ):
            raise ValueError("authorized party mismatch")
        if previous is not None and (
            previous.client_id != client_id or previous.subject != claims["sub"]
        ):
            raise ValueError("account mismatch")
        return _session(payload, client_id, claims["sub"], id_token)
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise failure(FailureKind.AUTHORIZATION, "invalid_id_token") from None


def _session(payload: dict[str, Any], client: str, subject: str, id_token: str) -> Session:
    if (
        (
            not isinstance(payload.get("token_type"), str)
            or payload["token_type"].lower() != "bearer"
        )
        or type(payload.get("expires_in")) is not int
        or not 0 < payload["expires_in"] <= 86400
        or not isinstance(payload.get("scope"), str)
    ):
        raise ValueError("invalid token response")
    for key in ["access_token", "refresh_token"]:
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError("missing token")
    return Session(
        client,
        subject,
        payload["access_token"],
        payload["refresh_token"],
        id_token,
        frozenset(payload["scope"].split()),
        time.time() + payload["expires_in"],
    )


class OAuthConnection:
    def __init__(self, store: CredentialStore, transport: HTTPTransport | None = None) -> None:
        self.store = store
        self.transport = transport or HTTPTransport()

    def access_token(self) -> str:
        with self.store.locked():
            return self._access_token()

    def _access_token(self) -> str:
        # Reload under an exclusive file lock before any rotating-token refresh.
        session = self.store.load()
        if session is None:
            raise failure(FailureKind.AUTHORIZATION, "not_signed_in")
        session.require_plan()
        if session.expires_at <= time.time() + 60:
            try:
                payload = self.transport.json(
                    "POST",
                    TOKEN,
                    data=urlencode(
                        {
                            "grant_type": "refresh_token",
                            "client_id": session.client_id,
                            "refresh_token": session.refresh_token,
                            "resource": RESOURCE,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            except ProviderFailure as exc:
                if exc.info.code in {
                    "invalid_grant",
                    "invalid_refresh_token",
                    "token_expired",
                    "refresh_token_expired",
                    "refresh_token_invalidated",
                    "refresh_token_reused",
                }:
                    self.store.invalidate(session)
                raise
            try:
                renewed = _session(payload, session.client_id, session.subject, session.id_token)
            except (ValueError, TypeError, KeyError):
                raise failure(FailureKind.AUTHORIZATION, "invalid_refresh_response") from None
            self.store.save(renewed)
            renewed.require_plan()
            session = renewed
        return session.access_token

    def login(self, *, timeout: float = 300) -> None:
        previous = self.store.load()
        outcome: list[tuple[str, str] | ProviderFailure] = []
        attempt: AuthorizationAttempt

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # OAuth callback query contains credentials. Never log requests.

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                if parsed.path != "/auth/callback":
                    self.send_error(404)
                    return
                try:
                    result = attempt.callback(parsed.query)
                except (ValueError, ProviderFailure) as exc:
                    # Invalid unsolicited requests must not consume a pending attempt.
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Authorization callback rejected.")
                    if isinstance(exc, ProviderFailure) and exc.info.code == "consent_denied":
                        outcome.append(exc)
                    return
                if outcome:
                    self.send_error(409)
                    return
                outcome.append(result)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(
                    b"Authorization received. Return to agent-harness-lab for validation."
                )

        with HTTPServer(("127.0.0.1", 0), Callback) as server:
            server.timeout = 1
            attempt = AuthorizationAttempt(
                self.store.host_id(),
                f"http://127.0.0.1:{server.server_port}/auth/callback",
                previous.client_id if previous else "dynamic_agent_client",
            )
            if not webbrowser.open(attempt.url(), new=1):
                raise failure(FailureKind.AUTHORIZATION, "browser_unavailable")
            print(
                "Continue with ChatGPT: official browser opened; awaiting authorization.",
                flush=True,
            )
            deadline = time.monotonic() + timeout
            while not outcome and time.monotonic() < deadline:
                server.handle_request()
        if not outcome:
            raise failure(FailureKind.AUTHORIZATION, "authorization_timeout")
        if isinstance(outcome[0], ProviderFailure):
            raise outcome[0]
        client, code = outcome[0]
        payload = self.transport.json(
            "POST",
            TOKEN,
            data=urlencode(
                {
                    "grant_type": "authorization_code",
                    "client_id": client,
                    "code": code,
                    "code_verifier": attempt.verifier,
                    "redirect_uri": attempt.redirect_uri,
                    "resource": RESOURCE,
                }
            ).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        session = validate_tokens(
            payload, client, attempt.nonce, self.transport.json("GET", JWKS), previous=previous
        )
        self.store.save(session)
        session.require_plan()
        print(
            "OAuth identity and ChatGPT plan scope validated; credentials stored privately.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent Continue with ChatGPT authorization")
    parser.add_argument("action", choices=["login", "status"])
    arguments = parser.parse_args()
    try:
        store = CredentialStore()
        if arguments.action == "login":
            OAuthConnection(store).login()
        else:
            session = store.load()
            if session is None:
                print("Not signed in.")
            else:
                session.require_plan()
                print("Saved independent ChatGPT plan authorization; no credentials displayed.")
    except ProviderFailure as exc:
        print(str(exc))
        raise SystemExit(1) from None
    except (ValueError, OSError):
        print("Authorization/storage failed; no credentials displayed.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
