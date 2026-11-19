"""Encrypted local custody, not same-user role isolation.

Requires the existing locked development cryptography dependency only when
opening/sealing private entries. No OAuth or other credential store is used.
"""

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from knowledge_ticket_agent.evaluation_maintenance import VERSION
from knowledge_ticket_agent.strict import FixtureError, parse_json, require

MAX_BYTES = 600 * 1024


def _file_permissions(fd: int) -> None:
    info = os.fstat(fd)
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o600,
        "vault_file_permissions",
    )


def _name(name: str) -> str:
    require(bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,63}\.(?:aesgcm|bin|jsonl)", name)), "vault_name")
    return name


def _actor(actor: str) -> None:
    require(bool(re.fullmatch(r"[a-z0-9_-]{1,64}", actor)), "audit_actor")


def _read(directory: int, name: str) -> bytes:
    try:
        fd = os.open(_name(name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as source:
            _file_permissions(source.fileno())
            require(os.fstat(source.fileno()).st_size <= MAX_BYTES, "vault_size")
            raw = source.read(MAX_BYTES + 1)
            require(len(raw) <= MAX_BYTES, "vault_size")
            return raw
    except OSError:
        raise FixtureError("vault_read") from None


def _write_new(directory: int, name: str, raw: bytes) -> None:
    require(len(raw) <= MAX_BYTES, "vault_size")
    try:
        fd = os.open(
            _name(name),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(fd, "wb") as target:
            _file_permissions(target.fileno())
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        os.fsync(directory)
    except OSError:
        raise FixtureError("vault_write_new") from None


def _chain(raw: bytes) -> list[dict[str, object]]:
    require(not raw or raw.endswith(b"\n"), "audit_partial_tail")
    previous = "0" * 64
    records = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            text = line.decode("utf-8")
        except UnicodeError:
            raise FixtureError("audit_encoding") from None
        value = parse_json(text)
        require(isinstance(value, dict), "audit_shape")
        assert isinstance(value, dict)
        require(
            value.keys()
            == {
                "sequence",
                "previous",
                "time_utc",
                "action",
                "actor",
                "artifact",
                "ciphertext_sha256",
            }
            and type(value.get("sequence")) is int
            and all(isinstance(value[k], str) for k in value if k != "sequence"),
            "audit_shape",
        )
        require(
            value.get("sequence") == number and value.get("previous") == previous, "audit_chain"
        )
        previous = hashlib.sha256(line).hexdigest()
        records.append(value)
    return records


def _append(directory: int, action: str, actor: str, artifact: str, digest: str) -> None:
    _actor(actor)
    _name(artifact)
    require(
        action
        in {
            "read",
            "import",
            "seal_begin",
            "seal_complete",
            "seal_failed",
            "review",
            "recover",
            "validate",
        },
        "audit_action",
    )
    require(bool(re.fullmatch(r"[a-f0-9]{64}", digest)), "audit_digest")
    # The enclosing vault lock serializes creation, validation and appending.
    try:
        fd = os.open(
            "access.jsonl",
            os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(fd, "a+b") as target:
            _file_permissions(target.fileno())
            require(os.fstat(target.fileno()).st_size <= MAX_BYTES, "audit_size")
            target.seek(0)
            existing = target.read(MAX_BYTES + 1)
            records = _chain(existing)
            previous = (
                hashlib.sha256(existing.splitlines()[-1]).hexdigest() if records else "0" * 64
            )
            record = {
                "sequence": len(records) + 1,
                "previous": previous,
                "time_utc": datetime.now(UTC).isoformat(),
                "action": action,
                "actor": actor,
                "artifact": artifact,
                "ciphertext_sha256": digest,
            }
            line = (json.dumps(record, sort_keys=True) + "\n").encode()
            require(len(existing) + len(line) <= MAX_BYTES, "audit_size")
            target.write(line)
            target.flush()
            os.fsync(target.fileno())
        os.fsync(directory)
    except OSError:
        raise FixtureError("audit_write") from None


class LocalVault:
    def __init__(self, root: Path) -> None:
        path = root.absolute()
        require(
            path.name == "m2e-steward-private"
            and path.parent.name == ".git"
            and (path.parent / "HEAD").is_file()
            and (path.parent / "objects").is_dir(),
            "vault_location",
        )
        require(all(not p.is_symlink() for p in (path, *path.parents)), "vault_symlink")
        require(path.is_dir(), "vault_missing")
        info = path.stat()
        require(
            info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700, "vault_permissions"
        )
        self.root = path
        self._identity = (info.st_dev, info.st_ino)
        # Initialize once before workers start. Transactions never create or
        # replace the lock inode; a missing/replaced lock is a custody failure.
        directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(directory)
            require(
                (opened.st_dev, opened.st_ino) == self._identity
                and opened.st_uid == os.getuid()
                and stat.S_IMODE(opened.st_mode) == 0o700,
                "vault_permissions",
            )
            try:
                lock = os.open(
                    "auditlock.bin",
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                lock = os.open("auditlock.bin", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
            try:
                _file_permissions(lock)
                locked_info = os.fstat(lock)
                self._lock_identity = (locked_info.st_dev, locked_info.st_ino)
            finally:
                os.close(lock)
        except OSError:
            raise FixtureError("vault_lock_initialize") from None
        finally:
            os.close(directory)

    @contextmanager
    def _transaction(self) -> Iterator[int]:
        # Pin the original directory inode and use only dir_fd leaf operations.
        require(all(not p.is_symlink() for p in (self.root, *self.root.parents)), "vault_symlink")
        try:
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:
            raise FixtureError("vault_directory") from None
        try:
            info = os.fstat(directory)
            require(
                (info.st_dev, info.st_ino) == self._identity
                and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o700,
                "vault_permissions",
            )
            lock = os.open(
                "auditlock.bin",
                os.O_RDWR | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            try:
                _file_permissions(lock)
                locked_info = os.fstat(lock)
                require(
                    (locked_info.st_dev, locked_info.st_ino) == self._lock_identity,
                    "vault_lock_replaced",
                )
                fcntl.flock(lock, fcntl.LOCK_EX)
                yield directory
            finally:
                os.close(lock)
        except OSError:
            raise FixtureError("vault_transaction") from None
        finally:
            os.close(directory)

    def open(self, name: str, *, actor: str) -> bytes:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        _actor(actor)
        _name(name)
        require(name.endswith(".aesgcm"), "vault_entry")
        with self._transaction() as directory:
            key, raw = _read(directory, "key.bin"), _read(directory, name)
            require(len(key) == 32 and len(raw) > 28, "vault_envelope")
            try:
                plaintext = AESGCM(key).decrypt(
                    raw[:12], raw[12:], VERSION.encode() + b"\0" + name.encode()
                )
            except (InvalidTag, ValueError):
                raise FixtureError("vault_authentication") from None
            _append(directory, "read", actor, name, hashlib.sha256(raw).hexdigest())
            return plaintext

    def seal(self, name: str, raw: bytes, *, actor: str) -> str:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        _actor(actor)
        _name(name)
        require(name.endswith(".aesgcm") and 0 < len(raw) <= 512 * 1024, "vault_entry")
        with self._transaction() as directory:
            try:
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FixtureError("vault_entry_exists")
            key = _read(directory, "key.bin")
            require(len(key) == 32, "vault_envelope")
            nonce = os.urandom(12)
            encrypted = nonce + AESGCM(key).encrypt(
                nonce, raw, VERSION.encode() + b"\0" + name.encode()
            )
            digest = hashlib.sha256(encrypted).hexdigest()
            _append(directory, "seal_begin", actor, name, digest)
            try:
                _write_new(directory, name, encrypted)
            except FixtureError:
                _append(directory, "seal_failed", actor, name, digest)
                raise
            # A crash/failure here leaves a visible pending intent. Never silently
            # treat the file as frozen. Inspect and retain journal before recovery.
            _append(directory, "seal_complete", actor, name, digest)
            return digest

    def log(self, action: str, *, actor: str, artifact: str, digest: str) -> None:
        with self._transaction() as directory:
            _append(directory, action, actor, artifact, digest)

    def audit_status(self) -> dict[str, object]:
        with self._transaction() as directory:
            raw = _read(directory, "access.jsonl")
            records = _chain(raw)
            pending: dict[str, object] = {}
            for record in records:
                name = record["artifact"]
                assert isinstance(name, str)
                if record["action"] == "seal_begin":
                    pending[name] = record["ciphertext_sha256"]
                elif record["action"] in {"seal_complete", "seal_failed"}:
                    require(pending.get(name) == record["ciphertext_sha256"], "audit_seal_pair")
                    if record["action"] == "seal_complete":
                        del pending[name]
            return {
                "events": len(records),
                "journal_sha256": hashlib.sha256(raw).hexdigest(),
                "pending_seals": sorted(pending),
                "independent_access_isolation": "NOT_ESTABLISHED",
            }
