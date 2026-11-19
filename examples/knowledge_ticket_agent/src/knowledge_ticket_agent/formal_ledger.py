"""Owner-only persistent one-run ledger; not same-uid tamper resistance.

No reset or retry operation. A claim survives every terminal state and failure.
Only the trusted Operator chooses the fixed common-Git location; test callers
use unrelated synthetic directories.
"""

import fcntl
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from knowledge_ticket_agent.strict import FixtureError, parse_json, require

LIMIT = 4 * 1024 * 1024


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def secure_info(fd: int, *, directory: bool = False) -> None:
    info = os.fstat(fd)
    require(
        info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == (0o700 if directory else 0o600),
        "formal_permissions",
    )
    require(
        stat.S_ISDIR(info.st_mode)
        if directory
        else stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
        "formal_file_type",
    )


def read_at(fd: int, name: str) -> bytes:
    leaf = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(leaf, "rb") as source:
        secure_info(source.fileno())
        require(os.fstat(source.fileno()).st_size <= LIMIT, "formal_size")
        raw = source.read(LIMIT + 1)
        require(len(raw) <= LIMIT, "formal_size")
        return raw


def create_at(fd: int, name: str, raw: bytes) -> None:
    leaf = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    with os.fdopen(leaf, "wb") as target:
        secure_info(target.fileno())
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    os.fsync(fd)


class Ledger:
    def __init__(self, root: Path) -> None:
        require(all(not p.is_symlink() for p in (root, *root.parents)), "formal_symlink")
        root.mkdir(mode=0o700, exist_ok=True)
        self.root = root.absolute()
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            secure_info(directory, directory=True)
            info = os.fstat(directory)
            self.identity = (info.st_dev, info.st_ino)
            try:
                create_at(directory, "lock.bin", b"")
            except FileExistsError:
                pass
            lock = os.open("lock.bin", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
            try:
                secure_info(lock)
                info = os.fstat(lock)
                self.lock_identity = (info.st_dev, info.st_ino)
            finally:
                os.close(lock)
        finally:
            os.close(directory)

    @contextmanager
    def transaction(self) -> Iterator[int]:
        require(all(not p.is_symlink() for p in (self.root, *self.root.parents)), "formal_symlink")
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            secure_info(directory, directory=True)
            info = os.fstat(directory)
            require((info.st_dev, info.st_ino) == self.identity, "formal_directory_replaced")
            lock = os.open("lock.bin", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
            try:
                secure_info(lock)
                info = os.fstat(lock)
                require((info.st_dev, info.st_ino) == self.lock_identity, "formal_lock_replaced")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise FixtureError("formal_busy") from None
                yield directory
            finally:
                os.close(lock)
        finally:
            os.close(directory)

    def state(self, fd: int) -> dict[str, object]:
        try:
            raw = read_at(fd, "terminal.bin")
        except FileNotFoundError:
            try:
                raw = read_at(fd, "state.bin")
            except FileNotFoundError:
                raw = read_at(fd, "claim.bin")
        value = parse_json(raw.decode())
        require(isinstance(value, dict), "formal_state")
        assert isinstance(value, dict)
        require(value.get("status") in {"STARTED", "COMPLETED", "FAILED"}, "formal_state")
        claim_raw = read_at(fd, "claim.bin")
        claim = parse_json(claim_raw.decode())
        require(isinstance(claim, dict), "formal_claim")
        assert isinstance(claim, dict)
        require(all(value.get(k) == v for k, v in claim.items() if k != "status"), "formal_claim")
        if raw != claim_raw:
            require(value.get("claim_sha256") == digest(claim_raw), "formal_claim")
        else:
            value["claim_sha256"] = digest(claim_raw)
        return value

    def _state(self, fd: int, state: dict[str, object]) -> None:
        # A retained pending file blocks writes after interrupted publication.
        create_at(fd, "state-pending.bin", canonical(state))
        os.replace("state-pending.bin", "state.bin", src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)

    def start(self, fd: int, binding: dict[str, object]) -> dict[str, object]:
        # Any existing claim, even an empty/partial one, permanently consumes run.
        require(not any(n != "lock.bin" for n in os.listdir(fd)), "formal_already_claimed")
        claim = {
            "status": "STARTED",
            "run_id": str(uuid.uuid4()),
            "binding": binding,
            "started_utc": datetime.now(UTC).isoformat(),
        }
        raw = canonical(claim)
        create_at(fd, "claim.bin", raw)
        state: dict[str, object] = {**claim, "claim_sha256": digest(raw), "status": "STARTED"}
        self._state(fd, state)
        create_at(fd, "result-key.bin", os.urandom(32))
        return state

    def save(self, fd: int, value: dict[str, object]) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        state = self.state(fd)
        require(state["status"] == "STARTED", "formal_terminal")
        nonce = os.urandom(12)
        # Separate immutable encrypted records survive abrupt interruption.
        count = len([n for n in os.listdir(fd) if n.startswith("record-")])
        name = f"record-{count:03}.bin"
        raw = nonce + AESGCM(read_at(fd, "result-key.bin")).encrypt(
            nonce, canonical(value), canonical([state["run_id"], name])
        )
        create_at(fd, name, raw)

    def records(self, fd: int) -> list[dict[str, object]]:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        state = self.state(fd)
        aes = AESGCM(read_at(fd, "result-key.bin"))
        names = sorted(n for n in os.listdir(fd) if n.startswith("record-"))
        require(names == [f"record-{i:03}.bin" for i in range(len(names))], "formal_record_gap")
        values = []
        for name in names:
            raw = read_at(fd, name)
            require(len(raw) > 28, "formal_envelope")
            value = parse_json(
                aes.decrypt(raw[:12], raw[12:], canonical([state["run_id"], name])).decode()
            )
            require(isinstance(value, dict), "formal_record")
            assert isinstance(value, dict)
            values.append(value)
        return values

    def finish(
        self,
        fd: int,
        status: str,
        *,
        report_sha256: str | None = None,
        error_type: str | None = None,
    ) -> dict[str, object]:
        require(status in {"FAILED", "COMPLETED"}, "formal_status")
        state = self.state(fd)
        require(state["status"] == "STARTED", "formal_terminal")
        state.update(
            status=status,
            finished_utc=datetime.now(UTC).isoformat(),
            report_sha256=report_sha256,
            error_type=error_type,
        )
        # Durable terminal event precedes the replaceable status cache. Recovery
        # still sees FAILED/COMPLETED if an interrupted cache update left pending.
        create_at(fd, "terminal.bin", canonical(state))
        self._state(fd, state)
        return state

    def recovery_report(self, fd: int, report: dict[str, object]) -> str:
        """Retain a terminal reconstruction without modifying original records."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        state = self.state(fd)
        require(state["status"] in {"FAILED", "COMPLETED"}, "formal_recovery_state")
        aes = AESGCM(read_at(fd, "result-key.bin"))
        aad = canonical([state["run_id"], "recovery-report.bin"])
        plain = canonical(report)
        try:
            existing = read_at(fd, "recovery-report.bin")
        except FileNotFoundError:
            nonce = os.urandom(12)
            create_at(fd, "recovery-report.bin", nonce + aes.encrypt(nonce, plain, aad))
        else:
            require(
                aes.decrypt(existing[:12], existing[12:], aad) == plain, "formal_recovery_changed"
            )
        return digest(plain)
