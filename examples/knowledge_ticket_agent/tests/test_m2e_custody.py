import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from knowledge_ticket_agent.evaluation_custody import LocalVault
from knowledge_ticket_agent.strict import FixtureError


@pytest.fixture
def vault(tmp_path: Path) -> LocalVault:
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/test\n")
    (git / "objects").mkdir()
    root = git / "m2e-steward-private"
    root.mkdir(mode=0o700)
    key = root / "key.bin"
    key.write_bytes(os.urandom(32))
    key.chmod(0o600)
    return LocalVault(root)


def test_encryption_and_audit_redaction(vault: LocalVault) -> None:
    assert (vault.root / "auditlock.bin").is_file()
    raw = b'{"test_only_secret":"fixture content not heldout"}'
    digest = vault.seal("test.aesgcm", raw, actor="test_steward")
    assert len(digest) == 64 and raw not in (vault.root / "test.aesgcm").read_bytes()
    assert vault.open("test.aesgcm", actor="test_reviewer") == raw
    journal = (vault.root / "access.jsonl").read_text()
    assert "test_only_secret" not in journal and "fixture content" not in journal
    assert vault.audit_status()["pending_seals"] == []
    assert vault.audit_status()["events"] == 3


def test_name_authenticated_not_swappable(vault: LocalVault) -> None:
    vault.seal("one.aesgcm", b"one", actor="test")
    copied = vault.root / "two.aesgcm"
    copied.write_bytes((vault.root / "one.aesgcm").read_bytes())
    copied.chmod(0o600)
    with pytest.raises(FixtureError, match="vault_authentication"):
        vault.open("two.aesgcm", actor="test")


@pytest.mark.parametrize("mode", [0o644, 0o666, 0o400])
def test_key_permissions_fail_closed(vault: LocalVault, mode: int) -> None:
    (vault.root / "key.bin").chmod(mode)
    with pytest.raises(FixtureError):
        vault.seal("test.aesgcm", b"data", actor="test")
    assert not (vault.root / "test.aesgcm").exists()


@pytest.mark.parametrize(
    "name",
    ["../outside.aesgcm", "/tmp/outside.aesgcm", "UPPER.aesgcm", "bad/name.aesgcm", "key.bin"],
)
def test_illegal_entry_rejected(vault: LocalVault, name: str) -> None:
    with pytest.raises(FixtureError):
        vault.seal(name, b"data", actor="test")


@pytest.mark.parametrize("raw", [b"", b"a" * (512 * 1024 + 1)])
def test_payload_bounds(vault: LocalVault, raw: bytes) -> None:
    with pytest.raises(FixtureError):
        vault.seal("test.aesgcm", raw, actor="test")
    assert not (vault.root / "test.aesgcm").exists()


def test_never_overwrites(vault: LocalVault) -> None:
    vault.seal("test.aesgcm", b"original", actor="test")
    before = (vault.root / "test.aesgcm").read_bytes()
    with pytest.raises(FixtureError):
        vault.seal("test.aesgcm", b"changed", actor="test")
    assert (vault.root / "test.aesgcm").read_bytes() == before


def test_invalid_actor_prevents_seal(vault: LocalVault) -> None:
    with pytest.raises(FixtureError):
        vault.seal("test.aesgcm", b"data", actor="../invalid")
    assert not (vault.root / "test.aesgcm").exists()


def test_partial_journal_blocks_new_seal(vault: LocalVault) -> None:
    journal = vault.root / "access.jsonl"
    journal.write_bytes(b'{"incomplete":')
    journal.chmod(0o600)
    with pytest.raises(FixtureError):
        vault.seal("test.aesgcm", b"data", actor="test")
    assert not (vault.root / "test.aesgcm").exists()
    assert journal.read_bytes() == b'{"incomplete":'


def test_pending_seal_is_visible_not_frozen(
    vault: LocalVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    import knowledge_ticket_agent.evaluation_custody as custody

    original = custody._append

    def fail(directory: int, action: str, actor: str, artifact: str, digest: str) -> None:
        if action == "seal_complete":
            raise FixtureError("synthetic_audit_failure")
        original(directory, action, actor, artifact, digest)

    monkeypatch.setattr(custody, "_append", fail)
    with pytest.raises(FixtureError):
        vault.seal("pending.aesgcm", b"data", actor="test")
    assert vault.audit_status()["pending_seals"] == ["pending.aesgcm"]
    assert (vault.root / "pending.aesgcm").exists()


def test_write_then_failure_cannot_clear_pending(
    vault: LocalVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    import knowledge_ticket_agent.evaluation_custody as custody

    original = custody._write_new

    def fail(directory: int, name: str, raw: bytes) -> None:
        original(directory, name, raw)
        raise FixtureError("synthetic_directory_fsync_failure")

    monkeypatch.setattr(custody, "_write_new", fail)
    with pytest.raises(FixtureError):
        vault.seal("failed.aesgcm", b"data", actor="test")
    assert (vault.root / "failed.aesgcm").exists()
    assert vault.audit_status()["pending_seals"] == ["failed.aesgcm"]


def test_non_utf8_journal_has_redacted_error(vault: LocalVault) -> None:
    journal = vault.root / "access.jsonl"
    journal.write_bytes(b"\xff\n")
    journal.chmod(0o600)
    with pytest.raises(FixtureError, match="audit_encoding"):
        vault.seal("failed.aesgcm", b"data", actor="test")
    assert not (vault.root / "failed.aesgcm").exists()


def test_symlink_and_hardlink_key_rejected(vault: LocalVault, tmp_path: Path) -> None:
    key = vault.root / "key.bin"
    outside = tmp_path / "outside.bin"
    key.rename(outside)
    key.symlink_to(outside)
    with pytest.raises(FixtureError):
        vault.seal("test.aesgcm", b"data", actor="test")
    key.unlink()
    os.link(outside, key)
    with pytest.raises(FixtureError):
        vault.seal("test.aesgcm", b"data", actor="test")


def test_root_replacement_rejected(vault: LocalVault) -> None:
    old = vault.root.with_name("previous")
    vault.root.rename(old)
    vault.root.symlink_to(old)
    with pytest.raises(FixtureError):
        vault.log("validate", actor="test", artifact="sample.aesgcm", digest="a" * 64)


def test_two_process_audit_serialization(vault: LocalVault) -> None:
    code = """import sys,json,traceback
from pathlib import Path
from knowledge_ticket_agent.evaluation_custody import LocalVault
try:
    v=LocalVault(Path(sys.argv[1]))
    for _ in range(12):v.log('validate',actor=sys.argv[2],artifact='sample.aesgcm',digest='a'*64)
except Exception as error:
    chain=[]
    while error is not None:
        chain.append({'type':type(error).__name__,'errno':getattr(error,'errno',None),
            'frames':[(f.name,f.lineno) for f in traceback.extract_tb(error.__traceback__)]})
        error=error.__context__
    print(json.dumps(chain),file=sys.stderr)
    sys.exit(1)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(vault.root), f"test{i}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for i in range(2)
    ]
    outcomes = []
    try:
        for process in processes:
            _, error = process.communicate(timeout=15)
            outcomes.append((process.returncode, error))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
    assert all(code == 0 for code, _ in outcomes), outcomes
    assert vault.audit_status()["events"] == 24
    events = [json.loads(line) for line in (vault.root / "access.jsonl").read_text().splitlines()]
    assert [e["sequence"] for e in events] == list(range(1, 25))


def test_backup_recovery_and_tamper_detection(vault: LocalVault) -> None:
    vault.seal("release.aesgcm", b"synthetic backup data", actor="test")
    restored = vault.open("release.aesgcm", actor="test")
    vault.seal("backup.aesgcm", restored, actor="test")
    assert vault.open("backup.aesgcm", actor="test") == restored
    path = vault.root / "backup.aesgcm"
    before = path.read_bytes()
    path.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
    with pytest.raises(FixtureError):
        vault.open("backup.aesgcm", actor="test")


def test_ordinary_worktree_not_vault(tmp_path: Path) -> None:
    with pytest.raises(FixtureError):
        LocalVault(tmp_path)


def test_lock_replacement_is_rejected(vault: LocalVault) -> None:
    lock = vault.root / "auditlock.bin"
    lock.rename(vault.root / "oldlock.bin")
    lock.write_bytes(b"")
    lock.chmod(0o600)
    with pytest.raises(FixtureError, match="vault_lock_replaced"):
        vault.log("validate", actor="test", artifact="sample.aesgcm", digest="a" * 64)


def test_constructor_directory_swap_cannot_write_lock(
    vault: LocalVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.open

    def swapped(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == vault.root:
            vault.root.rename(vault.root.with_name("moved-vault"))
            vault.root.mkdir(mode=0o700)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapped)
    with pytest.raises(FixtureError, match="vault_permissions"):
        LocalVault(vault.root)
    assert not (vault.root / "auditlock.bin").exists()
