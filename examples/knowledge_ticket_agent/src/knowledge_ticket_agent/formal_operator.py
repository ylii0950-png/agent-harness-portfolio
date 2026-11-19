"""Trusted local M2F operator. Never print private inputs or full results.

Fixed repository/common-Git custody, no output override, retry or reset. Public
tests use generic runner/ledger fixtures; the real release is never a CI input.
"""

import argparse
import base64
import fcntl
import importlib
import io
import json
import os
import subprocess
import tarfile
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.evaluation_custody import _chain, _file_permissions, _read
from knowledge_ticket_agent.evaluation_integrity import DEMO, _object, verify_frozen_inputs
from knowledge_ticket_agent.evaluation_integrity import _read as public_read
from knowledge_ticket_agent.evaluation_maintenance import (
    VERSION,
    HeldoutDataset,
    parse_payload,
    payload,
    validate_heldout,
)
from knowledge_ticket_agent.formal_ledger import Ledger, canonical, digest, secure_info
from knowledge_ticket_agent.formal_scoring import PROTOCOL, run, verify_records
from knowledge_ticket_agent.strict import FixtureError, require

BASE = "fb53d115aef25e08e6b9f4cf1e0cbc0e0326a602"
GOLD_SHA256 = "f7a25aadfa0e60449bbe03933c8352536a03d30283d6ce5dcdaecfa6e6109651"
GATES = frozenset(
    {
        "py311-core",
        "py311-demo",
        "py311-lint",
        "py311-format",
        "py311-mypy",
        "py311-demo-mypy",
        "py311-build",
        "py311-demo-build",
        "py311-clean",
        "py312-core",
        "py312-demo",
        "py312-lint",
        "py312-format",
        "py312-mypy",
        "py312-demo-mypy",
        "py312-build",
        "py312-demo-build",
        "py312-clean",
        "independent-review",
        "synthetic-rehearsal",
    }
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
    ).strip()


def location(root: Path) -> tuple[Path, Path]:
    require(not any(os.environ.get(n) for n in ("CI", "GITHUB_ACTIONS")), "formal_local_only")
    require(all(not p.is_symlink() for p in (root, *root.parents)), "formal_symlink")
    root = root.absolute()
    require(git(root, "rev-parse", "--show-toplevel") == str(root), "formal_repository")
    require(
        git(root, "remote", "get-url", "origin")
        == "https://github.com/ylii0950-png/agent-harness-lab.git",
        "formal_origin",
    )
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", BASE, "HEAD"],
        check=True,
        capture_output=True,
    )
    common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    require(common.name == ".git" and common.is_dir() and not common.is_symlink(), "formal_git")
    return root, common


def implementation(root: Path) -> dict[str, str]:
    paths = sorted((root / DEMO / "src/knowledge_ticket_agent").glob("*.py"))
    paths += sorted((root / DEMO / "tests").glob("test_m2f*.py"))
    paths += [root / "docs/phase3-m2f-protocol.md"]
    result = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        raw = public_read(root, relative)
        if "/src/" in relative:
            module = importlib.import_module("knowledge_ticket_agent." + path.stem)
            assert module.__file__ is not None
            require(Path(module.__file__).read_bytes() == raw, "formal_installed_bytes")
        result[relative] = digest(raw)
    return result


@contextmanager
def read_vault(common: Path) -> Iterator[int]:
    """No initialization, creates or journal writes, even if original lock is missing."""
    root = common / "m2e-steward-private"
    require(all(not p.is_symlink() for p in (root, *root.parents)), "formal_custody_symlink")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        secure_info(directory, directory=True)
        before = os.fstat(directory)
        lock = os.open(
            "auditlock.bin", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        try:
            _file_permissions(lock)
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            yield directory
            after = root.stat(follow_symlinks=False)
            require(
                (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino),
                "formal_custody_replaced",
            )
        finally:
            os.close(lock)
    finally:
        os.close(directory)


def custody(root: Path, common: Path, *, scan: bool) -> dict[str, object]:
    attempts = 0

    def deny(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise FixtureError("formal_preflight_rank_forbidden")

    with ExitStack() as stack:
        for name in (
            "knowledge_ticket_agent.retrieval.lexical_rank",
            "knowledge_ticket_agent.retrieval.Retriever.search",
            "knowledge_ticket_agent.retrieval_strategies.lexical_rank",
            "knowledge_ticket_agent.retrieval_strategies.StrategyRetriever.search",
            "knowledge_ticket_agent.formal_scoring.retrieve",
        ):
            stack.enter_context(patch(name, deny))
        result = _custody(root, common, scan=scan)
    require(attempts == 0, "formal_preflight_rank_attempt")
    result["rank_calls"] = attempts
    return result


def _custody(root: Path, common: Path, *, scan: bool) -> dict[str, object]:
    """Trusted memory-only integrity/recovery/leak scan; never calls a ranker.

    Original vault bytes and journal are read-only. Only fixed final entries use
    the filename-bound reviewed AAD. No historical initial decrypt fallback.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    freeze = _object(public_read(root, DEMO + "verification/m2e/freeze-manifest.json"))
    with read_vault(common) as directory:
        key = _read(directory, "key.bin")
        require(len(key) == 32, "formal_custody_key")
        ciphers = freeze["retained_cipher_sha256"]
        assert isinstance(ciphers, dict)
        cipher_bytes = []
        for name, expected in ciphers.items():
            raw = _read(directory, name)
            require(digest(raw) == expected, "formal_frozen_cipher")
            cipher_bytes.append(raw)
        journal = _read(directory, "access.jsonl")
        events = _chain(journal)
        pending: dict[str, object] = {}
        for event in events:
            name = event["artifact"]
            assert isinstance(name, str)
            if event["action"] == "seal_begin":
                pending[name] = event["ciphertext_sha256"]
            elif event["action"] in {"seal_complete", "seal_failed"}:
                require(pending.get(name) == event["ciphertext_sha256"], "formal_audit_pair")
                if event["action"] == "seal_complete":
                    del pending[name]
        require(not pending, "formal_pending_seal")
        prefix = b"\n".join(journal.splitlines()[:18]) + b"\n"
        require(
            digest(prefix) == "a1b90339818f2af97d2a0b0d447d0bcea12b8538ef03b7596ede61041d93abea",
            "formal_frozen_journal",
        )
        require(
            not any("formal" in str(e["action"]) or "score" in str(e["actor"]) for e in events),
            "formal_historical_run",
        )
        plains = []
        for name in ("reviewed.aesgcm", "backup-reviewed.aesgcm"):
            raw = _read(directory, name)
            plain = AESGCM(key).decrypt(
                raw[:12], raw[12:], VERSION.encode() + b"\0" + name.encode()
            )
            require(digest(plain) == GOLD_SHA256, "formal_gold_digest")
            plains.append(plain)
    require(plains[0] == plains[1], "formal_recovery")
    heldout = parse_payload(plains[0])
    validate_heldout(heldout, load_dataset())
    result: dict[str, object] = {
        "status": "PASS",
        "retained_ciphertexts": len(ciphers),
        "gold_sha256": GOLD_SHA256,
        "recovery": "PASS",
        "original_journal_sha256": digest(journal),
        "rank_calls": 0,
        "historical_formal_events": 0,
    }
    if scan:
        needles = [c.query.encode() for c in heldout.cases] + plains[:1] + cipher_bytes
        needles += [key, key.hex().encode(), base64.b64encode(key)]
        count = 0

        def check(raw: bytes) -> None:
            nonlocal count
            require(not any(n in raw for n in needles), "formal_private_leak")
            count += 1

        objects = git(root, "rev-list", "--objects", "--all").splitlines()
        for entry in objects:
            oid = entry.split()[0]
            kind = git(root, "cat-file", "-t", oid)
            if kind == "blob":
                check(subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", oid]))
        # Worktree/index plus all M2F logs; do not read unrelated local auth files.
        names = git(root, "ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        for name in names:
            check(public_read(root, name))
        staged = git(root, "ls-files", "-s").splitlines()
        for entry in staged:
            oid = entry.split()[1]
            check(subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", oid]))
        for path in (root / "artifacts/m2f-20261025").rglob("*.log"):
            check(public_read(root, path.relative_to(root).as_posix()))
        artifacts = build_artifacts(root)
        for path in artifacts:
            archive_bytes = io.BytesIO(public_read(root, path.relative_to(root).as_posix()))
            if path.suffix == ".whl":
                with zipfile.ZipFile(archive_bytes) as archive:
                    for member in archive.namelist():
                        check(archive.read(member))
            else:
                with tarfile.open(fileobj=archive_bytes) as source:
                    for info in source.getmembers():
                        if info.isfile():
                            stream = source.extractfile(info)
                            assert stream is not None
                            check(stream.read())
        result.update(leakage="PASS", scanned_blobs_files_members=count, artifacts=len(artifacts))
    return result


def build_artifacts(root: Path) -> tuple[Path, ...]:
    """Require current pair of pairs; scan retained historical archives too."""
    paths = tuple(
        sorted(
            p
            for folder in (root / "dist", root / DEMO / "dist")
            for p in folder.iterdir()
            if p.name.endswith((".whl", ".tar.gz"))
        )
    )
    required = {
        root / "dist/agent_harness_lab-0.9.0-py3-none-any.whl",
        root / "dist/agent_harness_lab-0.9.0.tar.gz",
        root / DEMO / "dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl",
        root / DEMO / "dist/knowledge_ticket_agent_demo-0.1.0.tar.gz",
    }
    require(required <= set(paths), "formal_artifacts")
    return paths


def preflight(root: Path, common: Path) -> dict[str, object]:
    verify_frozen_inputs(root)
    # Retain every original source/test/config/evidence file, including M2E's
    # newer tests that predate this work but were not in the precreation lock.
    permitted_docs = {
        "AGENTS.md",
        "PROGRESS.md",
        "docs/PROJECT_BRIEF.md",
        "docs/DECISIONS.md",
        "docs/ROADMAP.md",
    }
    originals = git(root, "ls-tree", "-r", "--name-only", BASE).splitlines()
    for name in originals:
        if name in permitted_docs:
            continue
        raw = subprocess.check_output(["git", "-C", str(root), "show", f"{BASE}:{name}"])
        require(public_read(root, name) == raw, "formal_baseline_preservation")
    receipt = _object(public_read(root, "docs/evidence/m2f/preflight.json"))
    require(
        receipt.get("status") == "PASS" and receipt.get("protocol") == PROTOCOL,
        "formal_preflight_status",
    )
    sources = implementation(root)
    require(receipt.get("implementation") == sources, "formal_preflight_code")
    checks = receipt.get("checks")
    require(isinstance(checks, dict) and checks.keys() == GATES, "formal_preflight_gates")
    assert isinstance(checks, dict)
    for entry in checks.values():
        require(isinstance(entry, dict), "formal_preflight_check")
        assert isinstance(entry, dict)
        require(
            entry.get("exit_code") == 0 and type(entry.get("exit_code")) is int,
            "formal_preflight_check",
        )
        log_name = entry.get("log")
        require(
            isinstance(log_name, str)
            and log_name.startswith("artifacts/m2f-20261025/")
            and log_name.endswith(".log"),
            "formal_preflight_log",
        )
        assert isinstance(log_name, str)
        require(digest(public_read(root, log_name)) == entry.get("sha256"), "formal_preflight_log")
    custody_result = custody(root, common, scan=True)
    return {
        "release": VERSION,
        "protocol": PROTOCOL,
        "implementation": sources,
        "preflight_sha256": digest(canonical(receipt)),
        "custody": custody_result,
        "code_head": git(root, "rev-parse", "HEAD"),
        "base": BASE,
        "baseline_files_retained": len(originals) - len(permitted_docs),
    }


def execute(root: Path, common: Path) -> dict[str, object]:
    ledger = Ledger(common / "m2f-formal-private")
    with ledger.transaction() as fd:
        # Refuse claimed release before any custody read, even if the output is changed.
        require(os.listdir(fd) == ["lock.bin"], "formal_already_claimed")
        binding = preflight(root, common)
        state = ledger.start(fd, binding)
        try:
            # Formal input touch happens only after STARTED has been fsynced.
            report = run(load_dataset(), _load_formal(common), lambda value: ledger.save(fd, value))
            ledger.save(fd, {"event": "REPORT", "report": report})
            verified = verify_records(load_dataset(), ledger.records(fd), complete=True)
            require(canonical(verified) == canonical(report), "formal_report_readback")
            state = ledger.finish(fd, "COMPLETED", report_sha256=digest(canonical(report)))
            return {
                "run_id": state["run_id"],
                "status": state["status"],
                "report_sha256": state["report_sha256"],
                "summary": report["summary"],
            }
        except BaseException as exc:
            if ledger.state(fd)["status"] == "STARTED":
                ledger.finish(fd, "FAILED", error_type=type(exc).__name__)
            raise


def _load_formal(common: Path) -> HeldoutDataset:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    with read_vault(common) as directory:
        raw = _read(directory, "reviewed.aesgcm")
        plain = AESGCM(_read(directory, "key.bin")).decrypt(
            raw[:12], raw[12:], VERSION.encode() + b"\0reviewed.aesgcm"
        )
    require(digest(plain) == GOLD_SHA256, "formal_gold_digest")
    return parse_payload(plain)


def recover(common: Path) -> dict[str, object]:
    return recover_saved(common, expected_gold_sha256=GOLD_SHA256)


def recover_saved(common: Path, *, expected_gold_sha256: str) -> dict[str, object]:
    """Generic saved-byte recovery used by public fixtures. CLI pins real Gold."""
    ledger = Ledger(common / "m2f-formal-private")
    with ledger.transaction() as fd:
        state = ledger.state(fd)
        if state["status"] == "STARTED":
            state = ledger.finish(fd, "FAILED", error_type="Interrupted")
        values = ledger.records(fd)
        if values:
            require(
                digest(payload(parse_payload(canonical(values[0].get("gold")))))
                == expected_gold_sha256,
                "formal_saved_gold_digest",
            )
        if state["status"] == "COMPLETED":
            require(
                len(values) == 74 and values[-1].get("event") == "REPORT",
                "formal_report_incomplete",
            )
            report = values[-1]["report"]
            assert isinstance(report, dict)
            require(digest(canonical(report)) == state["report_sha256"], "formal_report_digest")
            verified = verify_records(load_dataset(), values, complete=True)
            require(canonical(verified) == canonical(report), "formal_report_arithmetic")
            return {
                "run_id": state["run_id"],
                "status": state["status"],
                "report_sha256": state["report_sha256"],
                "summary": report["summary"],
                "recovery": "PASS_NO_RERANK",
            }
        recovered: dict[str, object] = (
            verify_records(load_dataset(), values, complete=len(values) == 74)
            if values
            else {
                "rows": [
                    {
                        "method": m,
                        "query_id": f"M2E-{polarity}{i:02}",
                        "status": "NOT_RUN",
                        "missing_evidence": "INPUT_NOT_LOADED",
                    }
                    for m in ("A1", "bm25", "keyword")
                    for polarity, count in (("P", 18), ("N", 6))
                    for i in range(1, count + 1)
                ],
                "summary": "INPUT_NOT_LOADED_NO_RETRIEVAL",
            }
        )
        recovery_digest = ledger.recovery_report(fd, recovered)
        return {
            "run_id": state["run_id"],
            "status": state["status"],
            "saved_encrypted_records": len(values),
            "retry_allowed": False,
            "recovery_report_sha256": recovery_digest,
            "denominator_cells": 72,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted local M2F operator; no reset or retry")
    parser.add_argument("--repository-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    try:
        root, common = location(args.repository_root.absolute())
        if args.execute:
            result = execute(root, common)
        elif args.recover:
            result = recover(common)
        else:
            result = {"status": "PASS", "binding": preflight(root, common)}
        print(json.dumps(result, sort_keys=True))
    except BaseException as exc:
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error_type": type(exc).__name__,
                    "error_code": exc.code
                    if isinstance(exc, FixtureError)
                    else "formal_operator_failure",
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
