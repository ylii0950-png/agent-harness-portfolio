"""Public rehearsal claims and an explicitly gated restricted Operator entry.

Default entry refuses before any private access. Restricted execution delegates
to its exact-release Operator; M2F custody is never opened here.
"""

import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from agent_harness_lab.workspace import FileErrorCode, FileToolError, WorkspaceFiles
from knowledge_ticket_agent.business_evaluation import (
    PUBLIC_SUITE,
    CaseManifest,
    CaseReceipt,
    EvaluationReport,
    empty_receipt,
    report,
    run_public,
    seal,
)
from knowledge_ticket_agent.evaluation_scope import digest
from knowledge_ticket_agent.schemas import Dataset
from knowledge_ticket_agent.strict import require

FORMAL_CASE_IDS = tuple(f"KT-E{n:02d}" for n in range(1, 25))
FORMAL_DENOMINATORS = (("retrieval_positive", 18), ("required_evidence", 20), ("business", 24))


@dataclass(frozen=True)
class ReleaseLock:
    suite_id: str
    manifest_identity: str
    commit: str
    source_digests: tuple[tuple[str, str], ...]
    corpus_identity: str
    method: str
    budget: tuple[int, ...]
    result_schema: str

    @property
    def identity(self) -> str:
        return digest(asdict(self))


def source_digests() -> tuple[tuple[str, str], ...]:
    """Bind actual installed package code, without importing custody/live modules."""
    import agent_harness_lab
    import knowledge_ticket_agent

    rows = []
    for package in (agent_harness_lab, knowledge_ticket_agent):
        folder = Path(cast(str, package.__file__)).parent
        for source in sorted(folder.glob("*.py")):
            require(not source.is_symlink(), "package_source_symlink")
            rows.append(
                (
                    package.__name__ + "." + source.stem,
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                )
            )
    return tuple(rows)


def current_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def lock_public(manifest: CaseManifest, commit: str) -> ReleaseLock:
    require(
        manifest.suite_id in {PUBLIC_SUITE, "kt-m5-synthetic-24-1.0.0"}
        and bool(re.fullmatch(r"[a-f0-9]{40}", commit)),
        "release_identity",
    )
    require(commit == current_revision(), "release_commit_mismatch")
    return ReleaseLock(
        manifest.suite_id,
        manifest.identity,
        commit,
        source_digests(),
        manifest.corpus_identity,
        manifest.method,
        manifest.budget,
        manifest.schema,
    )


def formal_preflight() -> tuple[tuple[str, str], ...]:
    """Metadata-only readiness; does not inspect a sealed case to decide readiness."""
    return (
        ("po_formal_authorization", "BLOCKED"),
        ("sealed_scope_issuer", "IMPLEMENTED_OPERATOR_SCOPED"),
        ("locked_formal_release", "NOT_CREATED"),
        ("all_fixture_runtime_paths", "PUBLIC_COMPATIBILITY_ONLY_FROZEN_CONTRACT_PENDING"),
        ("M2_quality_acceptance", "FAIL_HISTORICAL_OPEN"),
        ("formal_24_execution", "NOT_RUN"),
    )


def run_formal(
    *,
    release: ReleaseLock | None = None,
    payload_loader: Callable[[], object] | None = None,
    operator: object = None,
    formal_release: object = None,
) -> EvaluationReport:
    # Refuse before loader/ledger/path validation or any private data access.
    # No boolean switch, fabricated approval string or caller-selected output enables it.
    from knowledge_ticket_agent.formal_evaluation import FormalOperator, FormalRelease

    if type(operator) is FormalOperator and type(formal_release) is FormalRelease:
        require(release is None and payload_loader is None, "formal_no_public_override")
        return operator.execute(formal_release)
    raise PermissionError("M5_FORMAL_DISABLED_PO_AND_LOCKED_RELEASE_REQUIRED")


class PublicRunLedger:
    """One persistent claim per exact public suite in the configured central root."""

    def __init__(self, root: Path, *, suite_id: str = PUBLIC_SUITE) -> None:
        require(suite_id in {PUBLIC_SUITE, "kt-m5-synthetic-24-1.0.0"}, "public_suite_only")
        self.suite_id = suite_id
        parts = Path(os.path.abspath(root)).parts
        require(
            not any(
                p.casefold() == ".git"
                and i + 1 < len(parts)
                and parts[i + 1].casefold().startswith("m2")
                for i, p in enumerate(parts)
            ),
            "private_M2_storage_forbidden",
        )
        self._no_symlink_ancestors(root)
        self.root = root

    @staticmethod
    def _no_symlink_ancestors(path: Path) -> None:
        absolute = Path(os.path.abspath(path))
        for part in reversed((absolute, *absolute.parents)):
            try:
                mode = part.lstat().st_mode
            except FileNotFoundError:
                break
            require(not stat.S_ISLNK(mode), "ledger_symlink_ancestor")

    def inspect(self) -> dict[str, object]:
        """Read-only diagnostics; an incomplete/failed claim can never restart."""
        claim = self.root / self.suite_id
        try:
            self._no_symlink_ancestors(self.root)
        except (OSError, ValueError):
            return {"state": "CORRUPT_OR_INCOMPLETE", "retry_allowed": False}
        if self.root.is_symlink() or claim.is_symlink():
            return {"state": "CORRUPT_OR_INCOMPLETE", "retry_allowed": False}
        if not claim.exists():
            return {"state": "NOT_STARTED", "retry_allowed": False}
        try:
            with WorkspaceFiles(self.root) as files:

                def read(name: str) -> dict[str, object] | None:
                    try:
                        value = json.loads(files.read_text(self.suite_id + "/" + name))
                        require(type(value) is dict, "claim_schema")
                        return cast(dict[str, object], value)
                    except FileToolError as error:
                        if error.code is FileErrorCode.NOT_FOUND:
                            return None
                        raise

                started = read("STARTED.json")
                require(started is not None, "missing_start")
                assert started is not None
                require(
                    set(started) == {"state", "run_id", "release"}
                    and started["state"] == "STARTED"
                    and type(started["run_id"]) is str
                    and type(started["release"]) is dict,
                    "start_schema",
                )
                failed, completed = read("FAILED.json"), read("COMPLETED.json")
                require(
                    not (failed is not None and completed is not None),
                    "contradictory_terminal_states",
                )
                if failed is not None:
                    require(
                        set(failed) == {"state", "run_id", "reason"}
                        and failed["state"] == "FAILED"
                        and failed["reason"] == "execution_interrupted_or_failed",
                        "failed_schema",
                    )
                if completed is not None:
                    require(
                        set(completed) == {"state", "run_id", "report_sha256", "blocker"}
                        and completed["state"] == "COMPLETED"
                        and type(completed["blocker"]) is bool,
                        "completed_schema",
                    )
                state = (
                    failed
                    if failed is not None
                    else completed
                    if completed is not None
                    else started
                )
                if completed is not None and failed is None:
                    result = read("report.json")
                    require(
                        result is not None and digest(result) == completed["report_sha256"],
                        "report_integrity",
                    )
            require(state["run_id"] == started["run_id"], "claim_run_mismatch")
            return {"state": state["state"], "run_id": state["run_id"], "retry_allowed": False}
        except (OSError, ValueError, KeyError, TypeError, FileToolError):
            return {"state": "CORRUPT_OR_INCOMPLETE", "retry_allowed": False}

    def execute(
        self, lock: ReleaseLock, manifest: CaseManifest, data_factory: Callable[[], Dataset]
    ) -> EvaluationReport:
        require(lock.suite_id == manifest.suite_id == self.suite_id, "sealed_suite_disabled")
        require(
            lock.manifest_identity == manifest.identity
            and lock.corpus_identity == manifest.corpus_identity
            and lock.method == manifest.method
            and lock.budget == manifest.budget
            and lock.result_schema == manifest.schema
            and lock.source_digests == source_digests(),
            "release_mismatch",
        )
        require(lock.commit == current_revision(), "release_commit_mismatch")
        self._no_symlink_ancestors(self.root)
        require(not self.root.is_symlink(), "ledger_symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(self.root.is_dir() and self.root.stat().st_mode & 0o077 == 0, "ledger_permissions")
        fd = os.open(self.root / "mutex", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claim = self.root / manifest.suite_id
            try:
                claim.mkdir(mode=0o700)
            except FileExistsError:
                raise PermissionError("release_already_claimed_no_retry") from None
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
            run_id = str(uuid.uuid4())
            (claim / "case-starts").mkdir(mode=0o700)
            (claim / "case-receipts").mkdir(mode=0o700)
            self._write(
                claim / "STARTED.json",
                {"state": "STARTED", "run_id": run_id, "release": asdict(lock)},
            )
            completed_cases: dict[str, CaseReceipt] = {}
            active: set[str] = set()

            def persist(case_id: str, receipt: CaseReceipt | None) -> None:
                if receipt is None:
                    active.add(case_id)
                    self._write(
                        claim / "case-starts" / (case_id + ".json"),
                        {"case_id": case_id, "state": "STARTED", "run_id": run_id},
                    )
                else:
                    self._write(claim / "case-receipts" / (case_id + ".json"), asdict(receipt))
                    completed_cases[case_id] = receipt
                    active.discard(case_id)

            try:
                result = run_public(manifest, data_factory, on_case=persist)
                self._write(claim / "report.json", asdict(result))
                self._write(
                    claim / "COMPLETED.json",
                    {
                        "state": "COMPLETED",
                        "run_id": run_id,
                        "report_sha256": digest(asdict(result)),
                        "blocker": result.blocker,
                    },
                )
                return result
            except BaseException:
                rows = []
                for case in manifest.cases:
                    receipt = completed_cases.get(case.case_id)
                    if receipt is None:
                        receipt = empty_receipt(
                            case,
                            "infrastructure_error" if case.case_id in active else "not_run",
                            "release_interrupted",
                        )
                        if case.case_id in active:
                            receipt = seal(replace(receipt, blocker=True))
                    rows.append(receipt)
                partial = report(manifest, tuple(rows))
                self._write(claim / "partial-report.json", asdict(partial))
                self._write(
                    claim / "FAILED.json",
                    {
                        "state": "FAILED",
                        "run_id": run_id,
                        "reason": "execution_interrupted_or_failed",
                    },
                )
                raise
        finally:
            os.close(fd)

    @staticmethod
    def _write(path: Path, value: object) -> None:
        raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("receipt_write_failed")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def main() -> None:
    import argparse

    from knowledge_ticket_agent import load_dataset
    from knowledge_ticket_agent.business_evaluation import public_manifest

    parser = argparse.ArgumentParser(
        description="M5-P public rehearsal/preflight; sealed execution disabled"
    )
    parser.add_argument("--synthetic24", action="store_true")
    parser.add_argument("--rehearse", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--ledger-root", type=Path, default=Path("artifacts/m5p-public-control"))
    args = parser.parse_args()
    if args.formal:
        run_formal()
    suite_id = "kt-m5-synthetic-24-1.0.0" if args.synthetic24 else PUBLIC_SUITE
    if args.inspect:
        print(
            json.dumps(
                PublicRunLedger(args.ledger_root, suite_id=suite_id).inspect(), sort_keys=True
            )
        )
        return
    if not args.rehearse:
        print(
            json.dumps(
                {
                    "formal_preflight": formal_preflight(),
                    "formal_denominators": FORMAL_DENOMINATORS,
                    "formal_execution": "NOT_RUN",
                },
                sort_keys=True,
            )
        )
        return
    data = load_dataset()
    manifest = public_manifest(data)
    if args.synthetic24:
        from knowledge_ticket_agent.evaluation_matrix import manifest as matrix_manifest

        manifest = matrix_manifest(data)
    release = lock_public(manifest, current_revision())
    result = PublicRunLedger(args.ledger_root, suite_id=suite_id).execute(
        release, manifest, load_dataset
    )
    if args.synthetic24:
        from knowledge_ticket_agent.evaluation_matrix import conformance

        print(json.dumps(conformance(result), sort_keys=True))
    print(
        json.dumps(
            {
                "suite": result.suite_id,
                "case_denominator": result.case_denominator,
                "passed": result.passed,
                "blocker": result.blocker,
                "semantic_identity": result.semantic_identity,
                "report": str(args.ledger_root / manifest.suite_id / "report.json"),
                "formal_execution": "NOT_RUN",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
