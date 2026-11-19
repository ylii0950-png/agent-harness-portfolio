"""Trusted local M5 operator. No hidden inputs are opened until a durable claim.

Uses existing runner, receipt grader, atomic writer and M3 execution. File custody
is procedural/same-user, not independent blindness. Default calls remain denied.
"""

import fcntl
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from knowledge_ticket_agent.business_evaluation import (
    CaseManifest,
    CaseReceipt,
    EvaluationReport,
    empty_receipt,
    report,
    run_public,
    seal,
)
from knowledge_ticket_agent.evaluation_preflight import (
    PublicRunLedger,
    current_revision,
    source_digests,
)
from knowledge_ticket_agent.evaluation_runtime import execute_public_task
from knowledge_ticket_agent.evaluation_scope import digest
from knowledge_ticket_agent.formal_scope import FORMAL_SUITE, issue, operator_scope
from knowledge_ticket_agent.retrieval_profiles import A1_EVALUATOR, A1_SUITE
from knowledge_ticket_agent.schemas import Dataset
from knowledge_ticket_agent.strict import decode, parse_json, require

VERSION = "lqd-m5-formal-2.1.0"
M2_VERSION = "lqd-m2-formal-1.1.0"


@dataclass(frozen=True)
class FormalRelease:
    commit: str
    sources: tuple[tuple[str, str], ...]
    manifest_identity: str
    corpus_identity: str
    payload_sha256: str
    gold_sha256: str
    gold_version: str
    m2_contract: str
    m2_result_sha256: str
    # Historical FAIL is never converted to PASS. PO may explicitly accept an
    # experimental M5 run while the revised formal M2 gate is still open.
    m2_dependency: str
    profile: str = "formal"
    evaluator: str = VERSION
    suite: str = FORMAL_SUITE
    budgets: tuple[int, ...] = (12, 4, 16, 28, 1)
    denominators: tuple[int, ...] = (18, 20, 24)
    thresholds: tuple[tuple[str, float], ...] = (
        ("recall", 0.9),
        ("mrr", 0.8),
        ("evidence", 1.0),
        ("business", 1.0),
    )
    dataset_version: str = "UNSPECIFIED"

    @property
    def identity(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class FormalPayload:
    data: Dataset
    manifest: CaseManifest


@dataclass(frozen=True)
class GoldContract:
    version: str
    manifest: CaseManifest


def _read(path: Path) -> bytes:
    PublicRunLedger._no_symlink_ancestors(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_nlink == 1
            and info.st_size <= 4_000_000,
            "formal_private_file",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read(4_000_001)
    finally:
        os.close(fd)


def canonical_root(purpose: str = "M5") -> Path:
    import subprocess

    common = Path(
        subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()
    ).resolve()
    require(purpose in {"M2", "M5"}, "formal_purpose")
    return common / ("m5-formal-private" if purpose == "M5" else "m2-revised-formal-private")


class FormalOperator:
    """Production root fixed per repository; synthetic tests use a separate profile."""

    def __init__(self, root: Path, *, rehearsal: bool = False, purpose: str = "M5") -> None:
        require(purpose in {"M2", "M5"}, "formal_purpose")
        self.purpose = purpose
        if not rehearsal:
            require(not os.getenv("CI") and not os.getenv("GITHUB_ACTIONS"), "formal_ci_denied")
            require(root == canonical_root(purpose), "formal_canonical_root")
        else:
            require(not any(p.casefold() == ".git" for p in root.parts), "rehearsal_not_production")
        PublicRunLedger._no_symlink_ancestors(root)
        self.root, self.rehearsal = root, rehearsal

    def inspect(self) -> dict[str, object]:
        # Read saved state only. Never loads payload, reruns ranking, or retries.
        claim = self.root / "claim"
        if not claim.exists():
            return {"state": "NOT_STARTED", "retry_allowed": False}
        try:
            started = parse_json(_read(claim / "STARTED.json").decode())
            require(
                type(started) is dict
                and started.get("state") == "STARTED"
                and type(started.get("run_id")) is str,
                "formal_start_record",
            )
            assert isinstance(started, dict)
            completed, failed = claim / "COMPLETED.json", claim / "FAILED.json"
            require(not (completed.exists() and failed.exists()), "formal_terminal_conflict")
            for state, path in (("COMPLETED", completed), ("FAILED", failed)):
                if path.exists():
                    value = parse_json(_read(path).decode())
                    require(
                        type(value) is dict
                        and value.get("state") == state
                        and value.get("run_id") == started["run_id"],
                        "formal_state",
                    )
                    assert isinstance(value, dict)
                    if state == "COMPLETED":
                        saved = parse_json(_read(claim / "report.json").decode())
                        require(
                            type(saved) is dict and digest(saved) == value.get("report_sha256"),
                            "formal_saved_report_integrity",
                        )
                    return {"state": state, "record_sha256": digest(value), "retry_allowed": False}
            return {"state": "STARTED_OR_INCOMPLETE", "retry_allowed": False}
        except (OSError, ValueError, TypeError):
            return {"state": "CORRUPT_OR_INCOMPLETE", "retry_allowed": False}

    def execute(
        self, release: FormalRelease, *, loader: Callable[[], FormalPayload] | None = None
    ) -> EvaluationReport:
        require(self.purpose == "M5", "M5_operator_required")
        return cast(EvaluationReport, self._execute(release, loader=loader))

    def execute_m2(self, release: FormalRelease) -> dict[str, Any]:
        require(self.purpose == "M2", "M2_operator_required")
        return cast(dict[str, Any], self._execute(release))

    def _execute(
        self, release: FormalRelease, *, loader: Callable[[], FormalPayload] | None = None
    ) -> EvaluationReport | dict[str, Any]:
        from knowledge_ticket_agent.quality_formal import DENOMINATORS, SUITE

        m2 = self.purpose == "M2"
        require(
            type(release) is FormalRelease
            and release.commit == current_revision()
            and release.sources == source_digests()
            and release.evaluator
            == (M2_VERSION if m2 else A1_EVALUATOR if release.suite == A1_SUITE else VERSION)
            and release.suite in ({SUITE} if m2 else {FORMAL_SUITE, A1_SUITE})
            and release.budgets == (12, 4, 16, 28, 1)
            and release.denominators == (DENOMINATORS if m2 else (18, 20, 24))
            and release.profile == ("public_rehearsal" if self.rehearsal else "formal")
            and release.m2_dependency
            in ({"NOT_APPLICABLE"} if m2 else {"REVISED_FORMAL_PASS", "PO_EXPERIMENTAL_EXCEPTION"}),
            "formal_release_or_M2_dependency",
        )
        require(
            release.thresholds == FormalRelease.__dataclass_fields__["thresholds"].default
            and all(type(n) is int for n in (*release.budgets, *release.denominators))
            and all(type(k) is str and type(v) is float for k, v in release.thresholds)
            and all(
                len(h) == 64 and all(c in "0123456789abcdef" for c in h)
                for h in (
                    release.manifest_identity,
                    release.corpus_identity,
                    release.payload_sha256,
                    release.gold_sha256,
                    release.m2_result_sha256,
                )
            )
            and type(release.gold_version) is str
            and bool(release.gold_version)
            and type(release.m2_contract) is str
            and bool(release.m2_contract)
            and type(release.dataset_version) is str
            and bool(release.dataset_version)
            and (self.rehearsal or release.dataset_version not in {"", "UNSPECIFIED"}),
            "formal_lock_fields",
        )
        # A separate owner-only approval record, created only by trusted operator
        # after the exact PO decision. No model/CLI boolean supplies authority.
        approval = parse_json(_read(self.root / "approval.json").decode())
        require(
            approval
            == {
                "release_identity": release.identity,
                "decision": "AUTHORIZE_LOCKED_M2_ONCE" if m2 else "AUTHORIZE_LOCKED_M5_ONCE",
                "profile": "rehearsal" if self.rehearsal else "formal",
                "m2_dependency": release.m2_dependency,
            },
            "formal_exact_authorization",
        )
        if release.m2_dependency == "REVISED_FORMAL_PASS":
            m2_raw = _read(self.root / "m2-formal-result.json")
            dependency = parse_json(m2_raw.decode())
            require(
                hashlib.sha256(m2_raw).hexdigest() == release.m2_result_sha256
                and type(dependency) is dict
                and dependency.get("revised_formal_quality") == "PASS"
                and dependency.get("scope") == "INDEPENDENT_FORMAL"
                and dependency.get("contract") == release.m2_contract,
                "M2_formal_dependency_unverified",
            )
        info = self.root.stat()
        require(
            info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
            "formal_private_root",
        )
        if not self.rehearsal:
            require(loader is None, "formal_no_injected_loader")
            # Installed payloads must actually be the code named by the commit,
            # not just mutually consistent hashes from an uncommitted checkout.
            import subprocess

            for module, expected in release.sources:
                package, name = module.rsplit(".", 1)
                prefix = (
                    "src/agent_harness_lab"
                    if package == "agent_harness_lab"
                    else "examples/knowledge_ticket_agent/src/knowledge_ticket_agent"
                )
                raw = subprocess.check_output(
                    ["git", "show", release.commit + ":" + prefix + "/" + name + ".py"]
                )
                require(hashlib.sha256(raw).hexdigest() == expected, "formal_committed_sources")
        mutex = os.open(self.root / "mutex", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            mutex_info = os.fstat(mutex)
            require(
                stat.S_ISREG(mutex_info.st_mode)
                and mutex_info.st_uid == os.getuid()
                and stat.S_IMODE(mutex_info.st_mode) == 0o600
                and mutex_info.st_nlink == 1,
                "formal_mutex_permissions",
            )
            fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claim = self.root / "claim"
            claim.mkdir(mode=0o700)  # Atomic, fixed identity; changing output cannot retry.
            parent_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            write = PublicRunLedger._write
            run_id = str(uuid.uuid4())
            write(
                claim / "STARTED.json",
                {"state": "STARTED", "run_id": run_id, "release": asdict(release)},
            )
            receipts: dict[str, CaseReceipt] = {}
            manifest: CaseManifest | None = None
            active: set[str] = set()
            try:
                if m2:
                    require(loader is None, "M2_no_injected_loader")
                    return self._score_m2(release, claim, run_id)
                if loader is not None:
                    payload = loader()
                else:
                    raw = _read(self.root / "payload.json")
                    require(
                        hashlib.sha256(raw).hexdigest() == release.payload_sha256,
                        "formal_payload_integrity",
                    )
                    data = decode(Dataset, parse_json(raw.decode()))
                    # The independent custodian supplies the normative versioned
                    # CaseManifest (runtime inputs plus scorer-only oracle). It is
                    # the locked Gold contract, not developer-generated expectations.
                    gold_raw = _read(self.root / "gold-contract.json")
                    require(
                        hashlib.sha256(gold_raw).hexdigest() == release.gold_sha256,
                        "formal_gold_integrity",
                    )
                    contract = decode(GoldContract, parse_json(gold_raw.decode()))
                    require(contract.version == release.gold_version, "formal_gold_version")
                    payload = FormalPayload(data, contract.manifest)
                require(type(payload) is FormalPayload, "formal_payload_type")
                manifest = payload.manifest
                require(
                    manifest.suite_id == release.suite
                    and manifest.identity == release.manifest_identity
                    and manifest.corpus_identity
                    == release.corpus_identity
                    == digest(asdict(payload.data)),
                    "formal_input_binding",
                )
                from knowledge_ticket_agent.acceptance_inputs import validate_business

                validate_business(payload.data, GoldContract(release.gold_version, manifest))

                def persist(case_id: str, receipt: CaseReceipt | None) -> None:
                    if receipt is None:
                        active.add(case_id)
                        write(claim / (case_id + ".STARTED.json"), {"case_id": case_id})
                    else:
                        write(claim / (case_id + ".receipt.json"), asdict(receipt))
                        receipts[case_id] = receipt
                        active.discard(case_id)

                with operator_scope(release.identity, manifest.identity, release.corpus_identity):
                    result = run_public(
                        manifest,
                        lambda: payload.data,
                        lambda data, task: execute_public_task(
                            data, task, case_scope=issue(data, task.ticket_id)
                        ),
                        on_case=persist,
                    )
                result = replace(
                    result,
                    relevance_profile=(
                        "PUBLIC_SYNTHETIC_FORMAL_COMPATIBILITY_NOT_FORMAL"
                        if self.rehearsal
                        else "RESTRICTED_VERSIONED_GOLD"
                    ),
                )
                write(claim / "report.json", asdict(result))
                write(
                    claim / "COMPLETED.json",
                    {
                        "state": "COMPLETED",
                        "run_id": run_id,
                        "report_sha256": digest(asdict(result)),
                    },
                )
                return result
            except BaseException:
                write(
                    claim / "FAILED.json",
                    {
                        "state": "FAILED",
                        "run_id": run_id,
                        "reason": "formal_execution_failed_no_retry",
                    },
                )
                if m2:
                    write(
                        claim / "failure-summary.json",
                        {
                            "cases": [
                                {
                                    "slot": n,
                                    "status": "saved_receipt"
                                    if (claim / f"M2-{n:03d}.receipt.json").exists()
                                    else "unknown_effect"
                                    if (claim / f"M2-{n:03d}.STARTED.json").exists()
                                    else "not_run",
                                }
                                for n in range(1, 29)
                            ],
                            "retry_allowed": False,
                        },
                    )
                    raise
                write(
                    claim / "failure-summary.json",
                    {
                        "cases": [
                            {
                                "case_id": f"KT-E{n:02d}",
                                "status": "saved_receipt"
                                if f"KT-E{n:02d}" in receipts
                                else "unknown_effect"
                                if f"KT-E{n:02d}" in active
                                else "not_run",
                            }
                            for n in range(1, 25)
                        ],
                        "retry_allowed": False,
                    },
                )
                if manifest is not None and manifest.identity == release.manifest_identity:
                    rows = []
                    for case in manifest.cases:
                        r = receipts.get(case.case_id)
                        if r is None:
                            r = seal(
                                replace(
                                    empty_receipt(
                                        case,
                                        "infrastructure_error"
                                        if case.case_id in active
                                        else "not_run",
                                        "formal_interrupted",
                                    ),
                                    blocker=case.case_id in active,
                                )
                            )
                        rows.append(r)
                    with operator_scope(
                        release.identity, manifest.identity, release.corpus_identity
                    ):
                        write(claim / "partial-report.json", asdict(report(manifest, tuple(rows))))
                raise
        finally:
            os.close(mutex)

    def _score_m2(self, release: FormalRelease, claim: Path, run_id: str) -> dict[str, Any]:
        from knowledge_ticket_agent.quality_formal import QualityGold, QualityInputs, evaluate

        raw, gold_raw = _read(self.root / "payload.json"), _read(self.root / "gold-contract.json")
        require(
            hashlib.sha256(raw).hexdigest() == release.payload_sha256
            and hashlib.sha256(gold_raw).hexdigest() == release.gold_sha256,
            "M2_input_integrity",
        )
        inputs = decode(QualityInputs, parse_json(raw.decode()))
        gold = decode(QualityGold, parse_json(gold_raw.decode()))
        require(
            gold.version == release.gold_version
            and inputs.version == release.dataset_version
            and gold.contract == release.m2_contract
            and digest(asdict(gold)) == release.manifest_identity
            and digest(asdict(inputs.data)) == release.corpus_identity,
            "M2_input_binding",
        )

        def persist(ordinal: int, row: dict[str, Any] | None) -> None:
            suffix = ".STARTED.json" if row is None else ".receipt.json"
            PublicRunLedger._write(
                claim / (f"M2-{ordinal:03d}" + suffix), {"slot": ordinal} if row is None else row
            )

        result = evaluate(inputs, gold, on_case=persist)
        result.update(release_identity=release.identity, run_id=run_id)
        if not self.rehearsal:
            result.update(
                scope="INDEPENDENT_FORMAL", revised_formal_quality=result["engineering_contract"]
            )
        PublicRunLedger._write(claim / "report.json", result)
        PublicRunLedger._write(
            claim / "COMPLETED.json",
            {"state": "COMPLETED", "run_id": run_id, "report_sha256": digest(result)},
        )
        return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Restricted M5 operator; exact PO lock required")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--inspect", action="store_true")
    parser.add_argument("--purpose", choices=("M2", "M5"), default="M5")
    args = parser.parse_args()
    if not args.execute and not args.inspect:
        from knowledge_ticket_agent.quality_formal import DENOMINATORS, SUITE

        print(
            json.dumps(
                {
                    "evaluator": M2_VERSION if args.purpose == "M2" else A1_EVALUATOR,
                    "suite": SUITE if args.purpose == "M2" else A1_SUITE,
                    "commit": current_revision(),
                    "sources_sha256": digest(source_digests()),
                    "public_interface": "IMPLEMENTED",
                    "formal_result": "NOT_RUN",
                    "independent_dataset_and_gold": "AWAITING_CUSTODIAN_FREEZE",
                    "exact_release_authorization": "NOT_GRANTED",
                    "denominators": DENOMINATORS if args.purpose == "M2" else [18, 20, 24],
                    "M2_dependency": "HISTORICAL_FAIL_REVISED_FORMAL_NOT_RUN",
                },
                indent=2,
            )
        )
        return
    operator = FormalOperator(canonical_root(args.purpose), purpose=args.purpose)
    if args.inspect:
        print(json.dumps(operator.inspect(), indent=2))
        return
    parsed = parse_json(_read(operator.root / "release-lock.json").decode())
    require(type(parsed) is dict, "formal_release_schema")
    value = cast(dict[str, Any], parsed)
    for key in ("sources", "thresholds"):
        value[key] = tuple(tuple(row) for row in value[key])
    for key in ("budgets", "denominators"):
        value[key] = tuple(value[key])
    release = FormalRelease(**value)
    if args.purpose == "M2":
        result_m2 = operator.execute_m2(release)
        print(
            json.dumps(
                {
                    k: result_m2[k]
                    for k in (
                        "contract",
                        "release_identity",
                        "revised_formal_quality",
                        "denominators",
                        "recall_at_3",
                        "mrr_at_3",
                        "documentary_passed",
                        "negative_passed",
                    )
                }
            )
        )
        return
    result = operator.execute(release)
    # Full observed receipts remain in owner-only custody. No hidden input,
    # oracle, query, evidence mapping, or per-case policy diagnostics are printed.
    print(
        json.dumps(
            {
                "evaluator": release.evaluator,
                "release_identity": release.identity,
                "cases": result.case_denominator,
                "passed": result.passed,
                "blocker": result.blocker,
                "formal_execution": "COMPLETED",
                "quality_acceptance": "PASS"
                if result.passed == 24
                and not result.blocker
                and result.rec_macro == 1.0
                and dict(result.retrieval_macro).get("recall", 0) >= 0.9
                and dict(result.retrieval_macro).get("reciprocal_rank", 0) >= 0.8
                else "FAIL",
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError, KeyError):
        print(json.dumps({"operator": "FAILED_CLOSED", "details": "retained_locally"}))
        raise SystemExit(1) from None
