"""Public frozen-input verification; no private dataset, credential or ranker access.

The sole migration is an exact approved CI transition, not a mutable allowlist.
This is reproducibility evidence, not protection against a same-user code editor.
"""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

from knowledge_ticket_agent.strict import FixtureError, parse_json, require

DEMO = "examples/knowledge_ticket_agent/"
LOCK = DEMO + "verification/m2e/retriever-lock.json"
COMPAT = DEMO + "verification/m2e/ci-compat-v1/"
LOCK_SHA256 = "fef2103c881461ec5178a35b0439b7a450726a448fe569c8c17f4bb56db3b8bd"
ATTESTATION_SHA256 = "882921b48d0b8e945e4cd81791e5d0b133fe068669644723be55cece5dc49d61"
OLD_CI_SHA256 = "5dfe85a6be1606c611dbd397e95548eeefce2f7f2cbac7e948cbbfd46f18d9d0"
NEW_CI_SHA256 = "4086f67ad0519f060ba71ab99282fdebd58c9e7cd95f7be337d14ea4918638b5"
MAX_BYTES = 4 * 1024 * 1024


def _read(root: Path, relative: str) -> bytes:
    path = PurePosixPath(relative)
    require(
        not path.is_absolute()
        and bool(path.parts)
        and all(p not in {"..", "."} for p in path.parts)
        and str(path) == relative,
        "integrity_path",
    )
    require(not root.is_symlink(), "integrity_symlink")
    directory = -1
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in path.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_BYTES, "integrity_file")
            raw = source.read(MAX_BYTES + 1)
            require(len(raw) <= MAX_BYTES, "integrity_size")
            return raw
    except OSError:
        raise FixtureError("integrity_read") from None
    finally:
        if directory >= 0:
            os.close(directory)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(raw: bytes) -> dict[str, object]:
    try:
        obj = parse_json(raw.decode("utf-8"))
    except UnicodeError:
        raise FixtureError("integrity_encoding") from None
    require(isinstance(obj, dict), "integrity_object")
    assert isinstance(obj, dict)
    return obj


def approved_ci_transition(old: bytes) -> bytes:
    """Exact PR16 patch on M2's old CI, retaining every preexisting step."""
    require(_digest(old) == OLD_CI_SHA256, "original_ci_digest")
    changes = (
        (b"  push:\n", b"  push:\n    branches: [main]\n"),
        (
            b"  pull_request:\n",
            b"  pull_request:\n"
            b"    types: [opened, synchronize, reopened, ready_for_review, converted_to_draft]\n",
        ),
        (
            b"jobs:\n",
            b"# Newer runs replace obsolete runs for the same PR/ref, without cancelling a\n"
            b"# manual full verification or another PR's checks.\n"
            b"concurrency:\n"
            b"  group: ${{ github.workflow }}-${{ github.event_name }}-"
            b"${{ github.event.pull_request.number || github.ref }}\n"
            b"  cancel-in-progress: true\n\n"
            b"jobs:\n",
        ),
    )
    for before, after in changes:
        require(old.count(before) == 1, "ci_transition_shape")
        old = old.replace(before, after)
    before = b"        os: [ubuntu-latest, macos-latest]\n"
    after = (
        b"        os: ${{ fromJSON(github.event_name == 'pull_request' && "
        b"github.event.pull_request.draft && '[\"ubuntu-latest\"]' || "
        b'\'["ubuntu-latest","macos-latest"]\') }}\n'
    )
    require(old.count(before) == 2, "ci_transition_shape")
    return old.replace(before, after)


def verify_frozen_inputs(root: Path) -> dict[str, object]:
    """Verify all 192 original inputs, archived CI and original M2E evidence.

    No override arguments, changed-digest wildcard or alternate approval source.
    CI is checked against both immutable bytes and the exact approved patch.
    """
    lock_raw = _read(root, LOCK)
    require(_digest(lock_raw) == LOCK_SHA256, "original_lock_digest")
    lock = _object(lock_raw)
    attestation_raw = _read(root, COMPAT + "attestation.json")
    require(_digest(attestation_raw) == ATTESTATION_SHA256, "attestation_digest")
    attestation = _object(attestation_raw)
    require(attestation["original_lock_sha256"] == LOCK_SHA256, "compatibility_binding")
    files = lock["files_sha256"]
    assert isinstance(files, dict)
    require(len(files) == 192, "original_input_count")
    original_ci = _read(root, COMPAT + "ci-before.yml")
    require(_digest(original_ci) == OLD_CI_SHA256, "original_ci_digest")
    require(files[".github/workflows/ci.yml"] == OLD_CI_SHA256, "compatibility_binding")
    current_ci = _read(root, ".github/workflows/ci.yml")
    require(
        _digest(current_ci) == NEW_CI_SHA256 and current_ci == approved_ci_transition(original_ci),
        "approved_ci_digest",
    )
    for relative, digest in files.items():
        require(isinstance(relative, str) and isinstance(digest, str), "original_input_shape")
        if relative != ".github/workflows/ci.yml":
            require(_digest(_read(root, relative)) == digest, "frozen_input_digest")
    historical = attestation["historical_evidence_sha256"]
    assert isinstance(historical, dict)
    for relative, digest in historical.items():
        require(_digest(_read(root, relative)) == digest, "historical_evidence_digest")
    freeze = _object(_read(root, DEMO + "verification/m2e/freeze-manifest.json"))
    maintenance = freeze["maintenance_tools_sha256"]
    assert isinstance(maintenance, dict)
    require(len(maintenance) == 3, "maintenance_tool_count")
    for relative, digest in maintenance.items():
        require(_digest(_read(root, relative)) == digest, "maintenance_tool_digest")
    return {
        "status": "PASS",
        "original_inputs": 192,
        "unchanged_non_ci_inputs": 191,
        "retained_historical_evidence": len(historical),
        "retained_maintenance_tools": len(maintenance),
        "old_ci_sha256": OLD_CI_SHA256,
        "integrated_ci_sha256": NEW_CI_SHA256,
        "original_lock_sha256": LOCK_SHA256,
        "attestation_sha256": ATTESTATION_SHA256,
        "private_gold_loaded": False,
        "formal_scoring": "NOT_AUTHORIZED_NOT_RUN",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M2E public integrity only; no scoring or vault")
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_frozen_inputs(args.repository_root), sort_keys=True))
    except (FixtureError, OSError):
        print(json.dumps({"status": "error", "error_code": "frozen_input_integrity"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
