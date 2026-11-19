"""Public synthetic fixtures only. Never use the real formal vault or release."""

import json
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import InvalidTag
from knowledge_ticket_agent.chunking import CHUNKER_VERSION, build_index
from knowledge_ticket_agent.evaluation_maintenance import (
    CONFIG_SHA256,
    VERSION,
    Annotation,
    HeldoutCase,
    HeldoutDataset,
    payload,
)
from knowledge_ticket_agent.evidence import METRICS_VERSION, EvidenceUnit, map_anchor, score
from knowledge_ticket_agent.formal_ledger import Ledger, canonical, digest
from knowledge_ticket_agent.formal_operator import execute, location, recover, recover_saved
from knowledge_ticket_agent.formal_scoring import (
    METHODS,
    METRICS,
    PROTOCOL,
    method_identity,
    observation,
    run,
    summarize,
    verify_records,
)
from knowledge_ticket_agent.gold import Anchor
from knowledge_ticket_agent.retrieval import Retriever, SearchResult, catalog
from knowledge_ticket_agent.schemas import Category, Dataset
from knowledge_ticket_agent.strict import FixtureError


@pytest.fixture
def artificial(dataset: Dataset) -> HeldoutDataset:
    index = build_index(dataset)
    cases = []
    kinds = (
        "no_policy_evidence",
        "no_policy_evidence",
        "expired_only",
        "unverifiable_fact",
        "faq_not_authority",
        "execution_not_authorized",
    )
    for positive, count in ((True, 18), (False, 6)):
        for i in range(count):
            category = (Category.BILLING, Category.ACCESS, Category.EXPORT)[
                i // (6 if positive else 2)
            ]
            allowed = catalog(dataset, snapshot="normal", category=category).authoritative_refs
            doc = next(d for d in dataset.documents if d.metadata.ref in allowed)
            section = doc.sections[0]
            anchor = Anchor(
                "public_synthetic",
                doc.metadata.ref,
                section.section_id,
                section.start_char,
                section.end_char,
                doc.text[section.start_char : section.end_char],
            )
            mapped = map_anchor(anchor, index.chunks, doc.text)
            unit = Annotation(anchor.unit_id, (anchor,), mapped.alternatives)
            cases.append(
                HeldoutCase(
                    f"M2E-{'P' if positive else 'N'}{i + 1:02}",
                    f"Synthetic public validator example number {positive} {i} policy restriction.",
                    category,
                    "normal",
                    positive,
                    "public_synthetic",
                    (unit,),
                    (unit.unit_id,),
                    None if positive else kinds[i],
                    "Synthetic expectation",
                    "Not real heldout data",
                )
            )
    return HeldoutDataset(
        VERSION,
        True,
        "authored_fiction",
        dataset.manifest.corpus_version,
        dataset.manifest_sha256,
        index.sha256,
        CHUNKER_VERSION,
        CONFIG_SHA256,
        METRICS_VERSION,
        tuple(cases),
    )


def test_complete_rehearsal_and_recovery(
    tmp_path: Path, dataset: Dataset, artificial: HeldoutDataset
) -> None:
    ledger = Ledger(tmp_path / "ledger")
    with ledger.transaction() as fd:
        state = ledger.start(fd, {"release": "SYNTHETIC_ONLY"})
        report = run(dataset, artificial, lambda value: ledger.save(fd, value))
        ledger.save(fd, {"event": "REPORT", "report": report})
        ledger.finish(fd, "COMPLETED", report_sha256=digest(canonical(report)))
        assert len(ledger.records(fd)) == 74
        rows = report["rows"]
        assert isinstance(rows, list) and len(rows) == 72
        assert state["status"] == "STARTED"
    # Recover uses only saved encrypted bytes; no retrieval, no Gold reopening.
    common = tmp_path / "common"
    common.mkdir()
    ledger.root.rename(common / "m2f-formal-private")
    result = recover_saved(common, expected_gold_sha256=digest(payload(artificial)))
    assert result["recovery"] == "PASS_NO_RERANK"
    assert result["status"] == "COMPLETED"
    summary = report["summary"]
    assert isinstance(summary, dict)
    methods = summary["methods"]
    assert isinstance(methods, dict)
    for result in methods.values():
        assert result["positive_denominator"] == 18
        assert result["negative_denominator"] == 6
        assert len(result["negative_results"]) == 6
    assert artificial.cases[0].query.encode() not in b"".join(
        p.read_bytes() for p in common.rglob("record-*.bin")
    )


@pytest.mark.parametrize("method", METHODS)
def test_ranking_errors_keep_all_denominators(
    dataset: Dataset, artificial: HeldoutDataset, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    from knowledge_ticket_agent import formal_scoring

    original = formal_scoring.retrieve

    def bad(data: Dataset, case: HeldoutCase, name: str) -> SearchResult:
        if name == method:
            raise RuntimeError("PRIVATE QUERY MUST NOT LEAK")
        return original(data, case, name)

    monkeypatch.setattr(formal_scoring, "retrieve", bad)
    saved: list[dict[str, object]] = []
    report = run(dataset, artificial, saved.append)
    public = report["summary"]
    assert isinstance(public, dict)
    methods = public["methods"]
    assert isinstance(methods, dict)
    result = methods[method]
    assert result["errors"] == 24
    assert all(m["exact"] == "0" for m in result["metrics"].values())
    assert result["negative_passed"] == 0
    assert "PRIVATE QUERY" not in json.dumps(report)
    assert len(saved) == 73


def test_exact_summary_fractions_and_missing_report() -> None:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for i in range(24):
            rows.append(
                {
                    "method": method,
                    "query_id": str(i),
                    "positive": i < 18,
                    "status": "OK",
                    "metrics": dict.fromkeys(METRICS, "5/6") if i < 18 else {},
                    "negative_kind": "synthetic",
                    "negative_pass": i % 2 == 0,
                    "faq_hits": 0,
                }
            )
    result: Any = summarize(rows)
    assert result["methods"]["A1"]["metrics"]["mrr_at_3"]["exact"] == "5/6"
    with pytest.raises(FixtureError):
        summarize(rows[:-1])


@pytest.mark.parametrize("mutation", ["version", "config", "count", "gold"])
def test_version_and_gold_mismatch_before_any_search(
    dataset: Dataset, artificial: HeldoutDataset, mutation: str
) -> None:
    if mutation == "version":
        artificial = replace(artificial, version="unapproved")
    elif mutation == "config":
        artificial = replace(artificial, config_sha256="0" * 64)
    elif mutation == "count":
        artificial = replace(artificial, cases=artificial.cases[:-1])
    else:
        case = artificial.cases[0]
        unit = replace(case.units[0], alternatives=(("unknown",),))
        artificial = replace(
            artificial, cases=(replace(case, units=(unit,)), *artificial.cases[1:])
        )
    saved: list[dict[str, object]] = []
    with pytest.raises(FixtureError):
        run(dataset, artificial, saved.append)
    assert saved == []


def test_concurrent_single_claim(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    Ledger(root)
    code = """
import sys
from pathlib import Path
from knowledge_ticket_agent.formal_ledger import Ledger
from knowledge_ticket_agent.strict import FixtureError
ledger = Ledger(Path(sys.argv[1]))
try:
    with ledger.transaction() as fd:
        ledger.start(fd, {"release": "PUBLIC_SYNTH"})
    print("STARTED")
except FixtureError:
    print("DENIED")
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    try:
        results = [process.communicate(timeout=15)[0].strip() for process in processes]
        assert all(process.returncode == 0 for process in processes)
        assert results.count("STARTED") == 1 and results.count("DENIED") == 3
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("terminal", ["STARTED", "FAILED", "COMPLETED"])
def test_no_retry_even_new_instance(tmp_path: Path, terminal: str) -> None:
    root = tmp_path / "ledger"
    ledger = Ledger(root)
    with ledger.transaction() as fd:
        ledger.start(fd, {"release": "SYNTHETIC"})
        if terminal != "STARTED":
            ledger.finish(fd, terminal)
    with Ledger(root).transaction() as fd, pytest.raises(FixtureError):
        ledger.start(fd, {"release": "different_output_directory"})


def test_abrupt_interruption_recovery_never_reruns(
    tmp_path: Path, artificial: HeldoutDataset
) -> None:
    root = tmp_path / "m2f-formal-private"
    code = """
import os, signal, sys
from pathlib import Path
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.evaluation_maintenance import parse_payload
from knowledge_ticket_agent.formal_ledger import Ledger
from knowledge_ticket_agent.formal_scoring import run
ledger = Ledger(Path(sys.argv[1]))
with ledger.transaction() as fd:
    ledger.start(fd, {"release": "SYNTHETIC"})
    def save(value):
        ledger.save(fd, value)
        os.kill(os.getpid(), signal.SIGKILL)
    run(load_dataset(), parse_payload(Path(sys.argv[2]).read_bytes()), save)
"""
    fixture = tmp_path / "public-synthetic.json"
    fixture.write_bytes(payload(artificial))
    process = subprocess.Popen([sys.executable, "-c", code, str(root), str(fixture)])
    process.wait(timeout=15)
    try:
        assert process.returncode == -signal.SIGKILL
        result = recover_saved(tmp_path, expected_gold_sha256=digest(payload(artificial)))
        assert result["status"] == "FAILED" and result["saved_encrypted_records"] == 1
        assert result["denominator_cells"] == 72
        ledger = Ledger(root)
        with ledger.transaction() as fd, pytest.raises(FixtureError):
            ledger.start(fd, {})
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("mutation", ["claim", "cipher", "key", "gap", "permission", "symlink"])
def test_ledger_tamper_rejected(tmp_path: Path, mutation: str) -> None:
    ledger = Ledger(tmp_path / "m2f-formal-private")
    with ledger.transaction() as fd:
        ledger.start(fd, {})
        ledger.save(fd, {"synthetic": True})
    if mutation == "claim":
        (ledger.root / "claim.bin").write_bytes(b"changed")
    elif mutation == "cipher":
        (ledger.root / "record-000.bin").write_bytes(b"broken")
    elif mutation == "key":
        (ledger.root / "result-key.bin").write_bytes(b"a" * 32)
    elif mutation == "gap":
        (ledger.root / "record-000.bin").rename(ledger.root / "record-002.bin")
    elif mutation == "permission":
        (ledger.root / "state.bin").chmod(0o644)
    else:
        (ledger.root / "state.bin").unlink()
        (ledger.root / "state.bin").symlink_to(ledger.root / "claim.bin")
    with pytest.raises((FixtureError, InvalidTag, OSError)):
        recover(tmp_path)


def test_partial_claim_and_pending_atomic_state_block_restart(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger")
    (ledger.root / "claim.bin").write_bytes(b"")
    (ledger.root / "claim.bin").chmod(0o600)
    with ledger.transaction() as fd, pytest.raises(FixtureError):
        ledger.start(fd, {})


def test_operator_rejects_ci_before_any_private_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    with pytest.raises(FixtureError, match="formal_local_only"):
        location(tmp_path)


def test_negative_assertions_are_retrieval_only(
    dataset: Dataset, artificial: HeldoutDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knowledge_ticket_agent import formal_scoring

    case = artificial.cases[-1]
    target = case.units[0].alternatives[0][0]
    retriever = Retriever(dataset)
    result = retriever.search("policy", category=case.category)
    # Exact synthetic target independent of query scores; never actual Gold.
    chunk = retriever.get_chunk(target, category=case.category)
    from knowledge_ticket_agent.retrieval import SearchHit

    hit = SearchHit(
        chunk.chunk_id,
        chunk.doc_ref,
        chunk.section_id,
        chunk.source_path,
        chunk.start_char,
        chunk.end_char,
        chunk.start_line,
        chunk.end_line,
        1.0,
        chunk.text[:600],
        len(chunk.text) > 600,
    )
    monkeypatch.setattr(
        formal_scoring,
        "retrieve",
        lambda *args: replace(result, method=method_identity("A1"), hits=(hit,)),
    )
    assert observation(dataset, case, "A1")["negative_pass"] is True
    no_policy = replace(case, negative_kind="no_policy_evidence")
    assert observation(dataset, no_policy, "A1")["negative_pass"] is False
    monkeypatch.setattr(
        formal_scoring,
        "retrieve",
        lambda *args: replace(result, method=method_identity("A1"), status="no_match", hits=()),
    )
    assert observation(dataset, no_policy, "A1")["negative_pass"] is True
    assert observation(dataset, case, "A1")["negative_pass"] is False


def test_or_and_hand_arithmetic() -> None:
    units = (
        EvidenceUnit("complement", (("a", "b"),)),
        EvidenceUnit("equivalent", (("c",), ("d",))),
    )
    partial = score(units, ("unrelated", "a", "d"))
    assert (partial.hit, partial.recall, partial.reciprocal_rank, partial.raw_chunk_recall) == (
        1,
        0.5,
        0.5,
        0.5,
    )
    complete = score(units, ("a", "b", "c"))
    assert complete.recall == 1 and complete.raw_chunk_recall == 0.75


@pytest.mark.parametrize(
    "mutation", ["ordinal", "reorder", "missing", "math", "metadata", "report"]
)
def test_saved_transcript_rejects_inconsistency(
    dataset: Dataset, artificial: HeldoutDataset, mutation: str
) -> None:
    values: list[dict[str, object]] = []
    report = run(dataset, artificial, values.append)
    values.append({"event": "REPORT", "report": report})
    values = json.loads(canonical(values))
    verify_records(dataset, values, complete=True)
    if mutation == "ordinal":
        values[1]["ordinal"] = 5
    elif mutation == "reorder":
        values[1], values[2] = values[2], values[1]
    elif mutation == "missing":
        values.pop(5)
    elif mutation == "report":
        values[-1]["report"] = {}
    else:
        row = values[1]["row"]
        assert isinstance(row, dict)
        if mutation == "metadata":
            row["index_sha256"] = "0" * 64
        else:
            row["metrics"] = dict.fromkeys(METRICS, "1/7")
    with pytest.raises(FixtureError):
        verify_records(dataset, values, complete=True)


def test_operator_same_production_flow_with_public_input(
    tmp_path: Path, artificial: HeldoutDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knowledge_ticket_agent import formal_operator

    calls = []

    def load(common: Path) -> HeldoutDataset:
        ledger = Ledger(common / "m2f-formal-private")
        assert json.loads((ledger.root / "state.bin").read_bytes())["status"] == "STARTED"
        calls.append("LOAD_AFTER_STARTED")
        return artificial

    monkeypatch.setattr(formal_operator, "preflight", lambda *args: {"release": "PUBLIC_SYNTHETIC"})
    monkeypatch.setattr(formal_operator, "_load_formal", load)
    result = execute(tmp_path, tmp_path)
    assert result["status"] == "COMPLETED" and calls == ["LOAD_AFTER_STARTED"]
    with pytest.raises(FixtureError, match="formal_already_claimed"):
        execute(tmp_path, tmp_path)
    assert calls == ["LOAD_AFTER_STARTED"]

    def deny(*args: object, **kwargs: object) -> None:
        raise AssertionError("Recovery must not rank or reopen formal inputs")

    monkeypatch.setattr(formal_operator, "_load_formal", deny)
    monkeypatch.setattr("knowledge_ticket_agent.formal_scoring.retrieve", deny)
    saved = recover_saved(tmp_path, expected_gold_sha256=digest(payload(artificial)))
    assert saved["summary"] == result["summary"]
    with pytest.raises(FixtureError, match="formal_saved_gold_digest"):
        recover(tmp_path)


def test_operator_interruption_retains_full_failed_report(
    tmp_path: Path, artificial: HeldoutDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knowledge_ticket_agent import formal_operator, formal_scoring

    monkeypatch.setattr(formal_operator, "preflight", lambda *args: {"release": "PUBLIC_SYNTHETIC"})
    monkeypatch.setattr(formal_operator, "_load_formal", lambda *args: artificial)
    original = formal_scoring.retrieve
    calls = 0

    def interrupt(*args: Any) -> SearchResult:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args)

    monkeypatch.setattr(formal_scoring, "retrieve", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, tmp_path)
    assert calls == 3
    result = recover_saved(tmp_path, expected_gold_sha256=digest(payload(artificial)))
    assert calls == 3 and result["status"] == "FAILED" and result["denominator_cells"] == 72
    assert (tmp_path / "m2f-formal-private/recovery-report.bin").is_file()


@pytest.mark.parametrize("mutation", ["missing_gate", "code_mismatch", "log_mismatch"])
def test_preflight_rejects_before_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from knowledge_ticket_agent import formal_operator

    receipt: dict[str, object] = {
        "status": "PASS",
        "protocol": PROTOCOL,
        "implementation": {},
        "checks": {},
    }
    monkeypatch.setattr(formal_operator, "verify_frozen_inputs", lambda *args: {})
    monkeypatch.setattr(formal_operator, "git", lambda *args: "")
    monkeypatch.setattr(formal_operator, "implementation", lambda *args: {})
    if mutation == "code_mismatch":
        receipt["implementation"] = {"changed": "0" * 64}
    elif mutation == "log_mismatch":
        receipt["checks"] = {
            name: {"exit_code": 0, "log": "artifacts/m2f-20261025/fake.log", "sha256": "0" * 64}
            for name in formal_operator.GATES
        }
    monkeypatch.setattr(formal_operator, "public_read", lambda *args: canonical(receipt))

    def deny(*args: object, **kwargs: object) -> None:
        raise AssertionError("No private custody read before gates")

    monkeypatch.setattr(formal_operator, "custody", deny)
    with pytest.raises(FixtureError):
        formal_operator.preflight(tmp_path, tmp_path)


def test_custody_missing_lock_does_not_initialize(tmp_path: Path) -> None:
    from knowledge_ticket_agent.formal_operator import read_vault

    root = tmp_path / "m2e-steward-private"
    root.mkdir(mode=0o700)
    with pytest.raises(FileNotFoundError), read_vault(tmp_path):
        raise AssertionError("No private input read without existing lock")
    assert list(root.iterdir()) == []


def test_recovery_record_swaps_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger")
    with ledger.transaction() as fd:
        ledger.start(fd, {})
        ledger.save(fd, {"public": 1})
        ledger.save(fd, {"public": 2})
        first, second = ledger.root / "record-000.bin", ledger.root / "record-001.bin"
        a, b = first.read_bytes(), second.read_bytes()
        first.write_bytes(b)
        second.write_bytes(a)
        with pytest.raises(InvalidTag):
            ledger.records(fd)


def test_artifact_scan_retains_history_and_requires_current_pairs(tmp_path: Path) -> None:
    from knowledge_ticket_agent.formal_operator import build_artifacts

    core = tmp_path / "dist"
    demo = tmp_path / "examples/knowledge_ticket_agent/dist"
    core.mkdir()
    demo.mkdir(parents=True)
    names = [
        core / "agent_harness_lab-0.9.0-py3-none-any.whl",
        core / "agent_harness_lab-0.9.0.tar.gz",
        demo / "knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl",
        demo / "knowledge_ticket_agent_demo-0.1.0.tar.gz",
    ]
    for path in names:
        path.write_bytes(b"public synthetic archive-name fixture")
    (core / ".gitignore").write_bytes(b"*")
    old = core / "agent_harness_lab-0.1.0.tar.gz"
    old.write_bytes(b"public retained historical archive")
    assert set(build_artifacts(tmp_path)) == {*names, old}
    names[-1].unlink()
    with pytest.raises(FixtureError, match="formal_artifacts"):
        build_artifacts(tmp_path)
