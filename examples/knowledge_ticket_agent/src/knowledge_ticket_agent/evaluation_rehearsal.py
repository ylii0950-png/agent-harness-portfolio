"""Public synthetic protocol rehearsal, never a held-out retrieval runner.

There is no Gold path, key, query, ranker or formal-scoring authorization input.
The exclusive run directory is a process-level repeat guard, not an ACL or a
same-user tamper-proof log. Recovery verifies saved bytes; it never reruns grading.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path

from agent_harness_lab import FileToolError, WorkspaceFiles
from knowledge_ticket_agent.evidence import METRICS_VERSION, EvidenceUnit, minimum_cover, score
from knowledge_ticket_agent.strict import FixtureError, decode, parse_json, require

VERSION = "lqd-public-rehearsal-1.0.1"
SCOPE = "REHEARSAL_SYNTHETIC_ONLY"
MAX_BYTES = 128 * 1024
CHUNKS = frozenset({"SYNTH-c1", "SYNTH-c2", "SYNTH-c3", "SYNTH-noise"})
BEHAVIORS = (
    "no_evidence",
    "expired_not_current",
    "unknown_fact",
    "faq_navigation_only",
    "approval_required",
    "unknown_fact",
)


@dataclass(frozen=True)
class Configuration:
    version: str = VERSION
    scope: str = SCOPE
    metrics_version: str = METRICS_VERSION
    top_k: int = 3
    positive_denominator: int = 18
    negative_denominator: int = 6


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    positive: bool
    units: tuple[EvidenceUnit, ...]
    expected_behavior: str


@dataclass(frozen=True)
class PublicFixture:
    configuration: Configuration
    cases: tuple[PublicCase, ...]


@dataclass(frozen=True)
class Observation:
    case_id: str
    ranking: tuple[str, ...]
    behavior: str
    error: str | None = None


@dataclass(frozen=True)
class RehearsalInputs:
    fixture: PublicFixture
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    positive: bool
    status: str
    hit: Ratio
    recall: Ratio
    reciprocal_rank: Ratio
    required_evidence_complete: bool
    negative_behavior_passed: bool | None


@dataclass(frozen=True)
class Report:
    version: str
    scope: str
    formal_scoring: str
    input_sha256: str
    configuration_sha256: str
    positive_denominator: int
    negative_denominator: int
    hit_at_3: Ratio
    macro_recall_at_3: Ratio
    mrr_at_3: Ratio
    required_evidence_coverage: Ratio
    full_evidence_case_rate: Ratio
    negative_behavior: Ratio
    results: tuple[CaseResult, ...]
    model_requests: int = 0
    api_token_usage: None = None
    actual_cost: None = None


@dataclass(frozen=True)
class Receipt:
    version: str
    scope: str
    input_sha256: str
    report_sha256: str


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def public_fixture() -> PublicFixture:
    """Deliberately repetitive arithmetic examples; no policy texts or queries."""
    patterns = (
        (EvidenceUnit("single", (("SYNTH-c1",),)),),
        (EvidenceUnit("equivalent", (("SYNTH-c1",), ("SYNTH-c2",))),),
        (
            EvidenceUnit("complement", (("SYNTH-c1", "SYNTH-c2"),)),
            EvidenceUnit("independent", (("SYNTH-c3",),)),
        ),
    )
    cases = tuple(
        PublicCase(f"SYNTH-P{i + 1:02}", True, patterns[i % 3], "not_a_business_task")
        for i in range(18)
    ) + tuple(PublicCase(f"SYNTH-N{i + 1:02}", False, (), b) for i, b in enumerate(BEHAVIORS))
    return PublicFixture(Configuration(), cases)


def scripted_observations() -> tuple[Observation, ...]:
    """Fixed ranks are supplied examples, not produced by any retrieval strategy."""
    ranks = (
        ("SYNTH-noise", "SYNTH-c1"),
        ("SYNTH-c2",),
        ("SYNTH-c1", "SYNTH-noise", "SYNTH-c3"),
    )
    return tuple(
        Observation(f"SYNTH-P{i + 1:02}", ranks[i % 3], "not_a_business_task") for i in range(18)
    ) + tuple(
        Observation(f"SYNTH-N{i + 1:02}", (), b if i != 5 else "incorrect_claim")
        for i, b in enumerate(BEHAVIORS)
    )


def validate_inputs(inputs: RehearsalInputs) -> None:
    # This tool intentionally cannot accept a new or real held-out release.
    require(inputs.fixture == public_fixture(), "rehearsal_fixture_binding")
    cases = inputs.fixture.cases
    require(len(inputs.observations) == len(cases), "rehearsal_missing_observation")
    require(
        tuple(x.case_id for x in inputs.observations) == tuple(c.case_id for c in cases),
        "rehearsal_observation_order",
    )
    for case, item in zip(cases, inputs.observations, strict=True):
        require(len(item.ranking) <= 3 and len(set(item.ranking)) == len(item.ranking), "ranking")
        require(set(item.ranking) <= CHUNKS, "rehearsal_chunk")
        require(len(item.behavior) <= 64 and item.error in {None, "synthetic_error"}, "observation")
        if case.positive:
            require(bool(case.units) and minimum_cover(case.units) <= 3, "evidence")
        else:
            require(not item.ranking, "negative_ranking")


def _ratio(value: Fraction) -> Ratio:
    return Ratio(value.numerator, value.denominator)


def grade(inputs: RehearsalInputs) -> Report:
    """Pure fixed-fixture arithmetic; errors retain their declared denominators."""
    validate_inputs(inputs)
    results = []
    totals = [Fraction(0), Fraction(0), Fraction(0)]
    coverage = negatives = 0
    for case, item in zip(inputs.fixture.cases, inputs.observations, strict=True):
        if case.positive:
            metrics = score(case.units, () if item.error else item.ranking)
            values = (
                Fraction(int(metrics.hit)),
                Fraction(len(metrics.covered_units), len(case.units)),
                Fraction(1, metrics.first_relevant_rank)
                if metrics.first_relevant_rank
                else Fraction(0),
            )
            # Cross-check existing metric arithmetic without modifying its semantics.
            require(float(values[1]) == metrics.recall, "recall_arithmetic")
            require(float(values[2]) == metrics.reciprocal_rank, "mrr_arithmetic")
            complete = not metrics.missing_units
            coverage += int(complete)
            totals = [a + b for a, b in zip(totals, values, strict=True)]
            results.append(
                CaseResult(
                    case.case_id,
                    True,
                    "error" if item.error else "observed",
                    _ratio(values[0]),
                    _ratio(values[1]),
                    _ratio(values[2]),
                    complete,
                    None,
                )
            )
        else:
            passed = item.error is None and item.behavior == case.expected_behavior
            negatives += int(passed)
            results.append(
                CaseResult(
                    case.case_id,
                    False,
                    "error" if item.error else "observed",
                    Ratio(0, 1),
                    Ratio(0, 1),
                    Ratio(0, 1),
                    False,
                    passed,
                )
            )
    config = inputs.fixture.configuration
    return Report(
        VERSION,
        SCOPE,
        "NOT_RUN",
        digest(asdict(inputs)),
        digest(asdict(config)),
        config.positive_denominator,
        config.negative_denominator,
        _ratio(totals[0] / config.positive_denominator),
        _ratio(totals[1] / config.positive_denominator),
        _ratio(totals[2] / config.positive_denominator),
        # Every unit is required in this public fixture: REC is macro unit coverage,
        # whereas complete-case rate requires every unit of that case to be present.
        _ratio(totals[1] / config.positive_denominator),
        _ratio(Fraction(coverage, config.positive_denominator)),
        _ratio(Fraction(negatives, config.negative_denominator)),
        tuple(results),
    )


def _root(root: Path) -> None:
    require(
        root.is_absolute() and all(part not in {"..", ".git"} for part in root.parts),
        "rehearsal_path",
    )
    require(not any(p.is_symlink() for p in (root, *root.parents)), "rehearsal_symlink")


def run_rehearsal(root: Path) -> Report:
    """Exclusive new output directory blocks repeats, including partial failures."""
    _root(root)
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        raise FixtureError("rehearsal_already_started") from None
    inputs = RehearsalInputs(public_fixture(), scripted_observations())
    with WorkspaceFiles(root, allow_write=True, max_bytes=MAX_BYTES) as files:
        files.write_text("started.json", canonical({"scope": SCOPE, "version": VERSION}))
        files.write_text("inputs.json", canonical(asdict(inputs)))
        report = grade(inputs)
        files.write_text("report.json", canonical(asdict(report)))
        receipt = Receipt(VERSION, SCOPE, digest(asdict(inputs)), digest(asdict(report)))
        files.write_text("completed.json", canonical(asdict(receipt)))
    return report


def verify_saved(root: Path) -> Report:
    """Read-only recovery; no grading, ranking, overwrite, or budget reset."""
    _root(root)
    require(
        {p.name for p in islice(root.iterdir(), 5)}
        == {"started.json", "inputs.json", "report.json", "completed.json"},
        "rehearsal_incomplete",
    )
    with WorkspaceFiles(root, max_bytes=MAX_BYTES) as files:
        records = {}
        for name in ("started.json", "inputs.json", "report.json", "completed.json"):
            raw = files.read_text(name)
            value = parse_json(raw)
            require(canonical(value) == raw, "rehearsal_noncanonical_bytes")
            records[name] = value
        started = records["started.json"]
        inputs_raw = records["inputs.json"]
        report_raw = records["report.json"]
        receipt = decode(Receipt, records["completed.json"])
    require(started == {"scope": SCOPE, "version": VERSION}, "rehearsal_version")
    inputs = decode(RehearsalInputs, inputs_raw)
    validate_inputs(inputs)
    report = decode(Report, report_raw)
    require(receipt.version == VERSION and receipt.scope == SCOPE, "rehearsal_version")
    require(
        receipt.input_sha256 == digest(inputs_raw) == report.input_sha256
        and receipt.report_sha256 == digest(report_raw),
        "rehearsal_integrity",
    )
    require(
        report.version == VERSION
        and report.scope == SCOPE
        and report.formal_scoring == "NOT_RUN"
        and report.configuration_sha256 == digest(asdict(inputs.fixture.configuration))
        and report.positive_denominator == 18
        and report.negative_denominator == 6
        and tuple(x.case_id for x in report.results)
        == tuple(c.case_id for c in inputs.fixture.cases)
        and report.model_requests == 0
        and report.api_token_usage is None
        and report.actual_cost is None,
        "rehearsal_report_binding",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Public synthetic protocol rehearsal; no formal score"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rehearse", type=Path, metavar="NEW_DIRECTORY")
    mode.add_argument("--verify", type=Path, metavar="EXISTING_DIRECTORY")
    args = parser.parse_args()
    try:
        report = run_rehearsal(args.rehearse) if args.rehearse else verify_saved(args.verify)
        print(
            json.dumps(
                {
                    "scope": report.scope,
                    "formal_scoring": "NOT_RUN",
                    "cases": len(report.results),
                    "status": "verified",
                }
            )
        )
    except (FixtureError, FileToolError, OSError) as exc:
        code = exc.code if isinstance(exc, FixtureError) else "rehearsal_unavailable"
        print(json.dumps({"status": "error", "error_code": code, "formal_scoring": "NOT_RUN"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
