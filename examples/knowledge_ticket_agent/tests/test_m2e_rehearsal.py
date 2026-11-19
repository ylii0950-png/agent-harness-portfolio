"""All records below are public arithmetic examples, never sealed queries/Gold."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from knowledge_ticket_agent.evaluation_rehearsal import (
    SCOPE,
    Configuration,
    Observation,
    Ratio,
    RehearsalInputs,
    canonical,
    grade,
    public_fixture,
    run_rehearsal,
    scripted_observations,
    validate_inputs,
    verify_saved,
)
from knowledge_ticket_agent.evidence import EvidenceUnit
from knowledge_ticket_agent.strict import FixtureError

from agent_harness_lab import WorkspaceFiles


def inputs() -> RehearsalInputs:
    return RehearsalInputs(public_fixture(), scripted_observations())


def test_hand_computable_metrics_and_fixed_denominators() -> None:
    report = grade(inputs())
    assert report.scope == SCOPE and report.formal_scoring == "NOT_RUN"
    assert report.positive_denominator == 18 and report.negative_denominator == 6
    assert report.hit_at_3 == Ratio(1, 1)
    assert report.macro_recall_at_3 == Ratio(5, 6)
    assert report.mrr_at_3 == Ratio(5, 6)
    assert report.required_evidence_coverage == Ratio(5, 6)
    assert report.full_evidence_case_rate == Ratio(2, 3)
    assert report.negative_behavior == Ratio(5, 6)
    assert (
        report.model_requests == 0 and report.api_token_usage is None and report.actual_cost is None
    )


def test_equivalent_chunks_do_not_inflate_unit_denominator() -> None:
    result = grade(inputs()).results[1]
    assert result.recall == Ratio(1, 1) and result.reciprocal_rank == Ratio(1, 1)
    assert result.required_evidence_complete


def test_partial_complement_bundle_earns_no_unit() -> None:
    result = grade(inputs()).results[2]
    assert result.hit == Ratio(1, 1) and result.recall == Ratio(1, 2)
    assert not result.required_evidence_complete


def test_complete_bundle_and_independent_policy_both_required() -> None:
    source = inputs()
    observations = list(source.observations)
    observations[2] = replace(observations[2], ranking=("SYNTH-c1", "SYNTH-c2", "SYNTH-c3"))
    report = grade(replace(source, observations=tuple(observations)))
    assert report.results[2].recall == Ratio(1, 1)
    assert report.required_evidence_coverage == Ratio(31, 36)
    assert report.full_evidence_case_rate == Ratio(13, 18)


def test_errors_and_empty_results_keep_denominators() -> None:
    source = inputs()
    observations = list(source.observations)
    observations[0] = replace(observations[0], error="synthetic_error")
    observations[1] = replace(observations[1], ranking=())
    observations[18] = replace(observations[18], error="synthetic_error")
    report = grade(replace(source, observations=tuple(observations)))
    assert report.positive_denominator == 18 and report.negative_denominator == 6
    assert report.hit_at_3 == Ratio(8, 9)
    assert report.macro_recall_at_3 == Ratio(13, 18)
    assert report.negative_behavior == Ratio(2, 3)
    assert report.results[0].status == "error" and report.results[18].status == "error"


@pytest.mark.parametrize(
    "configuration",
    [
        replace(Configuration(), version="wrong"),
        replace(Configuration(), metrics_version="wrong"),
        replace(Configuration(), top_k=4),
        replace(Configuration(), positive_denominator=17),
        replace(Configuration(), negative_denominator=5),
        replace(Configuration(), scope="formal"),
    ],
)
def test_changed_configuration_fails_closed(configuration: Configuration) -> None:
    source = inputs()
    with pytest.raises(FixtureError, match="rehearsal_fixture_binding"):
        grade(replace(source, fixture=replace(source.fixture, configuration=configuration)))


def test_missing_evidence_gold_refused() -> None:
    source = inputs()
    cases = list(source.fixture.cases)
    cases[0] = replace(cases[0], units=())
    with pytest.raises(FixtureError, match="rehearsal_fixture_binding"):
        grade(replace(source, fixture=replace(source.fixture, cases=tuple(cases))))


def test_changed_gold_or_fake_chunk_refused() -> None:
    source = inputs()
    cases = list(source.fixture.cases)
    cases[0] = replace(cases[0], units=(EvidenceUnit("single", (("private-chunk",),)),))
    with pytest.raises(FixtureError, match="rehearsal_fixture_binding"):
        grade(replace(source, fixture=replace(source.fixture, cases=tuple(cases))))


@pytest.mark.parametrize("change", ["missing", "reordered", "duplicate", "unknown"])
def test_observation_set_and_order_are_exact(change: str) -> None:
    source = inputs()
    rows = list(source.observations)
    if change == "missing":
        rows.pop()
    elif change == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "duplicate":
        rows[1] = rows[0]
    else:
        rows[0] = replace(rows[0], case_id="unknown")
    with pytest.raises(FixtureError, match="rehearsal_(missing_observation|observation_order)"):
        grade(replace(source, observations=tuple(rows)))


@pytest.mark.parametrize(
    "ranking",
    [("SYNTH-c1",) * 2, ("unknown",), ("SYNTH-c1", "SYNTH-c2", "SYNTH-c3", "SYNTH-noise")],
)
def test_invalid_rankings_refused(ranking: tuple[str, ...]) -> None:
    source = inputs()
    observations = (replace(source.observations[0], ranking=ranking), *source.observations[1:])
    with pytest.raises(FixtureError, match="ranking|rehearsal_chunk"):
        grade(replace(source, observations=observations))


def test_negative_cases_are_behavior_checks_not_empty_search_claims() -> None:
    source = inputs()
    observations = list(source.observations)
    observations[22] = replace(observations[22], behavior="claimed_execution")
    report = grade(replace(source, observations=tuple(observations)))
    assert report.negative_behavior == Ratio(2, 3)
    assert report.results[22].negative_behavior_passed is False
    with pytest.raises(FixtureError, match="negative_ranking"):
        validate_inputs(
            replace(
                source,
                observations=(
                    *source.observations[:18],
                    replace(source.observations[18], ranking=("SYNTH-c1",)),
                    *source.observations[19:],
                ),
            )
        )


def test_saved_report_is_complete_and_recoverable_without_regrading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    report = run_rehearsal(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}

    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Recovery must not grade or rank")

    monkeypatch.setattr("knowledge_ticket_agent.evaluation_rehearsal.grade", denied)
    assert verify_saved(root) == report
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert len(report.results) == 24
    assert len({p.case_id for p in report.results}) == 24


def test_duplicate_execution_never_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "run"
    run_rehearsal(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FixtureError, match="rehearsal_already_started"):
        run_rehearsal(root)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_partial_write_failure_preserves_start_and_blocks_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    original = WorkspaceFiles.write_text

    def fail_report(self: WorkspaceFiles, path: str, text: str, *, overwrite: bool = False) -> None:
        if path == "report.json":
            raise OSError("synthetic interruption")
        original(self, path, text, overwrite=overwrite)

    monkeypatch.setattr(WorkspaceFiles, "write_text", fail_report)
    with pytest.raises(OSError, match="synthetic interruption"):
        run_rehearsal(root)
    assert (root / "started.json").exists() and not (root / "completed.json").exists()
    with pytest.raises(FixtureError, match="rehearsal_already_started"):
        run_rehearsal(root)
    with pytest.raises(FixtureError, match="rehearsal_incomplete"):
        verify_saved(root)


@pytest.mark.parametrize("name", ["inputs.json", "report.json", "completed.json", "started.json"])
def test_saved_bytes_tampering_fails_closed(tmp_path: Path, name: str) -> None:
    root = tmp_path / "run"
    run_rehearsal(root)
    record = json.loads((root / name).read_text())
    if name == "inputs.json":
        record["fixture"]["configuration"]["version"] = "changed"
    elif name == "report.json":
        record["mrr_at_3"] = {"numerator": 1, "denominator": 1}
    else:
        record["version"] = "changed"
    (root / name).write_text(canonical(record))
    with pytest.raises(FixtureError):
        verify_saved(root)


def test_partial_report_or_extra_file_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    run_rehearsal(root)
    (root / "extra.json").write_text("{}")
    with pytest.raises(FixtureError, match="rehearsal_incomplete"):
        verify_saved(root)


def test_symlink_output_directory_and_parent_denied(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    for root in (link, link / "new"):
        with pytest.raises(FixtureError, match="rehearsal_symlink"):
            run_rehearsal(root)
    assert list(target.iterdir()) == []


def test_relative_or_git_internal_path_denied(tmp_path: Path) -> None:
    for root in (Path("relative"), tmp_path / ".git" / "run", tmp_path / ".." / "run"):
        with pytest.raises(FixtureError, match="rehearsal_path"):
            run_rehearsal(root)


def test_symlink_report_and_oversized_file_denied(tmp_path: Path) -> None:
    root = tmp_path / "run"
    run_rehearsal(root)
    report = root / "report.json"
    outside = tmp_path / "outside.json"
    report.rename(outside)
    report.symlink_to(outside)
    from agent_harness_lab import FileToolError

    with pytest.raises(FileToolError):
        verify_saved(root)
    report.unlink()
    report.write_text(" " * (128 * 1024 + 1))
    with pytest.raises(FileToolError):
        verify_saved(root)


@pytest.mark.parametrize("flag", ["--score", "--formal", "--vault", "--gold-root"])
def test_cli_refuses_formal_scoring_and_private_input_flags(tmp_path: Path, flag: str) -> None:
    root = tmp_path / "no-run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_ticket_agent.evaluation_rehearsal",
            "--rehearse",
            str(root),
            flag,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and not root.exists()


def test_cli_public_rehearsal_and_read_only_verify(tmp_path: Path) -> None:
    root = tmp_path / "run"
    for option in ("--rehearse", "--verify"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "knowledge_ticket_agent.evaluation_rehearsal",
                option,
                str(root),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        record = json.loads(result.stdout)
        assert record["scope"] == SCOPE and record["formal_scoring"] == "NOT_RUN"
        assert record["cases"] == 24
        assert str(root) not in result.stdout
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    repeated = subprocess.run(
        [
            sys.executable,
            "-m",
            "knowledge_ticket_agent.evaluation_rehearsal",
            "--rehearse",
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    assert repeated.returncode == 1
    assert json.loads(repeated.stdout)["error_code"] == "rehearsal_already_started"
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_rehearsal_never_invokes_retrieval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("Rehearsal must not rank any queries")

    monkeypatch.setattr("knowledge_ticket_agent.retrieval.Retriever.search", denied)
    monkeypatch.setattr(
        "knowledge_ticket_agent.retrieval_strategies.StrategyRetriever.search", denied
    )
    monkeypatch.setattr("knowledge_ticket_agent.retrieval.lexical_rank", denied)
    report = run_rehearsal(tmp_path / "run")
    assert report.scope == SCOPE and report.formal_scoring == "NOT_RUN"
    assert "query" not in canonical(asdict(inputs()))


def test_observation_errors_do_not_disclose_parameters(tmp_path: Path) -> None:
    source = inputs()
    wrong = Observation(
        source.observations[0].case_id, ("private-sensitive-target",), "never-output"
    )
    with pytest.raises(FixtureError) as error:
        grade(replace(source, observations=(wrong, *source.observations[1:])))
    assert str(error.value) == "rehearsal_chunk"
