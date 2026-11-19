import json
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.integration_cases import mock_for
from agent_harness_lab.integration_suite import evaluate_integration
from agent_harness_lab.providers import MockProvider, ModelResponse


def batch(index: int, *calls: tuple[str, dict[str, Any]]) -> ModelResponse:
    return ModelResponse(
        tool_calls=tuple(
            ToolCall(f"{index}_{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)
        )
    )


def test_wrong_read_then_correct_overwrite_is_not_readback(tmp_path: Path) -> None:
    parent = MockProvider(
        [
            batch(0, ("delegate_task", {"task": "Load extract, read input.txt, return code."})),
            batch(
                1,
                ("write_file", {"path": "receipt.txt", "content": "WRONG\n"}),
                ("read_file", {"path": "receipt.txt"}),
                ("write_file", {"path": "receipt.txt", "content": "NOVA-52\n", "overwrite": True}),
            ),
            "done",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run",
        factory=lambda c, r: parent if c == "B" and r == "parent" else mock_for(c, r),
    )
    row = report["cases"][1]
    assert row["checks"]["receipt_exact"]
    assert not row["checks"]["parent_used_result"]
    assert row["status"] == "failed"


@pytest.mark.parametrize("path", ["missing.txt", "skills/extract/SKILL.md"])
def test_child_claim_after_failed_or_wrong_target_read_never_passes(
    tmp_path: Path,
    path: str,
) -> None:
    child = MockProvider(
        [
            batch(0, ("list_skills", {})),
            batch(1, ("load_skill", {"name": "extract"}), ("read_file", {"path": path})),
            "NOVA-52",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run",
        factory=lambda c, r: child if c == "B" and r == "child" else mock_for(c, r),
    )
    row = report["cases"][1]
    assert row["checks"]["receipt_exact"] and row["checks"]["structured_child_code"]
    assert not row["checks"]["child_read"]
    assert row["status"] == "failed"


def test_valid_child_and_final_readback_still_pass(tmp_path: Path) -> None:
    report = evaluate_integration(tmp_path / "run")
    assert report["grader_version"] == 2
    assert report["cases"][1]["status"] == "passed"


@pytest.mark.parametrize("overwrite", ["NOVA-52\n", "WRONG\n"])
def test_any_successful_later_overwrite_requires_a_new_readback(
    tmp_path: Path,
    overwrite: str,
) -> None:
    parent = MockProvider(
        [
            batch(0, ("delegate_task", {"task": "Read input using extract skill."})),
            batch(
                1,
                ("write_file", {"path": "receipt.txt", "content": "NOVA-52\n"}),
                ("read_file", {"path": "receipt.txt"}),
                ("write_file", {"path": "receipt.txt", "content": overwrite, "overwrite": True}),
            ),
            "done",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run",
        factory=lambda c, r: parent if c == "B" and r == "parent" else mock_for(c, r),
    )
    assert not report["cases"][1]["checks"]["parent_used_result"]
    assert report["cases"][1]["status"] == "failed"


def test_unpaired_or_wrong_content_is_not_a_child_read_proof() -> None:
    from agent_harness_lab.contracts import ToolResult
    from agent_harness_lab.integration_proofs import successful_read

    call = ToolCall("read", "read_file", '{"path":"input.txt"}')
    assert not successful_read(
        (call, ToolResult("other", "read_file", "CODE: NOVA-52\n")), "input.txt", "CODE: NOVA-52\n"
    )
    assert not successful_read(
        (call, ToolResult("read", "read_file", "WRONG\n")), "input.txt", "CODE: NOVA-52\n"
    )


def test_guessing_receipt_before_child_result_never_satisfies_parent_use(tmp_path: Path) -> None:
    parent = MockProvider(
        [
            batch(
                0,
                ("write_file", {"path": "receipt.txt", "content": "NOVA-52\n"}),
                ("read_file", {"path": "receipt.txt"}),
                (
                    "delegate_task",
                    {"task": "Discover extract skill and read input.txt; return code."},
                ),
            ),
            "done",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run",
        factory=lambda c, r: parent if c == "B" and r == "parent" else mock_for(c, r),
    )
    row = report["cases"][1]
    assert row["checks"]["receipt_exact"] and row["checks"]["child_read"]
    assert row["checks"]["structured_child_code"]
    assert not row["checks"]["parent_used_result"] and row["status"] == "failed"


def test_same_model_response_cannot_claim_to_have_observed_child_result(tmp_path: Path) -> None:
    parent = MockProvider(
        [
            batch(
                0,
                ("delegate_task", {"task": "Load extract and read input.txt; return code."}),
                ("write_file", {"path": "receipt.txt", "content": "NOVA-52\n"}),
                ("read_file", {"path": "receipt.txt"}),
            ),
            "done",
        ]
    )
    report = evaluate_integration(
        tmp_path / "run",
        factory=lambda c, r: parent if c == "B" and r == "parent" else mock_for(c, r),
    )
    row = report["cases"][1]
    assert row["checks"]["receipt_exact"] and row["checks"]["child_read"]
    assert not row["checks"]["parent_used_result"] and row["status"] == "failed"
