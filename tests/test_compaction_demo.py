import pytest

from agent_harness_lab.compaction_demo import main


def test_mock_demo_verifies_real_loop_context_state_and_denial(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()
    output = capsys.readouterr().out
    assert "STOP completed (model_rounds=10, tool_calls=9)" in output
    assert "todo_completed=1 skill_low_trust=true" in output
    assert "actual_file_absent=true" in output
    assert "actual_tokens=unknown actual_cost=unknown" in output
