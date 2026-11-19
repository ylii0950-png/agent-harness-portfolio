import runpy

import pytest


def test_todo_demo_uses_own_loop_and_verifies_actual_plan_and_file_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_module("agent_harness_lab.todo_demo", run_name="__main__")
    output = capsys.readouterr().out
    assert "APPROVAL_SOURCE scripted Mock approval; all 6 mutations used ASK" in output
    assert "todo_write read_source=pending, write_receipt=pending" in output
    assert "todo_write read_source=in_progress, write_receipt=pending" in output
    assert "todo_write read_source=completed, write_receipt=in_progress" in output
    assert "todo_read read_source=completed, write_receipt=completed" in output
    assert "PLAN completed=2/2; actual receipt verified=true" in output
    assert "TRACE created=1 updated=4; descriptions omitted" in output
    assert output.endswith("STOP completed (model_rounds=10, tool_calls=9)\n")
