import runpy

import pytest


def test_actual_project_subagent_demo_completes_with_verified_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_module("agent_harness_lab.subagent_demo", run_name="__main__")
    text = capsys.readouterr().out
    assert "DELEGATE child_status=completed stop_reason=completed" in text
    assert "PARENT structured_result_checked=true; continued=true" in text
    assert "RECEIPT actual_content_verified=true" in text
    assert "TRACE started=1 completed=1; histories omitted" in text
    assert "PARENT STOP completed (model_rounds=3, tool_calls=2)" in text
    assert "TOTAL model_calls=5 tool_calls=3 depth=1" in text
