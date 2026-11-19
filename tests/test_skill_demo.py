import runpy

import pytest


def test_skill_demo_runs_own_agent_and_verifies_actual_lazy_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_module("agent_harness_lab.skill_demo", run_name="__main__")
    text = capsys.readouterr().out
    assert "actual_body_reads=1" in text
    assert "REPEAT receipt status=already_loaded; body_omitted=true" in text
    assert "UNSELECTED unrelated loaded=false; body_absent_from_context=true" in text
    assert "RECEIPT actual_content=SKILL READY; verified=true" in text
    assert "STOP completed (model_rounds=5, tool_calls=4)" in text
