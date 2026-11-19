import runpy

import pytest


def test_permission_demo_runs_own_agent_registry_and_checks_actual_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_module("agent_harness_lab.permission_demo", run_name="__main__")
    output = capsys.readouterr().out
    assert "APPROVAL_SOURCE scripted Mock decisions" in output
    assert "REQUEST write_file policy=deny authorized=false reason=capability_denied" in output
    assert "REQUEST read_file policy=allow authorized=true reason=allowed" in output
    assert "REQUEST write_file policy=ask authorized=true reason=approved" in output
    assert "REQUEST write_file policy=ask authorized=false reason=approval_denied" in output
    assert "DEFAULT_WRITE denied; file_created=false" in output
    assert "OVERWRITE rejected; approved_file_unchanged=true" in output
    assert "FINAL Approved content preserved; overwrite denied." in output
    assert output.endswith("STOP completed\n")
