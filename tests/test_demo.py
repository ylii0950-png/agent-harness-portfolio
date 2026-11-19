import subprocess
import sys


def test_installed_mock_demo_runs_our_runtime_and_is_reproducible() -> None:
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-I", "-m", "agent_harness_lab.demo"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        outputs.append(result.stdout)
        assert result.stderr == ""
    assert outputs[0] == outputs[1]
    lines = outputs[0].splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("CALL call_1 write_file ")
    assert lines[1].startswith(
        'RESULT call_1 {"content": "File written.", "error": null, "ok": true}'
    )
    assert lines[2].startswith("CALL call_2 read_file ")
    assert "Hello from the Phase 1 mock agent." in lines[3]
    assert lines[4] == "FINAL Created and read note.txt successfully."
    assert lines[5] == "STOP completed (model_rounds=3, tool_calls=2)"
