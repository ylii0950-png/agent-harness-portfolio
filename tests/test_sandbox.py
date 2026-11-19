"""Synthetic daemon contracts; actual isolation test requires opt-in local image."""

import copy
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_harness_lab.contracts import ToolCall
from agent_harness_lab.extension_tools import ExtensionRegistry
from agent_harness_lab.permissions import Decision, Permission
from agent_harness_lab.providers import MockProvider, ModelResponse
from agent_harness_lab.runtime import Agent, AgentResult, RunLimits
from agent_harness_lab.sandbox import Capture, DockerConfig, DockerSandbox, capture
from agent_harness_lab.sandbox_tools import register_sandbox_tool

IMAGE = "sha256:" + "a" * 64
CID = "b" * 64


class SyntheticDocker(DockerSandbox):
    """Only CLI contract fixtures. Never claim these ran a real container."""

    def __init__(
        self, *, mode: str = "success", tamper: str = "", config: DockerConfig | None = None
    ) -> None:
        super().__init__(config or DockerConfig(image=IMAGE, enabled=True))
        self.mode, self.tamper = mode, tamper
        self.operations: list[list[str]] = []
        self.object: dict[str, Any] = {}
        self.environments: list[dict[str, str]] = []

    def _binary(self) -> str:
        return "/synthetic/docker"

    def _control(
        self, arguments: list[str], env: dict[str, str], *, timeout: float = 3, limit: int = 32768
    ) -> Capture:
        self.operations.append(arguments)
        self.environments.append(env)
        action = arguments[:2]
        if arguments[0] == "version":
            return Capture(0, "linux\n", "")
        if action == ["image", "inspect"]:
            return Capture(
                0,
                json.dumps(
                    {
                        "Id": IMAGE,
                        "Os": "linux",
                        "Config": {
                            "Volumes": {"/foreign": {}} if self.tamper == "image_volume" else None
                        },
                    }
                ),
                "",
            )
        if action == ["container", "create"]:
            token = arguments[arguments.index("--label") + 1].split("=", 1)[1]
            source = (
                arguments[arguments.index("--mount") + 1].split("source=", 1)[1].split(",", 1)[0]
            )
            assert {p.name for p in Path(source).iterdir()} == {"input.txt"}
            self.object = {
                "Id": CID,
                "Config": {
                    "Labels": {"agent-harness.owner": token},
                    "User": "65534:65534",
                    "WorkingDir": "/workspace",
                    "Entrypoint": [arguments[arguments.index("--entrypoint") + 1]],
                    "Cmd": arguments[arguments.index(IMAGE) + 1 :],
                },
                "HostConfig": {
                    "Privileged": False,
                    "ReadonlyRootfs": True,
                    "NetworkMode": "none",
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges=true"],
                    "IpcMode": "private",
                    "CgroupnsMode": "private",
                    "Memory": self.config.memory_mb * 1048576,
                    "MemorySwap": self.config.memory_mb * 1048576,
                    "NanoCpus": int(self.config.cpus * 1e9),
                    "PidsLimit": self.config.pids,
                    "LogConfig": {"Type": "none"},
                    "RestartPolicy": {"Name": "no"},
                    "ShmSize": 16777216,
                    "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777"},
                },
                "Mounts": [
                    {"Type": "bind", "Source": source, "Destination": "/workspace", "RW": False}
                ],
                "State": {"Running": False, "StartedAt": "0001-01-01T00:00:00Z", "ExitCode": 0},
            }
            if self.tamper in {"Privileged", "NetworkMode", "Memory", "CapDrop"}:
                self.object["HostConfig"][self.tamper] = {
                    "Privileged": True,
                    "NetworkMode": "host",
                    "Memory": 0,
                    "CapDrop": [],
                }[self.tamper]
            if self.tamper == "root":
                self.object["Config"]["User"] = "0"
            if self.tamper == "socket_mount":
                self.object["Mounts"][0]["Source"] = "/var/run/docker.sock"
            if self.tamper == "owner":
                self.object["Config"]["Labels"]["agent-harness.owner"] = "foreign"
            return Capture(0, CID + "\n", "")
        if action == ["container", "inspect"]:
            return Capture(0, json.dumps(copy.deepcopy(self.object)), "")
        if action == ["container", "start"]:
            self.object["State"].update(
                StartedAt="2026-10-27T00:00:00Z",
                Running=self.mode in {"timeout", "output"},
                ExitCode=7 if self.mode == "command_failure" else 0,
            )
            return Capture(
                0,
                "SYNTHETIC",
                "",
                timed_out=self.mode == "timeout",
                output_limited=self.mode == "output",
            )
        if action == ["container", "rm"]:
            if self.mode == "cleanup_failure":
                return Capture(1, "", "")
            return Capture(0, CID + "\n", "")
        raise AssertionError("unexpected synthetic Docker operation")


def agent(
    backend: DockerSandbox, *, approve: object = True, tools: int = 2, argv: object = None
) -> AgentResult:
    from typing import cast

    registry = ExtensionRegistry(
        permissions=(Permission.READ, Permission.WRITE),
        rules={"sandbox_execute": Decision.ASK},
        approval_handler=lambda _: cast(bool, approve),
    )
    register_sandbox_tool(registry, backend)
    arguments = {
        "argv_json": json.dumps(argv if argv is not None else ["python3", "-c", "print('PUBLIC')"])
    }
    return Agent(
        MockProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("sandbox-id", "sandbox_execute", json.dumps(arguments)),)
                ),
                "DONE",
            ]
        ),
        registry,
        limits=RunLimits(3, tools),
    ).run("Execute fictional command only inside restricted Docker")


def test_synthetic_daemon_success_ownership_receipt_and_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://untrusted.invalid:2375")
    monkeypatch.setenv("HOST_SECRET", "must-not-propagate")
    backend = SyntheticDocker()
    result = agent(backend)
    assert result.tool_results[0].ok
    value = json.loads(result.tool_results[0].content)
    assert value["call_id"] == "sandbox-id" and value["trust"] == "untrusted_process_output"
    assert value["result"]["status"] == "completed" and value["result"]["cleanup_acknowledged"]
    created = next(a for a in backend.operations if a[:2] == ["container", "create"])
    assert "--pull=never" in created and "--privileged=false" in created
    assert "--network=none" in created and "--cap-drop=ALL" in created
    assert created.count("--mount") == 1 and not any(a.startswith("--volume") for a in created)
    assert all(set(e) == {"PATH", "HOME", "DOCKER_CONFIG", "LANG"} for e in backend.environments)
    assert backend.operations[-1] == ["container", "rm", "--force", CID]


@pytest.mark.parametrize(
    "tamper", ["Privileged", "NetworkMode", "Memory", "CapDrop", "root", "socket_mount"]
)
def test_changed_daemon_constraints_prevent_start_and_cleanup_owned_container(tamper: str) -> None:
    backend = SyntheticDocker(tamper=tamper)
    result = agent(backend)
    assert not result.tool_results[0].ok
    assert not any(a[:2] == ["container", "start"] for a in backend.operations)
    assert backend.operations[-1] == ["container", "rm", "--force", CID]


def test_foreign_owner_never_deleted_and_unverified_cleanup_poisoned() -> None:
    backend = SyntheticDocker(tamper="owner")
    result = agent(backend)
    assert result.stop_reason.value == "tool_error" and backend._unhealthy
    assert not any(a[:2] == ["container", "rm"] for a in backend.operations)
    count = len(backend.operations)
    second = agent(backend)
    assert not second.tool_results[0].ok and len(backend.operations) == count


@pytest.mark.parametrize(
    ("mode", "status"),
    [("timeout", "timed_out"), ("output", "output_limited"), ("command_failure", "command_failed")],
)
def test_command_failures_keep_true_status_and_cleanup(mode: str, status: str) -> None:
    backend = SyntheticDocker(mode=mode)
    result = agent(backend)
    row = json.loads(result.tool_results[0].content)["result"]
    assert row["status"] == status and row["cleanup_acknowledged"]
    assert row["exit_code"] == (7 if mode == "command_failure" else None)


def test_cleanup_failure_is_fatal_and_image_volumes_rejected() -> None:
    backend = SyntheticDocker(mode="cleanup_failure")
    result = agent(backend)
    assert result.stop_reason.value == "tool_error" and backend._unhealthy
    assert result.tool_results[0].error is not None
    assert result.tool_results[0].error.code == "sandbox_cleanup_unverified"
    image = SyntheticDocker(tamper="image_volume")
    assert not agent(image).tool_results[0].ok
    assert not any(a[:2] == ["container", "create"] for a in image.operations)


def test_refused_invalid_approval_and_budget_precede_any_docker_control() -> None:
    for approval in (False, "yes", 1):
        backend = SyntheticDocker()
        assert not agent(backend, approve=approval).tool_results[0].ok and backend.operations == []
    backend = SyntheticDocker()
    assert (
        agent(backend, tools=0).stop_reason.value == "max_tool_calls" and backend.operations == []
    )
    with pytest.raises(ValueError):
        register_sandbox_tool(ExtensionRegistry(), backend)


def test_disabled_unavailable_and_invalid_argv_have_no_host_fallback(tmp_path: Path) -> None:
    missing = DockerSandbox(DockerConfig(image=IMAGE, enabled=True, binary="/nonexistent/docker"))
    result = agent(missing, argv=["touch", str(tmp_path / "must-not-exist")])
    assert not result.tool_results[0].ok and not (tmp_path / "must-not-exist").exists()
    assert (
        result.tool_results[0].error is not None
        and result.tool_results[0].error.code == "docker_unavailable"
    )
    assert not agent(DockerSandbox(DockerConfig())).tool_results[0].ok
    for argv in ({"privileged": True}, [], ["--privileged"], ["python3", 0]):
        backend = SyntheticDocker()
        assert not agent(backend, argv=argv).tool_results[0].ok and backend.operations == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image": "python:latest"},
        {"memory_mb": 0},
        {"pids": 0},
        {"cpus": float("inf")},
        {"inputs": (("../outside", "x"),)},
    ],
)
def test_invalid_config_fails_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(DockerConfig(image=IMAGE, enabled=True), **kwargs)


def test_cli_capture_bounds_output_and_time() -> None:
    env = {"PATH": os.defpath}
    output = capture([sys.executable, "-c", "print('X'*20000)"], env=env, timeout=2, limit=128)
    assert output.output_limited and len(output.stdout.encode()) <= 128
    start = time.monotonic()
    timeout = capture(
        [sys.executable, "-c", "import time; time.sleep(5)"], env=env, timeout=0.1, limit=128
    )
    assert timeout.timed_out and time.monotonic() - start < 3


@pytest.mark.docker
def test_actual_container_limits_when_explicit_local_runtime_and_image_available() -> None:
    image = os.environ.get("HARNESS_DOCKER_TEST_IMAGE")
    if not image:
        pytest.skip("UNAVAILABLE: no explicitly configured preloaded image; no automatic pull")
    backend = DockerSandbox(DockerConfig(image=image, enabled=True))
    if not backend.available():
        pytest.skip("UNAVAILABLE: local Linux Docker daemon")
    script = """import os,pathlib,json
p=pathlib.Path
lines=p('/proc/self/status').read_text().splitlines()
status=dict(line.split(':',1) for line in lines if ':' in line)
def denied(name):
 try:
  with open(name,'w') as f:f.write('SYNTHETIC')
  return False
 except OSError:return True
c=p('/sys/fs/cgroup')
if (c/'memory.max').exists():
 memory=int((c/'memory.max').read_text())
 pids=int((c/'pids.max').read_text())
 quota,period=(c/'cpu.max').read_text().split()
else:
 memory=int((c/'memory/memory.limit_in_bytes').read_text())
 pids=int((c/'pids/pids.max').read_text())
 quota=(c/'cpu/cpu.cfs_quota_us').read_text()
 period=(c/'cpu/cpu.cfs_period_us').read_text()
checks={
 'uid':os.getuid()==65534,
 'caps':int(status['CapEff'].strip(),16)==0,
 'nnp':status['NoNewPrivs'].strip()=='1',
 'root_ro':denied('/must-not-write'),
 'workspace_ro':denied('/workspace/input.txt'),
 'socket_absent':not p('/var/run/docker.sock').exists(),
 'interfaces':{x.name for x in p('/sys/class/net').iterdir()}=={'lo'},
 'memory':memory<=134217728,
 'pids':pids<=64,
 'cpu':0<int(quota)/int(period)<=1,
}
print(json.dumps(checks))
"""
    result = agent(backend, argv=["python3", "-c", script])
    assert result.tool_results[0].ok
    receipt = json.loads(result.tool_results[0].content)["result"]
    assert (
        receipt["status"] == "completed"
        and receipt["started_verified"]
        and receipt["cleanup_acknowledged"]
    )
    assert all(json.loads(receipt["stdout"]).values())
