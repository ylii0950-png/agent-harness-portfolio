"""Optional Docker execution body. No host-command fallback, image pull or host data import.

Docker daemon/CLI and explicitly provisioned immutable image are trusted runtime
prerequisites. Configuration checks are not a claim of verified kernel isolation.
"""

import json
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast


class SandboxFault(ValueError):
    def __init__(self, code: str, *, fatal: bool = False) -> None:
        self.code, self.fatal = code, fatal
        super().__init__(code)


@dataclass(frozen=True)
class DockerConfig:
    image: str = ""
    enabled: bool = False
    binary: str | None = None
    socket: str = "/var/run/docker.sock"
    cpus: float = 1.0
    memory_mb: int = 128
    pids: int = 64
    timeout_seconds: float = 5.0
    output_bytes: int = 16384
    inputs: tuple[tuple[str, str], ...] = (("input.txt", "SYNTHETIC INPUT\n"),)

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or (self.enabled and not re.fullmatch(r"sha256:[a-f0-9]{64}", self.image))
            or (self.binary is not None and not Path(self.binary).is_absolute())
            or not Path(self.socket).is_absolute()
            or not self.socket.endswith(".sock")
            or any(p.casefold() == ".git" for p in Path(self.socket).parts)
            or type(self.cpus) not in {int, float}
            or not math.isfinite(self.cpus)
            or not 0.1 <= self.cpus <= 2
            or type(self.memory_mb) is not int
            or not 32 <= self.memory_mb <= 512
            or type(self.pids) is not int
            or not 8 <= self.pids <= 128
            or type(self.timeout_seconds) not in {int, float}
            or not math.isfinite(self.timeout_seconds)
            or not 0.1 <= self.timeout_seconds <= 15
            or type(self.output_bytes) is not int
            or not 128 <= self.output_bytes <= 32768
            or type(self.inputs) is not tuple
            or len(self.inputs) > 16
            or len(dict(self.inputs)) != len(self.inputs)
            or any(
                type(n) is not str
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", n)
                or type(t) is not str
                or len(t.encode()) > 16384
                for n, t in self.inputs
            )
            or sum(len(t.encode()) for _, t in self.inputs) > 65536
        ):
            raise ValueError("invalid_docker_sandbox_configuration")


@dataclass(frozen=True)
class Capture:
    code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_limited: bool = False


def capture(argv: list[str], *, env: dict[str, str], timeout: float, limit: int) -> Capture:
    """Bound stdout+stderr while running only the trusted Docker CLI, never a shell."""
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except OSError:
        raise SandboxFault("docker_unavailable") from None
    assert process.stdout is not None and process.stderr is not None
    streams = {process.stdout: "out", process.stderr: "err"}
    chunks: dict[str, bytearray] = {"out": bytearray(), "err": bytearray()}
    deadline = time.monotonic() + timeout
    timed_out = limited = False
    with selectors.DefaultSelector() as selector:
        for stream in streams:
            selector.register(stream, selectors.EVENT_READ, streams[stream])
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                for event, _ in selector.select(min(remaining, 0.1)):
                    block = os.read(event.fd, 4096)
                    if not block:
                        selector.unregister(event.fd)
                        continue
                    room = limit - sum(len(x) for x in chunks.values())
                    chunks[cast(str, event.data)].extend(block[:room])
                    if len(block) > room:
                        limited = True
                        break
                if limited:
                    break
            remaining = max(0, deadline - time.monotonic())
            if not timed_out and not limited:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.stdout.close()
            process.stderr.close()
            process.wait(timeout=2)
    return Capture(
        process.returncode,
        chunks["out"].decode(errors="replace"),
        chunks["err"].decode(errors="replace"),
        timed_out,
        limited,
    )


@dataclass(frozen=True)
class SandboxResult:
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    started_verified: bool
    daemon_configuration_verified: bool
    cleanup_acknowledged: bool
    container_id: str
    runtime_isolation_claim: str = "COMMAND_EXIT_IS_NOT_ISOLATION_VERIFICATION"


class DockerSandbox:
    def __init__(self, config: DockerConfig) -> None:
        self.config = config
        self._unhealthy = False
        self.last_attempt: dict[str, object] = {}

    def _binary(self) -> str:
        value = self.config.binary or shutil.which("docker")
        if value is None or not Path(value).is_file() or not os.access(value, os.X_OK):
            raise SandboxFault("docker_unavailable")
        return value

    def _control(
        self, arguments: list[str], env: dict[str, str], *, timeout: float = 3, limit: int = 32768
    ) -> Capture:
        return capture(
            [self._binary(), "--host", "unix://" + self.config.socket, *arguments],
            env=env,
            timeout=timeout,
            limit=limit,
        )

    @staticmethod
    def _environment(root: Path) -> dict[str, str]:
        return {
            "PATH": os.defpath,
            "HOME": str(root),
            "DOCKER_CONFIG": str(root),
            "LANG": "C.UTF-8",
        }

    def available(self) -> bool:
        try:
            self._binary()
            with TemporaryDirectory(prefix="harness-docker-check-") as folder:
                result = self._control(
                    ["version", "--format", "{{.Server.Os}}"], self._environment(Path(folder))
                )
                return (
                    result.code == 0
                    and result.stdout.strip() == "linux"
                    and not result.timed_out
                    and not result.output_limited
                )
        except SandboxFault:
            return False

    def create_arguments(
        self, stage: Path, name: str, token: str, argv: tuple[str, ...]
    ) -> list[str]:
        c = self.config
        return [
            "container",
            "create",
            "--pull=never",
            "--name",
            name,
            "--label",
            "agent-harness.owner=" + token,
            "--privileged=false",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--user=65534:65534",
            "--workdir=/workspace",
            "--ipc=private",
            "--cgroupns=private",
            "--restart=no",
            "--log-driver=none",
            "--cpus",
            str(c.cpus),
            "--memory",
            str(c.memory_mb * 1048576),
            "--memory-swap",
            str(c.memory_mb * 1048576),
            "--pids-limit",
            str(c.pids),
            "--shm-size=16m",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
            "--mount",
            f"type=bind,source={stage},target=/workspace,readonly",
            "--env",
            "HOME=/tmp",
            "--entrypoint",
            argv[0],
            c.image,
            *argv[1:],
        ]

    def _object(self, result: Capture) -> dict[str, Any]:
        if result.code != 0 or result.timed_out or result.output_limited:
            raise SandboxFault("sandbox_daemon_error")
        try:
            value = json.loads(result.stdout)
            if type(value) is not dict:
                raise ValueError
            return value
        except (ValueError, TypeError):
            raise SandboxFault("sandbox_daemon_protocol_error") from None

    def _inspect(self, target: str, env: dict[str, str]) -> dict[str, Any]:
        return self._object(
            self._control(["container", "inspect", "--format", "{{json .}}", target], env)
        )

    def _verify(self, obj: dict[str, Any], stage: Path, token: str, argv: tuple[str, ...]) -> None:
        h, c = obj.get("HostConfig", {}), obj.get("Config", {})
        mounts = obj.get("Mounts", [])
        if (
            type(h) is not dict
            or type(c) is not dict
            or type(mounts) is not list
            or any(type(m) is not dict for m in mounts)
        ):
            raise SandboxFault("sandbox_constraints_mismatch")
        binds = [m for m in mounts if m.get("Type") == "bind"]
        others = [m for m in mounts if m.get("Type") != "bind"]
        valid = (
            c.get("Labels", {}).get("agent-harness.owner") == token
            and c.get("User") == "65534:65534"
            and c.get("WorkingDir") == "/workspace"
            and c.get("Entrypoint") == [argv[0]]
            and (c.get("Cmd") or []) == list(argv[1:])
            and h.get("Privileged") is False
            and h.get("ReadonlyRootfs") is True
            and h.get("NetworkMode") == "none"
            and not h.get("CapAdd")
            and h.get("CapDrop") == ["ALL"]
            and h.get("PidMode", "") == ""
            and h.get("IpcMode") == "private"
            and h.get("CgroupnsMode") == "private"
            and h.get("Memory") == self.config.memory_mb * 1048576
            and h.get("MemorySwap") == self.config.memory_mb * 1048576
            and h.get("NanoCpus") == int(self.config.cpus * 1e9)
            and h.get("PidsLimit") == self.config.pids
            and len(h.get("SecurityOpt", [])) == 1
            and h["SecurityOpt"][0]
            in {"no-new-privileges", "no-new-privileges=true", "no-new-privileges:true"}
            and not h.get("Devices")
            and not h.get("DeviceRequests")
            and not h.get("PortBindings")
            and h.get("LogConfig", {}).get("Type") == "none"
            and h.get("RestartPolicy", {}).get("Name") == "no"
            and h.get("ShmSize") == 16777216
            and h.get("Tmpfs") == {"/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777"}
            and not h.get("Binds")
            and len(binds) == 1
            and all(
                m.get("Type") == "tmpfs" and m.get("Destination") == "/tmp" and not m.get("Source")
                for m in others
            )
            and len(others) <= 1
            and binds[0].get("Type") == "bind"
            and binds[0].get("Source") == str(stage)
            and binds[0].get("Destination") == "/workspace"
            and binds[0].get("RW") is False
        )
        if not valid:
            raise SandboxFault("sandbox_constraints_mismatch")

    def _cleanup(self, target: str, token: str, env: dict[str, str]) -> bool:
        try:
            obj = self._inspect(target, env)
            identifier = obj.get("Id", "")
            if (
                not re.fullmatch(r"[a-f0-9]{64}", identifier)
                or obj.get("Config", {}).get("Labels", {}).get("agent-harness.owner") != token
            ):
                return False
            result = self._control(["container", "rm", "--force", identifier], env)
            return (
                result.code == 0
                and not result.timed_out
                and not result.output_limited
                and result.stdout.strip() == identifier
            )
        except (SandboxFault, ValueError, TypeError, AttributeError, KeyError):
            return False

    def _execute(self, argv: tuple[str, ...]) -> SandboxResult:
        if self.config.enabled is not True:
            raise SandboxFault("sandbox_disabled")
        if self._unhealthy:
            raise SandboxFault("sandbox_cleanup_unverified", fatal=True)
        if (
            type(argv) is not tuple
            or not 1 <= len(argv) <= 16
            or any(type(a) is not str or not a or len(a) > 8192 or "\x00" in a for a in argv)
            or argv[0].startswith("-")
            or sum(len(a.encode()) for a in argv) > 12288
        ):
            raise SandboxFault("sandbox_arguments_invalid")
        self._binary()  # Never execute argv on the host if Docker is missing.
        token = uuid.uuid4().hex
        name = "agent-harness-" + token
        target = name
        attempted = False
        cleanup = True
        with TemporaryDirectory(prefix="harness-docker-") as folder:
            root = Path(folder).resolve()
            stage = root / "workspace"
            stage.mkdir(mode=0o755)
            if "," in str(stage):
                raise SandboxFault("sandbox_workspace_invalid")
            for filename, text in self.config.inputs:
                item = stage / filename
                item.write_text(text)
                item.chmod(0o444)
            env = self._environment(root)
            self.last_attempt = {
                "container_name": name,
                "stage": "availability",
                "cleanup_acknowledged": False,
            }
            try:
                version = self._control(["version", "--format", "{{.Server.Os}}"], env)
                if (
                    version.code != 0
                    or version.stdout.strip() != "linux"
                    or version.timed_out
                    or version.output_limited
                ):
                    raise SandboxFault("docker_unavailable")
                image = self._object(
                    self._control(
                        ["image", "inspect", "--format", "{{json .}}", self.config.image], env
                    )
                )
                if (
                    image.get("Id") != self.config.image
                    or image.get("Os") != "linux"
                    or image.get("Config", {}).get("Volumes")
                ):
                    raise SandboxFault("sandbox_image_rejected")
                attempted = True
                created = self._control(self.create_arguments(stage, name, token, argv), env)
                if (
                    created.code != 0
                    or created.timed_out
                    or created.output_limited
                    or not re.fullmatch(r"[a-f0-9]{64}", created.stdout.strip())
                ):
                    raise SandboxFault("sandbox_create_failed")
                target = created.stdout.strip()
                self.last_attempt.update(container_id=target, stage="constraints")
                before = self._inspect(target, env)
                self._verify(before, stage, token, argv)
                if before.get("State", {}).get("Running") is not False:
                    raise SandboxFault("sandbox_state_unverified")
                self.last_attempt["stage"] = "execution"
                execution = self._control(
                    ["container", "start", "--attach", target],
                    env,
                    timeout=self.config.timeout_seconds,
                    limit=self.config.output_bytes,
                )
                after = self._inspect(target, env)
                self._verify(after, stage, token, argv)
                state = after.get("State", {})
                started = bool(state.get("StartedAt")) and not str(state["StartedAt"]).startswith(
                    "0001-"
                )
                if execution.timed_out or execution.output_limited:
                    status = "timed_out" if execution.timed_out else "output_limited"
                    exit_code = None
                else:
                    exit_code = state.get("ExitCode")
                    if (
                        not started
                        or state.get("Running") is not False
                        or type(exit_code) is not int
                        or (execution.code != 0 and exit_code == 0)
                    ):
                        raise SandboxFault("sandbox_state_unverified")
                    status = "completed" if exit_code == 0 else "command_failed"
                result = SandboxResult(
                    status,
                    exit_code,
                    execution.stdout,
                    execution.stderr,
                    started,
                    True,
                    False,
                    target,
                )
            finally:
                if attempted:
                    cleanup = self._cleanup(target, token, env)
                self.last_attempt["cleanup_acknowledged"] = cleanup
                if not cleanup:
                    self._unhealthy = True
                    raise SandboxFault("sandbox_cleanup_unverified", fatal=True)
            from dataclasses import replace

            return replace(result, cleanup_acknowledged=True)
