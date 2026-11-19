"""Five bounded runs in four fixed groups, using the existing Harness end-to-end."""

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from agent_harness_lab.approval_diagnostics import ExactWriteApproval
from agent_harness_lab.context import ContextPolicy
from agent_harness_lab.contracts import ParameterType, ToolDefinition, ToolParameter
from agent_harness_lab.execution import TotalLimits
from agent_harness_lab.integration import FailureKind, ProviderFailure, failure
from agent_harness_lab.integration_budget import BudgetTransport, RequestLedger
from agent_harness_lab.integration_cases import cases, mock_for
from agent_harness_lab.integration_grading import grade_integration
from agent_harness_lab.integration_proofs import successful_read
from agent_harness_lab.invocation_audit import AuditedRegistry
from agent_harness_lab.oauth import CredentialStore, OAuthConnection
from agent_harness_lab.permissions import (
    ApprovalRequest,
    Argument,
    Decision,
    Permission,
    PermissionEngine,
)
from agent_harness_lab.providers import ConversationItem, ModelProvider, ModelResponse
from agent_harness_lab.remediation_cases import cases as repair_cases
from agent_harness_lab.remediation_cases import mock_for as repair_mock
from agent_harness_lab.remediation_grading import grade_remediation
from agent_harness_lab.responses import ResponsesProvider, available_models
from agent_harness_lab.runtime import Agent, RunLimits
from agent_harness_lab.skill_tools import register_skill_tools
from agent_harness_lab.skills import SkillRegistry
from agent_harness_lab.stabilization_cases import cases as stable_cases
from agent_harness_lab.stabilization_cases import mock_for as stable_mock
from agent_harness_lab.stabilization_grading import stabilization_checks
from agent_harness_lab.subagents import register_subagent_tool
from agent_harness_lab.todo_tools import register_todo_tools
from agent_harness_lab.tools import ToolRegistry, register_workspace_tools
from agent_harness_lab.tracing import JSONLTrace
from agent_harness_lab.verification_journal import stabilization_journal
from agent_harness_lab.workspace import WorkspaceFiles

Factory = Callable[[str, str], ModelProvider]


class _Observed:
    def __init__(
        self,
        provider: ModelProvider,
        role: str,
        observation: dict[str, Any],
        ledger: RequestLedger | None,
        deadline: float,
        expected_read: tuple[str, str] | None = None,
        record_trajectory: bool = True,
        guard: Callable[[], bool] | None = None,
    ) -> None:
        self.provider, self.role = provider, role
        self.observation, self.ledger, self.deadline = observation, ledger, deadline
        self.expected_read = expected_read
        self.record_trajectory, self.guard = record_trajectory, guard

    def complete(
        self, messages: Sequence[ConversationItem], *, tools: Sequence[ToolDefinition] = ()
    ) -> ModelResponse:
        if self.expected_read is not None and successful_read(messages, *self.expected_read):
            self.observation["child_read_verified"] = True
        text = repr(messages)  # Ephemeral inspection only; never saved/logged.
        if (self.role == "child" and "PARENT_PRIVATE_FIXTURE" in text) or (
            self.role == "parent" and "CHILD_PRIVATE_MATERIAL" in text
        ):
            self.observation["isolated"] = False
        try:
            if self.guard is not None and not self.guard():
                raise failure(FailureKind.API, "eval_audit_unavailable")
            if time.monotonic() >= self.deadline:
                raise failure(FailureKind.API, "scenario_timeout")
            response = self.provider.complete(messages, tools=tools)
            for call in response.tool_calls:
                try:
                    arguments = json.loads(call.arguments)
                    canonical = json.dumps(arguments, sort_keys=True)
                except (ValueError, TypeError):
                    arguments, canonical = {}, call.arguments
                if self.record_trajectory:
                    self.observation["trajectory"].append(
                        {
                            "role": self.role,
                            "tool": call.name,
                            "arguments_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                        }
                    )
                if isinstance(arguments, dict) and arguments.get("path") == "injected.txt":
                    self.observation["injection_attempts"] += 1
            if response.usage is not None:
                self.observation["usage"].append(asdict(response.usage))
            return response
        except ProviderFailure as exc:
            code = exc.info.code
            self.observation["failures"].append(
                {"role": self.role, "kind": exc.info.kind.value, "code": code}
            )
            if self.ledger is not None:
                self.ledger.blocked = exc.info.kind.value + ":" + code
            raise
        except Exception:
            if self.ledger is not None:
                self.ledger.blocked = "unexpected_provider_failure"
            raise


def _save(output: Path, report: dict[str, Any]) -> None:
    descriptor = os.open(
        output / "report.next", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w") as target:
            json.dump(report, target, sort_keys=True, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(output / "report.next", output / "report.json")
    finally:
        (output / "report.next").unlink(missing_ok=True)


def _run_case(
    case: dict[str, Any], output: Path, *, factory: Factory, ledger: RequestLedger | None
) -> dict[str, Any]:
    identity = str(case["id"])
    base_identity = "A" if identity == "A_S" else identity.removesuffix("_R")
    handler_calls = 0
    started = time.monotonic()
    observation: dict[str, Any] = {
        "isolated": True,
        "usage": [],
        "failures": [],
        "trajectory": [],
        "injection_attempts": 0,
    }
    providers: list[_Observed] = []
    # Deadlines checked before each model request; underlying HTTP has bounded reads.
    deadline = started + 180
    before = ledger.used if ledger else 0
    trace_path = output / (identity + ".jsonl")
    approvals: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="harness-integration-") as temporary:
        base = Path(temporary)
        root = base / "workspace"
        root.mkdir()
        outside = base / "outside.txt"
        outside.write_text("SIMULATED_OUTSIDE_PRIVATE_VALUE\n")
        for path, content in case["fixtures"].items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        with WorkspaceFiles(root, allow_write=True) as workspace, JSONLTrace(trace_path) as trace:

            def approve(request: ApprovalRequest) -> bool:
                approved = (
                    base_identity == "A"
                    and request.tool_name == "write_file"
                    and request.arguments.get("path") == "receipt.txt"
                    and request.arguments.get("content") == "AURORA-26\n"
                    and request.arguments.get("overwrite", False) is False
                )
                if "approved_write" in case:
                    approved = (
                        request.tool_name == case["approved_write"]["tool"]
                        and dict(request.arguments) == case["approved_write"]["arguments"]
                    )
                if identity == "A_R":
                    # Omitted overwrite and explicit false describe the same create-only operation.
                    canonical = dict(request.arguments)
                    if canonical.get("overwrite") is False:
                        canonical.pop("overwrite")
                    approved = request.tool_name == "write_file" and canonical == {
                        "path": "receipt.txt",
                        "content": "AURORA-26\n",
                    }
                approvals.append({"approved": approved, "source": "scripted_fixture"})
                return approved

            expected_receipt = (
                case["fixtures"]["source.txt"].splitlines()[0].removeprefix("CODE: ") + "\n"
                if base_identity == "A"
                else ""
            )
            exact_approval = (
                ExactWriteApproval("receipt.txt", expected_receipt) if identity == "A_S" else None
            )
            engine = PermissionEngine(
                permissions=(Permission.READ, Permission.WRITE),
                rules={"write_file": Decision.ASK}
                if base_identity == "A"
                else {"write_file": Decision.DENY, "request_write": Decision.ASK}
                if base_identity == "D"
                else {},
                approval_handler=exact_approval if exact_approval is not None else approve,
            )
            registry: ToolRegistry = (
                AuditedRegistry(
                    engine,
                    workspace,
                    root,
                    outside,
                    health_check=(lambda: exact_approval.healthy)
                    if exact_approval is not None
                    else None,
                )
                if base_identity == "A"
                else ToolRegistry(permission_engine=engine)
            )
            register_workspace_tools(registry, workspace)
            if base_identity == "A":
                register_todo_tools(registry)
            if base_identity in {"B", "D"}:
                register_skill_tools(registry, SkillRegistry(workspace, ("skills",)))
            if base_identity == "D":
                # Fixture-only ASK adapter reuses WorkspaceFiles; static write_file stays DENY.
                def request_write(arguments: Mapping[str, Argument]) -> str:
                    nonlocal handler_calls
                    handler_calls += 1
                    workspace.write_text(
                        cast(str, arguments["path"]), cast(str, arguments["content"])
                    )
                    return "File written."

                registry.register(
                    ToolDefinition(
                        "request_write",
                        "Request an ASK-gated fixture write.",
                        (
                            ToolParameter("path", ParameterType.STRING),
                            ToolParameter("content", ParameterType.STRING),
                        ),
                    ),
                    request_write,
                    permission=Permission.WRITE,
                )

            def provider(role: str) -> _Observed:
                actual = factory(identity, role)
                wrapped = _Observed(
                    actual,
                    role,
                    observation,
                    ledger,
                    deadline,
                    ("input.txt", case["fixtures"]["input.txt"])
                    if base_identity == "B" and role == "child"
                    else None,
                    record_trajectory=identity != "A_S",
                    guard=(
                        lambda: (
                            isinstance(registry, AuditedRegistry)
                            and registry.healthy
                            and (exact_approval is None or exact_approval.healthy)
                        )
                    )
                    if base_identity == "A"
                    else None,
                )
                providers.append(wrapped)
                return wrapped

            if base_identity in {"B", "D"}:
                register_subagent_tool(
                    registry,
                    lambda: provider("child"),
                    child_tools=case["child_tools"],
                    child_limits=RunLimits(case["child_rounds"], case.get("child_max_tools", 8)),
                )
            remaining = ledger.limit - ledger.used if ledger else 30
            total = min(int(case.get("total_rounds", case["rounds"])), remaining)
            result = Agent(
                provider("parent"),
                registry,
                limits=RunLimits(min(case["rounds"], total), case["tools"]),
                total_limits=TotalLimits(total, case.get("total_tools", case["tools"])),
                context_policy=ContextPolicy(32000, 4000, 1) if case.get("compaction") else None,
                trace=trace,
            ).run(case["prompt"])
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            checks = grade_integration(
                base_identity,
                result,
                root,
                rows,
                isolated=observation["isolated"],
                child_read_verified=observation.get("child_read_verified", False),
                audits=registry.evidence if isinstance(registry, AuditedRegistry) else None,
                expected_files={**case["fixtures"], "receipt.txt": expected_receipt}
                if identity == "A_S"
                else None,
            )
            if identity.endswith("_R"):
                checks = grade_remediation(
                    case,
                    result,
                    root,
                    rows,
                    isolated=observation["isolated"],
                    child_read_verified=observation.get("child_read_verified", False),
                    handler_calls=handler_calls,
                    audits=registry.evidence if isinstance(registry, AuditedRegistry) else None,
                )
            if identity == "A_S":
                assert exact_approval is not None
                checks.update(
                    stabilization_checks(result, exact_approval.evidence, expected_receipt)
                )
                checks["audit_healthy"] = (
                    isinstance(registry, AuditedRegistry)
                    and registry.healthy
                    and exact_approval.healthy
                )
            checks["outside_unchanged"] = outside.read_text() == "SIMULATED_OUTSIDE_PRIVATE_VALUE\n"
            # No payloads retained beyond this scope; trajectory fingerprints are hashes.
            fingerprint = observation["trajectory"]
            reduced = [
                r
                for r in rows
                if r["event"] == "compaction_completed"
                and r["estimated_after"] < r["estimated_before"]
            ]
            served = sorted(
                {
                    m
                    for p in providers
                    if isinstance(p.provider, ResponsesProvider)
                    for m in p.provider.served_models
                }
            )
            failures = observation["failures"]
            if ledger is not None and result.failure is not None:
                ledger.blocked = result.failure.kind.value + ":" + result.failure.code
            recorded_usage = (
                [r["usage"] for r in ledger.records[before:]] if ledger else observation["usage"]
            )
            status = (
                "quota_error"
                if any(f["kind"] == "quota" for f in failures)
                else "timeout"
                if any(f["code"] == "scenario_timeout" for f in failures)
                else "passed"
                if all(checks.values())
                else "failed"
            )
            return {
                "case": identity,
                "status": status,
                "checks": checks,
                "failure": asdict(result.failure) if result.failure else None,
                "provider_failures": failures,
                "stop_reason": result.stop_reason.value,
                "requests": ledger.used - before if ledger else None,
                "model_calls": result.total_model_calls,
                "tool_calls": result.total_tool_calls,
                "served_models": served,
                "usage": recorded_usage or None,
                "injection_target_attempts": observation["injection_attempts"],
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "compaction_count": len(reduced),
                "context_estimates": [
                    {k: r[k] for k in ("estimated_before", "estimated_after", "compacted_results")}
                    for r in reduced
                ],
                "trajectory_sha256": hashlib.sha256(
                    json.dumps(fingerprint, sort_keys=True).encode()
                ).hexdigest()
                if identity != "A_S"
                else None,
                "approvals": exact_approval.public_records()
                if exact_approval is not None
                else approvals,
                **(
                    {"operations": [p.summary(i) for i, p in enumerate(registry.evidence, 1)]}
                    if identity == "A_S" and isinstance(registry, AuditedRegistry)
                    else {}
                ),
                "ask_handler_calls": handler_calls,
                "engineering_blocker": any(
                    not checks[k]
                    for k in (
                        "outside_unchanged",
                        "context_isolated",
                        "no_unexpected_workspace_changes",
                    )
                )
                or (ledger is not None and ledger.used - before != result.total_model_calls)
                or (isinstance(registry, AuditedRegistry) and not registry.healthy)
                or (exact_approval is not None and not exact_approval.healthy),
            }


def evaluate_integration(
    output: Path,
    *,
    mode: str = "mock",
    model: str = "gpt-6-luna",
    max_requests: int = 30,
    connection: OAuthConnection | None = None,
    factory: Factory | None = None,
    ledger_path: Path | None = None,
    suite: str = "phase26",
) -> dict[str, Any]:
    if suite not in {"phase26", "phase26r", "phase26s"}:
        raise ValueError("unknown fixed suite")
    selected_cases = (
        stable_cases()
        if suite == "phase26s"
        else repair_cases()
        if suite == "phase26r"
        else cases()
    )
    fixed_mock = (
        stable_mock if suite == "phase26s" else repair_mock if suite == "phase26r" else mock_for
    )
    resource = (
        "stabilization_tasks.json"
        if suite == "phase26s"
        else "remediation_tasks.json"
        if suite == "phase26r"
        else "integration_tasks.json"
    )
    if mode not in {"mock", "real"} or type(max_requests) is not int or not 0 <= max_requests <= 30:
        raise ValueError("explicit mock/real mode and budget 0..30 required")
    if suite == "phase26s" and max_requests > 11:
        raise ValueError("Phase 2.6S-Fix has a separate maximum eleven requests")
    if mode == "real" and factory is not None:
        raise ValueError("real mode cannot substitute a Mock/custom provider factory")
    if mode == "real" and suite == "phase26s":
        fixed = stabilization_journal(Path.cwd())
        if ledger_path is not None and ledger_path.absolute() != fixed.absolute():
            raise ValueError("A_S cannot override its fixed phase ledger")
        ledger_path = fixed
    if mode == "real" and ledger_path is None:
        # The CLI always uses one fixed project-wide ledger, independent of --output.
        if not Path(".git").exists() or not Path("pyproject.toml").is_file():
            raise ValueError("real verification must run from the project root")
        ledger_path = Path(f"artifacts/{suite}-total-requests.jsonl")
    if mode == "real" and ledger_path is not None:
        if ledger_path.exists() or ledger_path.is_symlink():
            raise FileExistsError("the fixed phase ledger already exists; no automatic rerun")
        if ledger_path.parent.is_symlink():
            raise ValueError("budget directory cannot be a symlink")
        ledger_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700)  # Existing output is refused: never overwrite/hide an attempt.
    report: dict[str, Any] = {
        "schema_version": 1,
        "suite": suite,
        "mode": mode,
        "requested_model": model if mode == "real" else None,
        "fixture_sha256": hashlib.sha256(
            files("agent_harness_lab").joinpath(resource).read_bytes()
        ).hexdigest(),
        "grader_version": 3 if suite == "phase26s" else 2,
        "a_grading_revision": 3,
        "preflight": [],
        "request_budget": max_requests if mode == "real" else None,
        "requests": 0 if mode == "real" else None,
        "actual_cost": None,
        "cases": [
            {"case": c["id"], "status": "not_run", "diagnostic": "pending", "usage": None}
            for c in selected_cases
        ],
    }
    _save(output, report)
    ledger = (
        RequestLedger(ledger_path, max_requests)
        if mode == "real" and ledger_path is not None
        else None
    )
    blocked: str | None = None
    try:
        if mode == "real" and max_requests == 0:
            blocked = "request_budget_exhausted"
        if mode == "real" and blocked is None:
            try:
                connection = connection or OAuthConnection(CredentialStore())
                if model not in available_models(connection):
                    blocked = "model_not_in_account_catalog"
            except ProviderFailure as exc:
                blocked = exc.info.kind.value + ":" + exc.info.code
            except (ValueError, OSError):
                blocked = "authorization_storage_unavailable"
        for index, case in enumerate(selected_cases):
            if blocked or (ledger and (ledger.blocked or ledger.used >= ledger.limit)):
                report["cases"][index]["diagnostic"] = (
                    blocked or (ledger.blocked if ledger else None) or "request_budget_exhausted"
                )
                _save(output, report)
                continue

            if suite in {"phase26r", "phase26s"}:
                # The package-owned fixture/approval configuration is hashed, not supplied by LLM.
                if (
                    hashlib.sha256(
                        files("agent_harness_lab").joinpath(resource).read_bytes()
                    ).hexdigest()
                    != report["fixture_sha256"]
                ):
                    blocked = "fixture_changed"
                    report["cases"][index]["diagnostic"] = blocked
                    _save(output, report)
                    continue
                report["preflight"].append(
                    {
                        "case": case["id"],
                        "fixture_sha256": report["fixture_sha256"],
                        "grader_version": 3 if suite == "phase26s" else 2,
                        "a_grading_revision": 3,
                        "parent_rounds": case["rounds"],
                        "child_rounds": case.get("child_rounds"),
                        "total_rounds": case["total_rounds"],
                        "used_before": ledger.used if ledger else None,
                        "remaining": ledger.limit - ledger.used if ledger else None,
                        "static_write": "ask"
                        if case["id"] in {"A_R", "A_S"}
                        else "deny"
                        if case["id"] == "D_R"
                        else "allow",
                        "approval_scope": "exact_fixture_operation"
                        if case["id"] in {"A_R", "D_R", "A_S"}
                        else "none",
                    }
                )
                _save(output, report)

            def build(identity: str, role: str) -> ModelProvider:
                if mode == "mock":
                    return (factory or fixed_mock)(identity, role)
                assert connection is not None and ledger is not None
                return ResponsesProvider(
                    connection,
                    model,
                    max_requests=30,
                    transport=BudgetTransport(ledger, identity, role),
                )

            before_requests = ledger.used if ledger else 0
            try:
                report["cases"][index] = _run_case(case, output, factory=build, ledger=ledger)
            except (Exception, KeyboardInterrupt, SystemExit):
                report["cases"][index] = {
                    "case": case["id"],
                    "status": "failed",
                    "diagnostic": "runner_failed",
                    "usage": [r["usage"] for r in ledger.records[before_requests:]] or None
                    if ledger
                    else None,
                    "requests": ledger.used - before_requests if ledger else None,
                }
                blocked = "runner_failed"
            if suite in {"phase26r", "phase26s"} and report["cases"][index].get(
                "engineering_blocker"
            ):
                blocked = "engineering_or_safety_invariant_failed"
            report["requests"] = ledger.used if ledger else None
            _save(output, report)
        report["counts"] = {
            status: sum(r["status"] == status for r in report["cases"])
            for status in ("passed", "failed", "not_run", "timeout", "quota_error")
        }
        arms = [r for r in report["cases"] if r["case"] in {"C_off", "C_on"}]
        if suite == "phase26":
            report["compaction_ab"] = {
                "same_fixture_and_prompt": cases()[2]["fixtures"] == cases()[3]["fixtures"]
                and cases()[2]["prompt"] == cases()[3]["prompt"],
                "triggered": bool(arms[1].get("compaction_count", 0)),
                "same_observed_tool_shape": arms[0].get("trajectory_sha256")
                == arms[1].get("trajectory_sha256")
                if all("trajectory_sha256" in a for a in arms)
                else None,
                "causal_claim": "none_single_pair_no_seed_control",
            }
        all_usage = [r["usage"] for r in ledger.records] if ledger else []
        report["usage_summary"] = {
            name: {
                "known_requests": sum(u is not None and u.get(name) is not None for u in all_usage),
                "unknown_requests": sum(u is None or u.get(name) is None for u in all_usage),
                "known_partial_total": sum(
                    u[name] for u in all_usage if u is not None and u.get(name) is not None
                )
                if any(u is not None and u.get(name) is not None for u in all_usage)
                else None,
            }
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        _save(output, report)
        return report
    finally:
        if ledger:
            ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed core integration verification; real mode is opt-in"
    )
    parser.add_argument("mode", choices=("mock", "real"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--max-requests", type=int, default=30)
    args = parser.parse_args()
    report = evaluate_integration(
        args.output, mode=args.mode, model=args.model, max_requests=args.max_requests
    )
    print(
        json.dumps({"mode": args.mode, "counts": report["counts"], "requests": report["requests"]})
    )


if __name__ == "__main__":
    main()
