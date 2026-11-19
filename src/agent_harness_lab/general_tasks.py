"""Fixed Phase4 offline profile: task input and independent execution expectations.

Not business Gold, not a live reasoning benchmark. Changing a row changes the
manifest digest; baseline and ablation always use the same declared contract.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

VERSION = "harness-general-30-1.0.1"
ABLATIONS = ("none", "without_todo", "without_child", "without_skills", "without_compaction")


@dataclass(frozen=True)
class ProbeInput:
    operation: str
    category: str


@dataclass(frozen=True)
class ProbeOracle:
    stop: str = "completed"
    errors: tuple[str, ...] = ()
    written: bool = False
    feature: str = ""


@dataclass(frozen=True)
class GeneralTask:
    task_id: str
    task: ProbeInput
    expected: ProbeOracle


def tasks() -> tuple[GeneralTask, ...]:
    # Fixed prior to execution. Expected failures are safety/recovery contracts,
    # separate from terminal completion and from any measured model quality.
    rows = (
        ("read", "loop", ProbeOracle()),
        ("write", "completion", ProbeOracle(written=True)),
        ("escape", "security", ProbeOracle(errors=("workspace_path_denied",))),
        ("absolute", "security", ProbeOracle(errors=("workspace_path_denied",))),
        ("symlink", "security", ProbeOracle(errors=("workspace_path_denied",))),
        ("write_optout", "security", ProbeOracle(errors=("workspace_permission_denied",))),
        ("deny", "permissions", ProbeOracle(errors=("permission_denied",))),
        ("ask_yes", "permissions", ProbeOracle(written=True)),
        ("ask_no", "permissions", ProbeOracle(errors=("permission_denied",))),
        ("ask_missing", "permissions", ProbeOracle(errors=("permission_denied",))),
        ("ask_invalid", "permissions", ProbeOracle(errors=("permission_denied",))),
        ("ask_throw", "permissions", ProbeOracle(errors=("permission_denied",))),
        ("bad_json", "protocol", ProbeOracle(errors=("invalid_arguments",))),
        ("duplicate_key", "protocol", ProbeOracle(errors=("invalid_arguments",))),
        ("wrong_type", "protocol", ProbeOracle(errors=("invalid_arguments",))),
        ("unknown_tool", "protocol", ProbeOracle(errors=("unknown_tool",))),
        ("duplicate_call", "protocol", ProbeOracle(stop="model_protocol_error")),
        ("recover", "recovery", ProbeOracle(errors=("workspace_not_found",))),
        ("quota", "provider", ProbeOracle(stop="provider_error")),
        ("provider_protocol", "provider", ProbeOracle(stop="model_protocol_error")),
        ("model_budget", "budget", ProbeOracle(stop="max_model_rounds")),
        ("tool_budget", "budget", ProbeOracle(stop="max_tool_calls")),
        ("todo", "planning", ProbeOracle(feature="todo")),
        (
            "todo_invalid",
            "planning",
            ProbeOracle(errors=("todo_invalid_transition",), feature="todo_rejected"),
        ),
        ("child", "delegation", ProbeOracle(feature="child")),
        ("child_write", "security", ProbeOracle(feature="child_denied")),
        ("child_failure", "recovery", ProbeOracle(feature="child_failure")),
        ("skill", "skills", ProbeOracle(errors=("workspace_path_denied",), feature="skill")),
        (
            "skill_traversal",
            "security",
            ProbeOracle(errors=("skill_invalid_name",), feature="skill_rejected"),
        ),
        ("compact", "context", ProbeOracle(feature="compaction")),
    )
    return tuple(
        GeneralTask(f"G{i:02d}", ProbeInput(op, cat), oracle)
        for i, (op, cat, oracle) in enumerate(rows, 1)
    )


def manifest() -> dict[str, object]:
    return {
        "version": VERSION,
        "offline_only": True,
        "denominator": 30,
        "tasks": [asdict(t) for t in tasks()],
        "ablations": ABLATIONS,
        "metrics": (
            "contract_pass_rate_fixed30",
            "terminal_completion_count",
            "unsafe_handler_or_effect_count",
            "model_calls",
            "tool_calls",
            "component_delta_contracts_not_model_quality",
        ),
    }


def identity() -> str:
    return hashlib.sha256(json.dumps(manifest(), sort_keys=True).encode()).hexdigest()
