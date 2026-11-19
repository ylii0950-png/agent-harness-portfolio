"""Local-only workflow contract audit. Requires PyYAML==6.0.3, no runner execution.

actionlint checks the GitHub syntax/types separately. Scenario rows below are a
contract model, not a substitute for hosted event execution or a general GitHub
expression interpreter. No project runtime dependency is added.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

BASE = "6a55688172ba6bfac09d60e1c6982c94389e7743"
OS_EXPRESSION = (
    "${{ fromJSON(github.event_name == 'pull_request' && github.event.pull_request.draft "
    '&& \'["ubuntu-latest"]\' || \'["ubuntu-latest","macos-latest"]\') }}'
)
GROUP = (
    "${{ github.workflow }}-${{ github.event_name }}-"
    "${{ github.event.pull_request.number || github.ref }}"
)
EVENTS = ["opened", "synchronize", "reopened", "ready_for_review", "converted_to_draft"]


class Loader(yaml.SafeLoader):
    """YAML 1.2-style booleans, duplicate-key rejection; no aliases/custom tags."""


Loader.yaml_implicit_resolvers = {
    key: [(tag, pattern) for tag, pattern in items if tag != "tag:yaml.org,2002:bool"]
    for key, items in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.I), list("tTfF")
)


def mapping(loader: Loader, node: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("non_string_or_duplicate_yaml_key")
        result[key] = loader.construct_object(value_node)
    return result


Loader.add_constructor("tag:yaml.org,2002:map", mapping)


def load(raw: str) -> dict[str, Any]:
    if len(raw.encode()) > 64 * 1024:
        raise ValueError("workflow_size")
    if any(
        isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)) for t in yaml.scan(raw)
    ):
        raise ValueError("workflow_alias")
    result = yaml.load(raw, Loader=Loader)
    if not isinstance(result, dict):
        raise ValueError("workflow_mapping")
    return result


def audit(workflow: dict[str, Any], baseline: dict[str, Any]) -> None:
    assert workflow["name"] == baseline["name"] == "CI"
    assert workflow["on"] == {
        "push": {"branches": ["main"]},
        "pull_request": {"types": EVENTS},
        "workflow_dispatch": None,
    }, "events_or_filters"
    assert workflow["permissions"] == {"contents": "read"}, "token_permissions"
    assert workflow["concurrency"] == {"group": GROUP, "cancel-in-progress": True}, "concurrency"
    assert workflow.keys() == {"name", "on", "permissions", "concurrency", "jobs"}
    assert workflow["jobs"].keys() == baseline["jobs"].keys() == {"checks", "demo_checks"}
    for key, job in workflow["jobs"].items():
        old = baseline["jobs"][key]
        expected = copy.deepcopy(old)
        expected["strategy"]["matrix"]["os"] = OS_EXPRESSION
        assert job == expected, f"job_contract:{key}"
        assert job["strategy"]["matrix"]["python"] == ["3.11", "3.12"]
        assert job["strategy"]["fail-fast"] is False
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[a-zA-Z0-9_/-]+@[a-f0-9]{40}", step["uses"])
            assert "continue-on-error" not in step
    # No job-level skip/privilege path, no pull_request_target/path filter or
    # conditional omission of existing pytest/security/type/build/CLI steps.


def rows(event: str, draft: bool | None, ref: str, action: str = "") -> dict[str, Any]:
    triggered = (
        event == "workflow_dispatch"
        or (event == "push" and ref == "refs/heads/main")
        or (event == "pull_request" and action in EVENTS)
    )
    systems = (
        (
            ["ubuntu-latest"]
            if event == "pull_request" and draft is True
            else ["ubuntu-latest", "macos-latest"]
        )
        if triggered
        else []
    )
    contexts = [
        prefix + system + " / Python " + python
        for prefix in ("", "M1 Demo / ")
        for system in systems
        for python in ("3.11", "3.12")
    ]
    return {
        "event": event,
        "action": action,
        "draft": draft,
        "ref": ref,
        "triggered": triggered,
        "jobs": len(contexts),
        "contexts": contexts,
    }


def self_test(workflow: dict[str, Any], baseline: dict[str, Any]) -> int:
    mutations = []
    changed = copy.deepcopy(workflow)
    changed["on"]["push"] = None
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["on"]["pull_request"]["types"].remove("ready_for_review")
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["on"]["pull_request_target"] = None
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["on"]["pull_request"]["paths"] = ["src/**"]
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["permissions"] = {"contents": "write"}
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["concurrency"]["group"] = "CI"
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["jobs"]["checks"]["name"] = "renamed"
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["jobs"]["checks"]["strategy"]["matrix"]["python"] = ["3.11"]
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["jobs"]["demo_checks"]["strategy"]["matrix"]["os"] = ["ubuntu-latest"]
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["jobs"]["checks"]["steps"].pop(3)
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["jobs"]["checks"]["steps"][0]["with"]["persist-credentials"] = True
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["jobs"]["checks"]["steps"][0]["uses"] = "actions/checkout@main"
    mutations.append(changed)
    changed = copy.deepcopy(workflow)
    changed["jobs"]["checks"]["if"] = False
    mutations.append(changed)
    for number, changed in enumerate(mutations):
        try:
            audit(changed, baseline)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"undetected_mutation:{number}")
    for raw in ("name: CI\nname: changed", "a: &x [1]\nb: *x", "!!python/object:unsafe {}"):
        try:
            load(raw)
        except (ValueError, yaml.YAMLError):
            pass
        else:
            raise AssertionError("unsafe_yaml_accepted")
    return len(mutations) + 3


def main() -> None:
    if not __debug__ or yaml.__version__ != "6.0.3":
        raise RuntimeError("Use non-optimized Python with PyYAML==6.0.3")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    current = Path(".github/workflows/ci.yml").read_text()
    old = subprocess.check_output(["git", "show", f"{BASE}:.github/workflows/ci.yml"], text=True)
    workflow, baseline = load(current), load(old)
    audit(workflow, baseline)
    negatives = self_test(workflow, baseline)
    cases = [
        rows("push", None, "refs/heads/main"),
        rows("push", None, "refs/heads/codex/ci-efficiency"),
        rows("push", None, "refs/tags/v1"),
        rows("workflow_dispatch", None, "refs/heads/codex/ci-efficiency"),
        *[
            rows("pull_request", draft, "refs/pull/16/merge", action)
            for action in EVENTS
            for draft in (True, False)
        ],
        rows("pull_request", None, "refs/pull/16/merge", "synchronize"),
        rows("pull_request", False, "refs/pull/16/merge", "edited"),
    ]
    expected_jobs = [8, 0, 0, 8] + [n for _ in EVENTS for n in (4, 8)] + [8, 0]
    assert [row["jobs"] for row in cases] == expected_jobs
    result = {
        "baseline": BASE,
        "workflow_sha256": hashlib.sha256(current.encode()).hexdigest(),
        "yaml_parser": "PyYAML6.0.3 SafeLoader with strict keys/no aliases/YAML1.2 booleans",
        "negative_contract_tests_passed": negatives,
        "actual_job_step_objects_equal_baseline": True,
        "scenarios": cases,
        "scenario_count": len(cases),
        "scenario_validation": "literal-contract model; not execution of GitHub expression engine",
        "required_contexts_proposal": rows("workflow_dispatch", None, "refs/heads/main")[
            "contexts"
        ],
        "hosted_runs": "separate_after_budget_confirmation",
        "private_gold_loaded": False,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
