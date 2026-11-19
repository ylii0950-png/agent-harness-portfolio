# Captured offline demonstrations — 2026-10-27

Actual installed-package outputs, fictional inputs only. All model steps here
are scripted Mock, not real autonomous model decisions. Temporary files, actual
stdio RPC and SQLite effects are real. Business tools deterministically validate
evidence/facts/permission and mutate only the per-run in-memory ticket store.

## General Harness

`python -m agent_harness_lab.demo`:

```text
CALL call_1 write_file {"path": "note.txt", "content": "Hello from the Phase 1 mock agent.\n"}
RESULT call_1 {"content": "File written.", "error": null, "ok": true}
CALL call_2 read_file {"path":"note.txt"}
RESULT call_2 {"content": "Hello from the Phase 1 mock agent.\n", "error": null, "ok": true}
FINAL Created and read note.txt successfully.
STOP completed (model_rounds=3, tool_calls=2)
```

`python -m agent_harness_lab.todo_demo`: actual read/write,6 scripted ASK
approvals, completed Todo2/2, trace created1/updated4,10 model rounds/9 tools.
The receipt is read from the actual workspace, not inferred from final text.

## Business

Run `lqd-agent-demo --scenario NAME` for human output; add `--json` for actual
trace, tool observations and business audit. `all` retains the original three
scenarios; conflict and missing-facts are explicit separate commands.

| Scenario | Trusted assessment | Parent full sources | ASK | Handler / commits | State | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| valid | eligible | 6/6 | approved | 1 / 1 | open r0 → resolved r1 | PASS |
| insufficient | eligible | 3/6 | not_requested | 0 / 0 | open r0 → open r0 | PASS |
| denied | eligible | 5/5 | approval_denied | 0 / 0 | open r0 → open r0 | PASS |
| conflict | policy_conflict | 8/8 | not_requested | 0 / 0 | open r0 → open r0 | PASS |
| missing-facts | load_error | 6/6 | not_requested | 0 / 0 | open r0 → open r0 | PASS |

Approved readback verified. Other four snapshots verified unchanged. The
insufficient case can be rule-eligible while lacking Parent documentary evidence:
it still receives no executable proposal or ASK. Missing-facts produces load_error
with6/6 policy sources; documents cannot fill missing trusted invoice facts.
A PASS here means the expected scenario contract, including safe denial.

## Optional extensions

`python -m agent_harness_lab.mcp_demo` (actual stdio):

```json
{"stop": "completed", "protocol_calls": 1, "model_calls": 2, "tool_calls": 1, "call_id": "mcp-demo-1", "origin": "external_mcp"}
```

`python -m agent_harness_lab.memory_demo` (six real process lifetimes):

```json
{"cross_process_persistence": "PASS", "processes": 6, "enabled_answer": "NEBULA-52", "disabled_answer": "UNKNOWN", "update": "PASS", "delete_readback": "PASS", "ASK_calls": 3, "mode": "offline_mock_not_model_intelligence"}
```

`python -m agent_harness_lab.sandbox_demo --check` on this macOS host:

```json
{"docker_available": false, "actual_container_run": false, "isolation": "NOT_VERIFIED"}
```

Actual Linux restrictions were verified separately by the unchanged container
CI test: UID65534, capabilities0, NNP1, denied root/workspace writes, no Docker
socket, loopback-only interface and bounded CPU/memory/PIDs. See the
[Docker CI workflow](../.github/workflows/docker-sandbox.yml) for the test setup.
