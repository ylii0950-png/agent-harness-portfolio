# Upstream reference

- Repository: https://github.com/shareAI-lab/learn-claude-code
- Default branch inspected: `main`
- Commit: `ce8f9f186058939da54c9d6fead78dfb5d0fd6c3`
- Commit timestamp: `2026-09-28T13:49:16Z`
- Inspection date: 2026-10-22 (Asia/Kuala_Lumpur)
- License: MIT, Copyright (c) 2024 shareAI Lab

The snapshot contains top-level `s01_agent_loop` through `s17_goal_loop` tutorial
directories, `agents/`, `docs/`, `skills/`, `tests/`, and `web/`. The inspected
`s01_agent_loop`, `s02_tool_use`, and `s03_permission` directories each contain
`code.py`, three language README files, and `images/`. The older `agents/`
directory also contains session implementations with a different numbering
sequence, so directory names are recorded rather than inferred from old lessons.
See `upstream-reference.json` for the exact inspected listing and license hash.

Only repository metadata, directory listings, and the full license were fetched
for Phase 0. No upstream Python, setup command, workflow, or executable was run.
The local provider API, descriptor-based file tools, and tests were written
independently for this project; this is not a tutorial-code fork or endorsement.

The complete upstream license is retained verbatim at
`licenses/learn-claude-code-MIT.txt`. `LICENSE` covers this project's original
work, and `NOTICE` records the upstream attribution. Setuptools includes both
licenses and NOTICE in source and wheel distributions. Any later reuse of code
or substantial documentation must preserve its original notices and be recorded.

## Phase 1 inspection

Read `s01_agent_loop/code.py` and `s02_tool_use/code.py` at the same pinned SHA;
file URLs and SHA-256 digests are appended to `upstream-reference.json`.
The first illustrates model/tool feedback; the second adds named dispatch and
file tools. Both include model SDK/shell execution that this project does not
adopt. No upstream code, setup scripts, or tutorial commands were executed or
copied. The local implementation instead uses finite synchronous iteration,
validated immutable decisions, explicit quotas, capability-checked dispatch,
and the unchanged Phase 0 descriptor-based WorkspaceFiles implementation.
No reference-version or license change was made.

## Phase 2.1 inspection

On 2026-10-23 read `s03_permission/code.py` and its English README at the same
pinned SHA `ce8f9f186058939da54c9d6fead78dfb5d0fd6c3`. Exact source URLs and
SHA-256 digests are appended under `phase21_code_inspection` in
`upstream-reference.json`. The chapter illustrates hard denial, rule matching,
and human approval before dispatch. It also exposes shell execution, displays
raw arguments, and can ask permission for paths outside the workspace; these
teaching choices are not adopted here. Our original WorkspaceFiles boundary
continues to refuse those paths even after approval. No upstream code or command
was executed or copied. Complete upstream attribution and license are unchanged.

## Phase 2.2 s05 inspection

Read `s05_todo_write/code.py` and its README at the same pinned SHA on
2026-10-23; exact content hashes are in `phase22_code_inspection` in
[the reference record](upstream-reference.json). No upstream code or commands
were executed or copied. The upstream bounded list and single active task inform
the design, but this project independently implements IDs, atomic strict forward
transitions and per-run scope. It does not adopt upstream's global TODO manager,
shell, hooks, model calls, literal fallback, or reminder prompt injection.
The complete upstream MIT license and NOTICE remain unchanged.

## Phase 2.3 s06 inspection

Read pinned `s06_subagent/code.py` and README on 2026-10-23 without executing
commands/code or copying Python. Exact hashes are in `phase23_code_inspection`
in [the reference record](upstream-reference.json). Upstream uses fresh child
messages, base tools without task delegation, and final-summary return. This
project independently reuses its single Agent Loop, immutable permission engine,
narrowed tool scopes, isolated Todo state and shared/local budgets instead of
copying a second loop, global histories, hooks, arbitrary shell or API client.
The pinned SHA and complete MIT license/NOTICE remain unchanged. Message/context
isolation is distinct from operating-system isolation in both implementations.

## Phase 2.4 s07 inspection

Read pinned s07_skill_loading/code.py and README on 2026-10-23, as data only.
Exact digests are in `phase24_code_inspection` in
[the reference record](upstream-reference.json). No upstream command or Python
was executed/copied. Upstream provides the */SKILL.md name/description structure
and catalog-first/tool-result concept, but reads/caches full documents at scan.
This project independently reads only headers at discovery, delays body I/O,
rejects unsupported metadata rather than adding full YAML/body fallbacks, and
uses existing safe file/permission/run boundaries. No system-level skill prompt,
shell, hooks, global body cache or model client is adopted. Reference SHA and
complete MIT license/NOTICE stay unchanged.


## Phase 2.5 reference inspection

Read pinned `s08_context_compact/code.py` and `README.md` as untrusted data only;
exact digests are appended to upstream-reference.json. Retain the original SHA,
complete MIT license and attribution. No upstream commands/setup/code executed.
Our independent implementation keeps the safe model boundary and call pairing
concept, but uses deterministic in-memory low-trust previews, no file persistence,
model summarization or reactive inference retry. See [architecture](architecture.md).
