# Clean installation and reproduction

Use a clean Git checkout with Python 3.11 or 3.12 and uv 0.13.5. Several public
Demo regression tests bind synthetic inputs to the current Git commit. The
default checks require no credentials, Docker or model assets.
Run from the repository root. Replace every `3.12` in the commands below with
`3.11` to select the other supported version. Keep the same `--python` selection
on every `uv sync`, including optional dependencies. Check the environment's own
executable after each sync; `uv run --no-sync --python` can warn about an
incompatible environment and still run its existing interpreter. Stop on that
warning; a successful exit alone does not verify the selected Python version.

```sh
uv sync --locked --no-editable --python 3.12
.venv/bin/python -c 'import sys; assert sys.version.startswith("3.12."); print(sys.version)'
uv build --no-build-isolation --python "$PWD/.venv/bin/python"
uv build --project examples/knowledge_ticket_agent --no-build-isolation --python "$PWD/.venv/bin/python"
uv pip install --python .venv/bin/python --no-index --no-deps examples/knowledge_ticket_agent/dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl
uv run --no-sync --python 3.12 python -m agent_harness_lab.demo
uv run --no-sync --python 3.12 lqd-validate
uv run --no-sync --python 3.12 lqd-agent-demo
uv run --no-sync --python 3.12 lqd-agent-demo --scenario conflict
uv run --no-sync --python 3.12 lqd-agent-demo --scenario missing-facts
```

`uv sync --no-editable` installs Core, not the separate Demo. The second build
packages Demo sources and fictional resources independently; install its wheel
explicitly. No source-path injection. Development dependencies come from uv.lock.
`uv sync` prunes separately installed packages, so reinstall the Demo wheel
after a later sync. Selecting the interpreter on sync prevents the repository's
pinned3.12 from replacing a3.11 environment. Ordinary runtime-only installation
can instead use both built wheels:

```sh
uv venv .venv-clean --python 3.12
uv pip install --python .venv-clean/bin/python --no-index --no-deps dist/agent_harness_lab-0.9.0-py3-none-any.whl examples/knowledge_ticket_agent/dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl
.venv-clean/bin/python -m agent_harness_lab.demo
.venv-clean/bin/lqd-validate
.venv-clean/bin/lqd-agent-demo --scenario valid
```

The clean wheel-only environment has no optional SDK or OAuth dependencies.
Both builds produce a wheel and a source distribution. The source distribution
uses the pinned setuptools build dependency.

## Optional capabilities

The locked dev group already supplies MCP for tests. For an SDK-enabled sync:

```sh
uv sync --locked --no-editable --extra mcp --python 3.12
.venv/bin/python -c 'import sys; assert sys.version.startswith("3.12."); print(sys.version)'
uv pip install --python .venv/bin/python --no-index --no-deps examples/knowledge_ticket_agent/dist/knowledge_ticket_agent_demo-0.1.0-py3-none-any.whl
uv run --no-sync --python 3.12 python -m agent_harness_lab.mcp_demo
uv run --no-sync --python 3.12 python -m agent_harness_lab.memory_demo
uv run --no-sync --python 3.12 python -m agent_harness_lab.sandbox_demo --check
```

MCP uses an actual trusted local stdio server in a separate process. Memory demo
uses six processes, a disposable owner-only SQLite database and three scripted
ASK approvals; it verifies persistence, retrieval, update and deletion. These
commands use Mock responses, not AI-generated decisions.

Sandbox execution additionally requires Docker and a preloaded immutable Linux
Python image ID. Provision the image separately under trusted local setup; the
backend never downloads or executes on the host as a fallback:

```sh
uv run --no-sync --python 3.12 python -m agent_harness_lab.sandbox_demo --enable-sandbox --image sha256:LOCAL_IMAGE_ID --approval approve
```

The placeholder must be replaced with an actual local image ID. Exit success is
not an isolation proof. The unchanged [Linux CI test](../.github/workflows/docker-sandbox.yml)
checks actual constraints; macOS without Docker reports unavailable explicitly.

## Regression and integration

```sh
uv run --no-sync --python 3.12 pytest
uv run --no-sync --python 3.12 pytest -c examples/knowledge_ticket_agent/pyproject.toml examples/knowledge_ticket_agent/tests
uv run --no-sync --python 3.12 ruff check .
uv run --no-sync --python 3.12 ruff format --check .
uv run --no-sync --python 3.12 mypy
uv run --no-sync --python 3.12 mypy --strict examples/knowledge_ticket_agent/src examples/knowledge_ticket_agent/tests
```

The documented checks use public fixtures and remain offline.
`lqd-agent-demo --json` includes actual fictional tool/trace/audit observations;
its default human output is shorter. See [observed outputs](demo-output.md).
