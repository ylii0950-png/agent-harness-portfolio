# Portfolio MVP demo

Build Core 0.9.0 and Demo 0.1.0 into `.venv-portfolio` using the
[README](../README.md). All tickets, actors, policies and approval choices here
are fictional. The default path uses A1 and deterministic offline Mock.

```sh
.venv-portfolio/bin/python -m agent_harness_lab.demo
.venv-portfolio/bin/lqd-search "first purchase credit" --category billing_refund
.venv-portfolio/bin/lqd-agent-demo --ticket KT-D01 --language en \
  --request-text "Please process this first purchase credit policy request." --approval approve --json
.venv-portfolio/bin/lqd-agent-demo --ticket KT-D01 --language zh-CN \
  --request-text "请按首次购买的购买额度调整政策处理这张模拟工单。" --approval refuse --json
.venv-portfolio/bin/lqd-agent-demo --scenario insufficient --json
.venv-portfolio/bin/lqd-agent-demo --scenario missing-facts --json
.venv-portfolio/bin/lqd-agent-demo --scenario conflict --json
```

| Installed demo | Observed result |
| --- | --- |
| Core | Actual temporary-file WRITE and READ; 3 rounds, 2 tool calls, completed |
| Read-only policy search | 3 valid canonical citation checks |
| English, approved | 6 full source reads, 6 verified citations, exact ASK approved, one journaled commit, open r0 → resolved r1, readback verified |
| Chinese, refused | 6 full source reads, 6 verified citations, ASK denied, 0 handler entries and 0 commits, unchanged |
| Missing evidence / invoice / conflicting policy | Expected safe denial, 0 handler entries and 0 commits, unchanged |

For Controller and fault controls, use the locked test environment with both wheels
installed as described in [reproduce.md](reproduce.md). The Controller runs with a
synthetic Responses transport through the real application and Tool Registry.

```sh
uv run --no-sync pytest -c examples/knowledge_ticket_agent/pyproject.toml \
  examples/knowledge_ticket_agent/tests/test_workflow_controller.py \
  examples/knowledge_ticket_agent/tests/test_business_demo.py::test_budget_stop_is_not_a_verified_business_result \
  examples/knowledge_ticket_agent/tests/test_phase3_completion.py::test_atomic_failure_and_failed_postcommit_readback_remain_truthful \
  examples/knowledge_ticket_agent/tests/test_evidence_contract.py::test_canonical_source_tampering_and_partial_reads_are_rejected
```

Controller approval committed once and verified state; refusal committed zero
times. Budget exhaustion and corrupt evidence stopped with zero commits. The
atomic abort preserved the initial snapshot. The postcommit readback fault kept
the actual commit and reported `postcommit_verification_error`.

## Optional offline R1

Prepare the pinned assets once using the documented [E5 setup](systematic-remediation.md#running-the-experimental-profile)
and [BGE preparation](retrieval-relevance-evidence.md#reproduction-and-integration).
The Demo's locked semantic extra supplies ONNX Runtime and Tokenizers. These
commands assume `E5_DIR` and `BGE_DIR` point to the prepared directories.

```sh
uv sync --project examples/knowledge_ticket_agent --locked --no-editable --extra semantic --python 3.12
uv run --project examples/knowledge_ticket_agent --no-sync lqd-agent-demo \
  --ticket KT-D01 --language zh-CN --request-text "请按首次购买的购买额度调整政策处理这张模拟工单。" \
  --approval approve --retrieval-profile vnext --retrieval-variant R1 \
  --model-directory "$E5_DIR" --pair-model-directory "$BGE_DIR" --json
```

Observed one commit and verified readback with R1 3.0.1 and the trusted M3
documentary path. The separate read-only source-assertion flow used Evidence
3.0.0, Source Selection 2.0.0 and Assertion Search View 1.0.0. Its three native
Agent cases returned SUPPORTED from full source reads. These results cover
declared-domain fixtures. Generalization outside the declared domain is unverified.

See the [machine-readable receipt](portfolio-demo-receipt.json) for observed
outcomes, installed source fingerprint and model/config identities.
