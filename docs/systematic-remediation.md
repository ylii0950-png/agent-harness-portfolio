# Systematic remediation: audit and fixed implementation plan

2026-11-02. PO authorizes implementation, affected Phase interfaces and a finite
retrieval architecture comparison. PR38's negative study is merged normally at
`d3c4ecbcad6adc8c265ec6e0ebeec62752274b9a`; post-main run41333180197 passed8/8.
Its original A1/G1/G2 conclusions remain unchanged. No private acceptance input,
individual receipt, claim, approval or execution root is inspected. Historical
M2 FAIL is consumed, M5 NOT_RUN, Final Release BLOCKED.

## Repository inspection and dependency map

Tracked inspection covers193 Python files and16 principal policy/config/workflow
documents. The full structural index records each file's hash, definitions,
imports and test functions; a second complete AST scan records authorization,
source/file, compaction, budget and commit call sites plus test assertions.
Artifacts: `artifacts/systematic-remediation-20261102/{tracked-review-index,
call-boundary-review}.json`. This is code/static dependency review, with manual
review of critical execution paths below, not an independent security acceptance.
It does not read actual private records behind any inspected operator source.

Core contracts/providers -> Agent runtime -> Registry -> PermissionEngine ->
trusted handlers. Context trims only provider-visible result content; execution
records/budgets remain separate. Todo/Skills/Subagent narrow the same Registry
and shared RunBudget. No Core source imports the business package. Optional
MCP/memory/sandbox use existing dispatch; none needs replacement for retrieval.

Application: loader/strict schema/validation -> original-span index -> retriever
-> actual kb_search/kb_get observations -> exact citation/parent evidence ->
trusted rules -> immutable proposal -> exact ASK -> atomic TicketStore snapshot
and commit journal -> actual readback -> independently checked final receipt.
Public evaluation supplies task inputs to runtime and keeps oracle grading
outside it. Restricted/formal gates additionally bind exact release identities
and one-shot authority; their old protocols must not gain the new opt-in profile.

## Findings and implementation consequences

| ID / classification | Exact location and evidence / trigger | Impact / root cause | Smallest repair and verification |
| --- | --- | --- | --- |
| R1 quality bottleneck, P1 | retrieval.py:terms/lexical_rank; retrieval_strategies.py:StrategyRetriever.search. PR38 retains5 DEV A1 misses; function-word-only G1 cannot recover them. | M2 lacks a learned semantic representation when terms differ; downstream navigation completeness cannot establish ranking quality. | Separate versioned local multilingual encoder and semantic/hybrid retriever; fresh English/Chinese/competitive-source ranking comparison. Keep A1 unchanged. |
| R2 architecture limitation, P1 | business_demo.py:run_ticket; workflow_controller.py:WorkflowController.__init__; evaluation_runtime.py:execute_public_task; agent_tools.py:DemoApplication.__init__. Each constructs/assigns a concrete retriever. | New initial ranking can be measured without the same configured strategy reaching actual tools. Multiple construction sites make provenance drift easy. | Explicit trusted retriever injection with index binding, one model/profile identity, tool-result method audit, CLI selection and E2E tests. Legacy defaults remain identical. |
| R3 quality bottleneck, P1 | retrieval_profiles.py:A1Retriever.search (>256 characters prioritizes title/heading vocabulary, truncating remaining query). | Body-only or late query intent can be lost; bilingual fixed bridging is not general semantics. | New profile processes every bounded query segment and encoder token window, with explicit limits/errors; test late intent and mixed/CJK requests. Do not rewrite old query-version semantics. |
| R4 evaluation coupling limitation, P1 | evaluation_runtime.py:execute_public_task obtains provider queries from source_script(app), while the initial ranking task has its own query. | A successful scripted catalog acquisition cannot demonstrate the same query's initial retrieval. This is historical orchestration, not proof of ranking success. | New opt-in public runtime begins with task.query, then retains bounded verified navigation. Assert genuine first search/query/method, separate initial vs acquired coverage. Reject this opt-in at sealed entry points. |
| R5 necessary safety complexity | quality_evidence.py:assess_parent_answer/acquire; agent_tools.py:verified_parent_citations. Actual root records, preceding search, full get, authority and spans are rechecked. | Compaction previews, child text and high ranking scores must not become evidence or permission. | Preserve checks; test tampering, context compaction and missing sources with new retriever. No new write path or query-derived trusted facts. |
| R6 normal frozen-fixture constraint | validation.py:validate_dataset fixes full/seed inventories, counts and clock; chunking.py:build_index binds every source byte. | Arbitrary dataset replacement is intentionally disallowed in the historical fixture. | New public in-memory corpus retains rule-linked sections, rebinds new bytes/manifest, passes original validation; no relaxed validator or rewritten packaged policy. |
| R7 release compatibility risk | formal_evaluation.py:FormalRelease and operator source_digests/current_revision checks. | Adding optional modules changes the whole-source inventory even if A1 behavior is identical. | Report new fingerprint and versions; old M5 3.0.0 cannot be claimed to cover vNext. No new formal profile, approval, claim root or retry. |
| R8 necessary correctness complexity | approvals.py:prepare/_valid/_apply_transition/execute; ticket_store.py:_commit. | Authorization, freshness, replay binding and one snapshot publication jointly protect journal/state. | Retain primitives; exercise refuse/invalid/stale approval, atomic abort, replay and readback faults through existing interfaces. |
| R9 necessary Core boundaries | permissions.py:authorize; tools.py:invoke; runtime.py:_run; context.py:prepare; subagents.py:_dispatch. | Exact bool approval, validated full batches, shared quotas, untrusted context and fresh children are complementary controls. | No speculative Core rewrite; complete Core regression plus new application compaction/provenance tests. No P0 bypass confirmed by this review. |
| R10 operational risk for new dependency | default packages currently need no encoder runtime or model weights. | Automatic downloads, mutable weights or silent lexical fallback would break offline reproducibility and honest method identity. | Optional Demo extra, fixed revision/file hashes, explicit preparation only, CPU-only pinned runtime, no custom remote code, missing/corrupt model explicit failure. Standard CI never downloads models. |
| R11 historical complexity | versioned public/formal evaluators, legacy strategies, Gold schemas and operator tests. | Similar functions preserve consumed/failed histories and old contracts, not proven dead code. | No deletion now. Only after quality stabilizes, simplify demonstrated new duplication without erasing historical execution. |

No repository-wide LOC reduction or new generic framework is part of this repair.

## Fixed vNext candidates and model identity (before implementation)

Use original MIT-licensed `intfloat/multilingual-e5-small` ONNX artifact at
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, quantized118346824-byte model
SHA256 `dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88`;
tokenizer SHA256 `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`.
Complete file lock retained before coding. CPU ONNX Runtime1.31.1,
tokenizers0.23.3; dependency resolution locked separately for the Demo extra.
No training, paid inference, model search or downloaded Python execution.
Use attention-mask mean pooling/L2 and query:/passage: prefixes, per the
[upstream model card](https://huggingface.co/intfloat/multilingual-e5-small).
Upstream quantization portability is verified locally, not assumed from its name.

At most two immutable candidates: D semantic-only control and H hybrid coverage.
Both reuse original source chunks/catalog, filter scope/authority before scoring,
retain every query sentence/bounded segment, and verify original citation spans.
D ranks cosine similarity. H unions top16 dense and top16 lexical candidates,
reranks by dense score + .10 normalized BM25, then selects up to3 with .08 penalty
for excess cosine redundancy above.90 and .03 incremental query-facet coverage.
No hard one-document cap; complementary sections may come from the same policy.
Similarity floor.70 is a retrieval heuristic, never proof of entailment or grant.
Empty queries return no_match. Model/config/input/shape/NaN errors fail explicitly,
with no fallback. Token windows include the entire bounded input, not prefix-only
truncation. Limits:4000 query characters,32 segments/windows,512 tokens/window,
384 dimensions,1 CPU thread, bounded batches and in-memory embedding cache.
Version strategy/query/model/config/method independently; no rebranding as A1.

## Fixed new public evaluation

New independently authored public DEV48/challenge48, each8 distinct service
intents × canonical/paraphrase/Chinese/mixed/long/complementary queries. Each
split uses one fixed, source-bound fictional corpus with related policy pairs,
preserved old rule-linked scaffolding, and separate rule/record sections.
Prior PR38 data are historical regression only, never the new selection set.

| Public version1.0.0 | SHA256 |
| --- | --- |
| DEV48 | 57ae516cc3f1cde82be1ef181433f23c0759a58fb496083aedc122bc97b92d4d |
| Challenge48 | 5ca6cbbeb4c7ae6ace7f3b81fa2da1d91665a588c37bbd2ef043c99182de1eeb |

Scorer-only source expectations never enter search inputs. Compare unchanged A1,
D and H; report every row/error, Macro Recall@3, MRR@3, complete verified evidence,
English/CJK/mixed/variant groups, regressions, empty/partial/conflict/FAQ/category
and tamper controls. Measure actual local model startup/encoding/search CPU/RSS,
determinism and package/runtime versions. Keep original adverse results immutable.

Also run actual existing public business tasks/DEV tickets through both A1 and
new profile: real Registry tools, canonical full gets, rules, approval/refusal,
missing facts/sources, actor denial, atomic/readback/provenance faults and forced
compaction. Explicit public oracle expectations stay outside runtime. Audit the
initial method/query and actual runtime method separately. Business simulation
and model-free control tests are engineering evidence, not semantic or blind PASS.

Select on DEV once: >=.10 macro Recall gain OR >=.10 MRR gain with nondecreasing
recall; >=.05 English Recall gain; nondecreasing complete evidence; <=4 recall
regressions/48; zero unauthorized effects or source/permission violations; no
unrecorded errors/fallback; query p95<2s and peak process RSS<2GB on this host.
Tie: Recall, MRR, H before D. Freeze selection/config before one final challenge;
challenge must sustain the same benefit/safety limits. No adaptive challenge
tuning, candidate/threshold expansion or model swapping. If not met, retain a
working opt-in vNext and report failure objectively, with no readiness claim.

Full installed Python3.11/3.12 Core/Demo/public M2/M5/security regression, English
and Chinese E2E, Ruff, strict mypy, wheel/sdist parity, clean install and optional
real-model offline validation precede one grouped implementation PR. No model
downloads in normal CI. Costs/limits remain$0 overage; no protection bypass.
No successor formal data, protocol, root, Source Lock or authorization is created.

## Pre-challenge implementation correction (2026-11-02)

The first real DEV run is retained unchanged under local evidence
`dev-{A1,D,H}-first.json`: A1 Recall/MRR .625/.486111; D .895833/.75;
H .854167/.694444. All 13 nominal business cases were correct, but D/H returned
three sources for the unrelated OOV control. No candidate was selected and no
challenge run occurred. These are adverse results, not an accepted release.

A tokenizer boundary diagnostic then confirmed that encoding a prefix ending in
space alone adds token6, absent from prefix+text. Reconstructing windows with this
header discarded the first body token. Correct the header to its trimmed encoding,
require it to match the complete input, and test that a short input is unchanged.
Strategy/query patch versions become1.0.1; model, weights, data, ranking constants,
selection rules and the two candidate families are unchanged. Re-evaluate both
corrected implementations once on DEV before freezing the challenge decision.
The original reports and identities remain preserved.

The first fault observer also expected a nonempty grader issue list for readback
failure. The existing runtime instead correctly returned tool_error, one authorized
commit and unverified readback. Correct this observer to require those facts;
retain the first9/10 control reports. This changes measurement, not the commit,
journal, readback implementation or historical formal grader.

## Final public comparison and stop decision

The DEV choice was committed at98058c3 before any challenge invocation; decision
SHA256 `da6189e4a9efaf7bfdf9be0dd3ce76d4b986fa5d496d333439aef4311ee0c450`.
All initial, corrected DEV and one-time challenge reports are retained in
`examples/knowledge_ticket_agent/verification/retrieval-vnext-v1/results/`.
D was provisionally selected by the fixed DEV ordering, with the OOV failure
explicitly blocking readiness. No candidate was changed after challenge exposure.

| Split / method | Macro Recall@3 | MRR@3 | Complete evidence | English Recall | Recall regressions vs A1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEV48 A1 | .625000 | .486111 | .604167 | .687500 | 0 |
| DEV48 D | .906250 | .819444 | .833333 | .859375 | 3 |
| DEV48 H | .885417 | .743056 | .833333 | .828125 | 1 |
| Challenge48 A1 | .729167 | .579861 | .666667 | .843750 | 0 |
| Challenge48 D | .812500 | .760417 | .770833 | .812500 | 4 |
| Challenge48 H | .864583 | .788194 | .812500 | .828125 | 1 |

Each split also has8 retrieval controls,13 nominal business outcomes and10
injected fault controls. Each method obtained13/13 business outcomes,10/10 fault
controls, zero observed unauthorized effects, no ranking errors, and deterministic
three repeated rankings per row. A1 passed8/8 retrieval controls; D and H passed
7/8: an unrelated OOV string still returns3 authoritative-but-irrelevant sources.
A valid source/citation is not proof of relevance. No threshold was tuned to hide
this failure. The old product's finite unsupported-intent guard separately blocks
such business requests; this is not general semantic rejection.

**STOP: neither candidate qualifies for independent acceptance preparation.**
Selected D fails the challenge English-gain condition (delta-.031250), retains
4 recall regressions, and fails OOV abstention. H fails the same English-gain
condition (delta-.015625) and OOV control; its better challenge score cannot
retroactively replace the DEV choice. Both improve aggregate ranking/CJK coverage,
but complementary-source coverage remains a bottleneck. Challenge complementary
Recall: A1 .375, D .375, H .3125. Max-over-segment cosine does not reliably encode
AND evidence obligations, and the weak diversity gain can displace useful sections.
This is observed architecture limitation, not evidence about any hidden failure.
No more models, candidates, weights, thresholds or challenge runs are added.

On macOS arm64/Python3.12, D/H measured query p95 .0123/.0139 seconds on challenge
(first call per row, including citation verification; initial corpus encoding is
in the first row), startup about.83s and peak process RSS about818MB. DEV D startup
2.01s. These are this host's observations, not a cross-platform latency guarantee.
A1 peak RSS about41MB. Base Core/Demo remain optional-dependency-free; the semantic
extra adds pinned native/runtime transitive dependencies and about130MiB local
assets. Actual package versions and CPU/memory/timing are in every report.

## Running the experimental profile

The default application remains A1. Both D and H are experimental; explicitly
name D to reproduce the DEV-selected candidate (the optional CLI variant default
is H, as originally preregistered). No live workflow CLI enables vNext.

```sh
uv sync --project examples/knowledge_ticket_agent --locked --no-editable --extra semantic --python 3.12
MODEL_DIRECTORY="$HOME/.cache/agent-harness-lab/models/e5-small-614241f-q8"
# Explicit one-time public model preparation; never done by imports, search or CI.
uv run --project examples/knowledge_ticket_agent --no-sync python -m knowledge_ticket_agent.semantic_model "$MODEL_DIRECTORY" --prepare
# Later checks and all inference use local assets only; corrupt/missing assets fail.
uv run --project examples/knowledge_ticket_agent --no-sync python -m knowledge_ticket_agent.semantic_model "$MODEL_DIRECTORY"
uv run --project examples/knowledge_ticket_agent --no-sync lqd-agent-demo --ticket KT-D01 --retrieval-profile vnext --retrieval-variant D --model-directory "$MODEL_DIRECTORY" --approval refuse --json
uv run --project examples/knowledge_ticket_agent --no-sync python examples/knowledge_ticket_agent/tests/vnext_offline_smoke.py --model-directory "$MODEL_DIRECTORY" --output /tmp/vnext-new-smoke.json
```

Public measurement uses `tests/vnext_public_evaluation.py dev A1|D|H`, an explicit
model directory for D/H and a new `--output` path. Challenge additionally requires
`--dev-decision` bound to the helper and installed source fingerprint. Saved results
are immutable records; any local reproduction is public engineering, never formal.
Do not use the already-exposed challenge for future strategy selection.

## Cross-Phase validation and compatibility

Fresh external exports of source commitfdcff4c, both Python3.11/3.12:96 focused
PASS first, Core848 PASS/1 Docker environment skip, Demo938 PASS, Ruff lint/format,
Core mypy and strict Demo mypy PASS. Both wheel/sdist payload inventories match
59 Core/85 Demo files; clean installation, valid/conflict/missing-facts and Chinese
approve/refuse CLI receipts PASS. No system Python changed or path injection used.
Final documentation/results and the explicit native smoke helper are later
additions, not changes to the tested110-module production source inventory.

Explicit native smoke on both Python versions passes direct complete-input pooling
parity (<1e-6 maximum vector error),8 English/Chinese/mixed/unsupported request
flows,2 forced-compaction flows and2 supervised Controller approve/refuse flows.
Actual Registry full-source observations, exact ASK, atomic commit/journal/readback,
method identity and unchanged denial state are checked. The encoder is real E5;
the decision provider is offline Mock, not a claim about live model reasoning.
Full existing tests include non-consuming public/synthetic FormalOperator claim,
fail-closed and recovery contracts. No actual independent root was opened/run.

| Contract / identity | Effect |
| --- | --- |
| Core0.9.0 / Demo0.1.0 package distribution | Versions retained; new implementation identified by source/config/method identities. Core source/lock unchanged. |
| Original A1, chunker1.0.0, packaged corpus/rules and bilingual bridge | Byte-identical source/resources/config retained. No algorithm or old score rewritten. |
| New strategy/query | lqd-semantic-hybrid-1.0.1 / lqd-semantic-query-windows-1.0.1; D/H config hashes differ; encoder assets separately pinned. |
| M2 formal1.1.0 / M5 A1 evaluator+suite3.0.0 | Code/contracts/thresholds unchanged; existing formal path cannot select vNext. Old results do not cover the new whole-source set. |
| Source Lock | Future lock must rebind commit, all source digests and collection digest, new model/spec/config/query/method identities, dependency lock, tested artifacts and report identity. No actual lock/approval/claim is created here. |
| Frozen independent data / Gold / consumed history | Untouched and unread. Old M2 FAIL remains consumed; M5 NOT_RUN, Final Release BLOCKED. |

Repository simplification is **deferred** under the correctness-first condition:
new quality gates did not stabilize. There is no evidence to delete historical
strategies/operators or collapse independent source/authorization checks. No broad
cleanup, new framework, weaker validation or source/Gold migration is performed.
