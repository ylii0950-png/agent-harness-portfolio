# Finite policy evidence contract 2.0.0

PUBLIC ENGINEERING ONLY. Replaces relevance contrast and literal modal matching;
R1/E5/BGE ranking and assets remain the fixed baseline. No new semantic model,
formal protocol, private Gold, claim, approval or ledger operation.

The supported product domain is explicit source-bound policy assertions, not
unrestricted natural-language entailment. An authoritative policy publisher
supplies independent query-free context/schema/term metadata and normative
claims in the canonical policy body. Unknown syntax, condition or reference
cannot yield SUPPORTED. Existing frozen M3 policies remain immutable.

Source line grammar (strict JSON, dataclasses in `policy_assertions.py`):

- `@context`: name + Scope(subject/action/object/scenario/time/conditions).
- `@schema`: context, semantic family name, noun labels, AND/OR fact requirements.
  Requirements contain keys/operators/modalities/functional uniqueness, never answer values.
- `@term`: canonical entity or role + equivalent names. Definitions have spans.
- `@claim`: context reference, key, operator, typed value/unit, modality.
- `@note`: explicitly non-normative background; normative unparsed text is Unknown.

Scope fields and text-valued conditions are publisher-defined canonical identifiers;
equivalent surface forms require source-owned alias declarations. This is a finite
symbolic policy model, not automatic natural-language ontology or NLI extraction.
Independent required fields/roles must be declared as distinct fact keys.

Each claim binds document/index/quote hashes, exact canonical character spans,
authority, as_of, snapshot and context/alias definition spans. Only actual full
kb_get observations supply claims. Context/schema metadata is canonical source
metadata independent of questions; unseen claim values cannot discharge a proof.
`lqd-obligation-source-selection-1.0.0` uses only source-owned fact-key locations
and compatible scopes to plan full reads. Metadata/background sections cannot
displace the requested fact sections; R1 orders eligible locations and metadata
can supplement its shortlist. All possibly overlapping counterclaims are read
dependencies. The new source-selection identity is distinct from unchanged R1.
Neither a location nor a selected source establishes a fact.
Unsupported Boolean grouping and impossible source contexts are Unknown.

Queries select noun labels or finite startup/completion/review requests with
source-defined entity aliases. English, Chinese and mixed requests share the
same obligations. Query request-span offsets use the NFKC/casefold/whitespace-normalized
request; the original query remains bound to the plan. Source spans always use exact
canonical document character offsets. All unparsed qualifiers, numbers, negation, roles and conditions
remain Unknown. Unsupported natural-language questions are NOT_ASSESSABLE;
this contract does not pretend to understand arbitrary prose. Typed Source Claim
conditions preserve units, exact decimal comparisons and applicability.

Obligations have AND bundles/OR alternatives; the query has explicit Boolean
branches. A source may discharge several requirements; several sources can
jointly discharge one. Unknown is never an OR shortcut. Opposite modalities
conflict only in overlapping normalized scopes. Different conditions do not
conflict; permissions do not become mandatory facts. Deictic references require
a unique canonical same-document antecedent; ambiguity is Unknown. Supersession,
expiry, snapshot/category and FAQ authority filtering precede proof.

SUPPORTED requires complete applicable fact bindings and no Unknown/conflict.
INSUFFICIENT lists absent fact keys. NOT_ASSESSABLE covers unresolved conflicts,
unknown semantics/conditions and ambiguous references. No verdict grants WRITE.
M3 documentary projection additionally requires the original exact source/rule
lock; its hashes prove specified source presence, not arbitrary query entailment.
Customer/invoice/identity facts and exact ASK/atomic/journal/readback remain with
original business rules. Both application paths expose the same evidence method
identity alongside the unchanged R1 retrieval identity.

## Fixed public validation plan

Deterministic functional/metamorphic/actual-tool tests precede semantic runs.
New DEV:24 positives (8/language;12 single/12 multi),12 negatives (4/language),
10 evidence controls (2 each complete/partial/missing/conflict/disjoint).
No prior exposed Challenge selects this implementation. Use only new source
assertions and source-only corpus construction; Oracle stays in the scorer.
Reuse the existing public scorer for R1 only; do not create another ranker sweep
or duplicate large per-case traces. Diagnose metadata-only source selection as a
separate source-type concern if observed; keep R1 scoring/model bytes unchanged.

Gates fixed before DEV: macro Recall@3 >=.90, MRR@3 >=.80; each language >=.90/.80;
single complete >=.95, multi complete >=.90; evidence FA0/12, positive FR<=1/24;
all10 evidence controls; citation/authority integrity100%, unauthorized effects0;
source+verification p95<=15s, startup<=10s, RSS<=3GiB. Existing formal thresholds
and results do not change. Every gate is conjunctive, not a weighted average.

After functional and DEV gates pass, freeze code/config/selection/scorer and use
one new public Challenge from an author not involved in debugging. Same sizes
and gates; once only, no adaptive correction. Preserve every failure. Full dual
Python Core/Demo/native/security/lint/type/build/clean checks precede one grouped
PR39 push and normal CI, with $0 overage hard stop. Until DEV passes: no push/CI.
No independent formal acceptance or Final Release claim follows this protocol.

## Locked public outcome: FAIL (2026-11-02)

The rewrite is implemented; stable product-quality delivery is **not achieved**.
The new independently authored public Challenge was evaluated once after source,
selection, data and scorer locks. No runtime/config/scorer changes followed its
exposure. This is public engineering, not independent formal blind acceptance.
The author's exposure/provenance attestation is in `challenge-lock.json`.

| Split | Candidate Recall | Recall@3 | MRR@3 | Single complete | Multi complete | Evidence FA | Evidence FR | Controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | 1 | 1 | 1 | 12/12 | 12/12 | 0/12 | 0/24 | 10/10 |
| Challenge | 1 | .687500 | .722222 | 8/12 | 8/12 | 0/12 | 24/24 | 3/10 |

| Split/language | Recall@3 | MRR@3 | Complete | Evidence FR |
|---|---:|---:|---:|---:|
| DEV/en, zh, mixed (each8) | 1 | 1 | 8/8 | 0/8 |
| Challenge/en (8) | .062500 | .166667 | 0/8 | 8/8 |
| Challenge/zh (8) | 1 | 1 | 8/8 | 8/8 |
| Challenge/mixed (8) | 1 | 1 | 8/8 | 8/8 |

Raw unchanged R1 on these new structured corpora: DEV Candidate Recall .944444,
Recall/MRR .652778/.548611; Challenge .729167, .062500/.138889. Both adverse raw
results are retained. The new source view improves factual-read coverage only
inside its finite parsed domain; this is not general R1 promotion or proof that
old plain-language quality is retained on new distributions. No exposed old
Challenge was used to select or retune the new implementation.

Actual gaps, diagnosed without changing this Challenge or using its Oracle at
runtime:

- The finite English request grammar misses source entity + bare family labels;
  source selection then falls back to R1 and metadata sections displace facts.
- JSON declaration shape is valid but runtime quantity vocabulary is too narrow.
  Publisher units/counts outside its finite registry become `invalid_claim`/
  Unknown. Those unsupported required facts block otherwise retrieved sources.
  The published source contract did not adequately expose that vocabulary limit.
- Full-source reads and correct citation hashes do not establish answerability:
  all24 Challenge positives are NOT_ASSESSABLE. Zero negative acceptance is not
  sufficient quality. The 3/10 controls are status matches; one conflict control
  matches NOT_ASSESSABLE because of unrelated Unknown rather than a detected
  conflict. Only one of the two explicit conflict controls records a conflict.
- CPU-only E5/BGE on the larger source declarations misses the fixed latency gate:
  DEV startup1.877s/p95 8.672s/RSS1770045440 bytes; Challenge
  startup1.736s/p95 33.489s/RSS2036973568 bytes, one configured CPU thread.
  Total process CPU for the whole quality run was not captured; no CPU total is
  inferred from wall time. The unchanged ranker still scores metadata passages.

All canonical retrieval citations pass revalidation; actual read-only/domain and
business safety checks record no unauthorized effects. These are integrity and
safety observations, not a substitute for the failed relevance/sufficiency gates.
No thresholds, denominators, source assertions, Oracle or runtime were adjusted
after the Challenge. No further model comparison, third model, push or CI follows
this failed quality gate in this work package.

Fresh installed checks on frozen source116 (both Python3.11.17/3.12.15):
Core848 passed +1 unavailable-Docker skip; Demo994 passed; focused152;
Ruff/format/strict mypy; Core/Demo build, wheel/sdist/source parity and clean
install/CLI PASS. A later test-only compaction extension adds a case: all40 proof
cases pass per interpreter, including actual compressed tool history and fresh
run isolation; it changes no runtime/scorer code. The earlier smaller fixture
context cap stopped before proof with context_error; its assertion logs remain.
Native E5+BGE each:6 bilingual approval/refusal flows,4 Controller controls,
2 compaction flows,13 business cases,10 security/fault controls,3 real QA tool
flows,2 unsupported request refusals, long windows and installed Chinese R1
CLI/readback PASS. Providers are Mock/synthetic transport, not live reasoning.

New product modules: `evidence_contract.py`, `policy_assertions.py`. Replaced
experimental `relevance_evidence.py` contrast/modal logic. Reused Agent tools,
Controller, business receipts, read-only answer adapter and existing public
scorer/corpus/native helpers. No Core, model/ranking, dependency lock, packaged
policy/rules, old strategy, formal evaluator, historical report or Gold changes.
All46 pre-existing public sync copies retain identical SHA-256.

Runtime source116 SHA256:
`d73bf1c8a7582fd1d167e4710de91f5b96a56632c42aa238a25e21f599b010ef`.
R1 version3.0.1 and config
`b5aa9dc1d3b99e8598d47ba9906a6d31245f4027cc0a7a6508e8b6610968d5da`
remain unchanged. Evidence2.0.0 and source-selection1.0.0 are new identities;
public scorer2.0.0 and public DEV/Challenge2.0.0 are separately versioned.
`selection-lock.json` records the pre-Challenge candidate/scorer/DEV identity;
`challenge-lock.json` records new data provenance/hashes. Raw evidence stays in
task-local `artifacts/evidence-contract-20261102`; no large per-case report is
copied into another framework or rewritten to PASS.

Source Lock `commit`, `sources` and derived release identity require future
rebinding. Original frozen A1/M5 execution semantics are retained; independent
custody must separately confirm source/contract compatibility. No formal lock,
approval, claim or protocol was created or modified. Existing consumed M2 FAIL
and M5 NOT_RUN remain. Current readiness: **BLOCKED** by the explicit quality
gaps above, future independent contract/source confirmation, formal gates and
final independent review. No new independent acceptance authorization is sought.

- dev-r1-raw-valid.json SHA256 `078135dd1a1fef1a9b2c722a2ad916bba1abab93b1bc48f26298fbbf7fd7a3eb`.
- dev-obligation-selection.json SHA256 `150aeeb7377aa0f1de5eb014a2edacc54c4bf0ba4e791d538b30831385a58f16`.
- challenge-once.json SHA256 `f7da3b28cea8e8a14994fd26861e75117d8fa1b6adf5b7c5ba2819c7e8b002c7`.
