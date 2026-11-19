# Retrieval relevance and evidence remediation — public engineering

The prospective R1 implementation improves ranking and Chinese support on the
new DEV material but **fails the preregistered public quality gates**. It is not
an acceptance-ready retriever or a general answer-sufficiency verifier. A1 stays
the default. Original A1/D/H/S2.1, all prior results and formal histories remain.
No new independent/formal readiness or release authority follows from this work.

The registered protocol and public evidence locks
(`verification/relevance-evidence-v1/data-lock.json` and `selection.json`) precede
candidate comparison/Challenge use. The separate public Challenge author did not
select candidates or read candidate code. Its provenance explicitly records
high-level context exposure; this is not a formal blind test. Source-only corpus
construction and a separate scorer keep public Oracle units out of Runtime.

## Actual implementation

- `retrieval_relevance.py`: versioned R0/R1 with independent Dense/Lexical union,
  64-candidate generation and 24-source shortlist. Whole-query relevance anchors
  the first result; facets can select a shared source or add a complementary one.
  No title stripping, topic projection, early sign-null rejection or mandatory
  section quota. All request clauses remain in the plan or capacity fails
  explicitly; conditions/negations are not discarded by first/last selection.
- `pair_model.py`: one pinned bilingual BAAI/bge-reranker-base quantized ONNX
  conversion, fixed revision and every asset hash/size/license. CPU only, existing
  optional ORT/tokenizers dependencies, bounded token windows/cache, explicit
  unavailable/hash/runtime failure. Inference never downloads or uses custom
  remote code. All query/passage tokens are windowed; no silent middle truncation.
- `relevance_evidence.py`: full canonical citation/authority checks before
  source-content ablation and unavailable-source contrast. This is an experimental
  relevance screen, **not entailment**. Same-subject/modal/predicate/condition
  opposition has a deterministic veto. General paraphrase, implicit numerical
  inconsistency, arbitrary exceptions and natural-language contradiction remain
  unverified. Different conditions are not declared contradictory merely because
  their modal verbs oppose each other.
- Agent/Controller and read-only answer assessment add a new-profile-only
  canonical-source conflict veto. Business facts still come exclusively from
  the existing trusted finite rules; documentary requirements, actual kb_search
  then kb_get, exact ASK, atomic commit, journal and actual readback remain.
  Model relevance/contrast cannot issue proposals or authorize writes.

Two actual integration defects were fixed without weakening existing contracts:
new method identity exceeded the M2G 256-character bound, and raw negative logits
violated its positive-score transport contract. The final identity hashes all
components compactly; SearchHit uses a positive monotonic transport transform,
with raw scores retained in traces. The unchanged evidence adapter does not
interpret scores as truth. Native workflow testing also exposed commit-before-
budget-exhaustion/readback failure. Prospective Mock navigation now batches at
most three independent read searches and fetches only pre-existing trusted M3
prerequisite sections. The original 12/28 limits, default A1 navigation and
formal execution protocol are unchanged. Final native bilingual flows use at
most nine model rounds and verify readback after every successful commit.

## Results and limitations

Machine-readable complete comparisons, gates, raw report hashes and engineering
verification are under `verification/relevance-evidence-v1/`. DEV and Challenge
metrics are kept separate. Original raw files/logs and initial adverse attempts
are retained locally; lossless public reports are included as `.json.gz`.

On DEV, R1 Recall/MRR is .9375/.979167, versus S2.1 .78125/.666667. Chinese
Recall/MRR is .96875/.9375 and early positive rejections are zero. Candidate and
shortlist Recall are 1.0. Single-evidence completeness is 1.0. These gains do not
cancel English Recall .875 (best baseline 1.0), multi completeness .75, negative
relevance-screen false acceptance 17/24 or partial-source failures 3/3. All six
profiles use the same new screen for the evidence comparison. Raw negative search
hits are separately reported; R1 returns related/irrelevant candidates for all
24 negatives rather than pretending relevance ranking is answerability.

The omitted DEV evidence units are not lost at candidate generation or
shortlisting: their sources remain available to ranking. Whole-query anchoring
helps first-result precision but still overweights one need in some multi-need
requests. The source-ablation experiment proves contribution to relevance, not
that every requested fact is present. Its partial-source failures and negative
false accepts are actual counterexamples. No fixed-logit threshold, extra model,
new candidate, changed denominator or relaxed gate is introduced after this
comparison. The frozen decision selects **no qualified profile**.

## Reproduction and integration

Use built, non-editable Core/Demo wheels in explicit Python 3.11/3.12 environments.
Default tests need no semantic extra, network or credentials. Native checks are
explicit `tests.native_relevance_check` with separately prepared, verified local
E5 and pair-model directories. `python -m knowledge_ticket_agent.pair_model DIR
--prepare` is the explicit public-asset preparation operation; it is disabled in
CI. The [upstream card](https://huggingface.co/BAAI/bge-reranker-base),
[conversion card](https://huggingface.co/Xenova/bge-reranker-base) and pinned MIT
license are recorded and retained beside assets; weights are not in Git.

The installed offline business CLI accepts `--retrieval-profile vnext
--retrieval-variant R1 --model-directory E5_DIR --pair-model-directory PAIR_DIR`.
These are explicit experimental options. Native inference is local; business
decisions use deterministic Mock/Synthetic Responses transport, not a live model.
Both Python versions run full Core/Demo suites, native bilingual Agent/Controller,
invalid/missing/refused approvals, compaction, public business and fault controls,
Ruff, strict mypy, builds/payload parity and clean installation. Missing Docker
runtime remains a truthful environment skip.

## Source identity and acceptance boundary

Runtime collection: 114 modules, SHA256
`def4a33dd28805c8a8510cfe90b47ebf49035d48d3ecaa644ad4d9bf7d3bc764`.
R1 `lqd-relevance-retrieval-3.0.1`, config
`b5aa9dc1d3b99e8598d47ba9906a6d31245f4027cc0a7a6508e8b6610968d5da`;
query plan `lqd-lossless-request-plan-1.0.0`; evidence screen
`lqd-source-contrast-evidence-1.0.0`; pair model identity
`df1026556bb1f4aeb9e50bf9638a0922e7b39a095ef782ef03ffa11fa6109171`.
Original E5 identity and S2.1 config remain unchanged. No dependency lock changes.

New future Source Lock must bind the eventual exact commit, installed/committed
source digests and their aggregate, new method/config/model-asset identities,
public test/helper/data versions and exact CI evidence. `FormalRelease.commit`,
`sources` and consequently `identity`, and `ReleaseLock.commit`, `source_digests`,
`method` and consequently `identity`, cannot reuse old source identities. Frozen
M5 A1 3.0.0 code/contract semantics are retained, but shared source identity has
changed and independent compatibility confirmation is still necessary. This
record does not create or amend a formal release/approval/claim/root/ledger.

Consumed independent M2 FAIL is permanent; M5 Formal NOT_RUN; Final Release is
BLOCKED. No private Gold, hidden questions or individual receipts were accessed.
No live/paid model call, formal scoring, authority reset or Repository Reduction.

## Fixed six-profile comparison

Each split has 48 positive requests, 24 negatives; evidence controls 15. Macro
Recall/MRR are positive-only; completeness counts all required units. Evidence
false acceptance uses the common experimental screen, not business authorization.

| Split | Profile | Candidate Recall | Recall@3 | MRR@3 | Complete | Single complete | Multi complete | Raw negative hits | Evidence FA | Evidence FR | Controls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | A1 | 0.666667 | 0.666667 | 0.545139 | 0.666667 | 0.666667 | 0.666667 | 14/24 | 12/24 | 16/48 | 12/15 |
| DEV | D | 1.000000 | 0.843750 | 0.826389 | 0.791667 | 0.875000 | 0.708333 | 24/24 | 13/24 | 1/48 | 12/15 |
| DEV | H | 1.000000 | 0.885417 | 0.802083 | 0.854167 | 0.916667 | 0.791667 | 24/24 | 13/24 | 1/48 | 12/15 |
| DEV | S | 1.000000 | 0.781250 | 0.666667 | 0.729167 | 0.833333 | 0.625000 | 18/24 | 13/24 | 8/48 | 12/15 |
| DEV | R0 | 1.000000 | 0.812500 | 0.871528 | 0.687500 | 0.958333 | 0.416667 | 24/24 | 13/24 | 0/48 | 12/15 |
| DEV | R1 | 1.000000 | 0.937500 | 0.979167 | 0.875000 | 1.000000 | 0.750000 | 24/24 | 17/24 | 0/48 | 12/15 |
| challenge | A1 | 0.770833 | 0.708333 | 0.718750 | 0.645833 | 0.750000 | 0.541667 | 16/24 | 5/24 | 12/48 | 7/15 |
| challenge | D | 0.989583 | 0.895833 | 0.902778 | 0.812500 | 1.000000 | 0.625000 | 24/24 | 7/24 | 2/48 | 7/15 |
| challenge | H | 0.989583 | 0.906250 | 0.920139 | 0.833333 | 1.000000 | 0.666667 | 24/24 | 8/24 | 2/48 | 7/15 |
| challenge | S | 1.000000 | 0.593750 | 0.590278 | 0.541667 | 0.666667 | 0.416667 | 12/24 | 1/24 | 14/48 | 7/15 |
| challenge | R0 | 1.000000 | 0.906250 | 0.947917 | 0.833333 | 0.958333 | 0.708333 | 24/24 | 7/24 | 2/48 | 7/15 |
| challenge | R1 | 1.000000 | 0.968750 | 1.000000 | 0.937500 | 1.000000 | 0.875000 | 24/24 | 9/24 | 1/48 | 7/15 |

| Split/profile | Language | Recall@3 | MRR@3 | Complete | Evidence FR |
|---|---|---:|---:|---:|---:|
| DEV/A1 | en (16) | 1.000000 | 0.968750 | 1.000000 | 0/16 |
| DEV/A1 | zh (16) | 0.000000 | 0.000000 | 0.000000 | 16/16 |
| DEV/A1 | mixed (16) | 1.000000 | 0.666667 | 1.000000 | 0/16 |
| DEV/D | en (16) | 0.968750 | 1.000000 | 0.937500 | 0/16 |
| DEV/D | zh (16) | 0.593750 | 0.510417 | 0.500000 | 1/16 |
| DEV/D | mixed (16) | 0.968750 | 0.968750 | 0.937500 | 0/16 |
| DEV/H | en (16) | 1.000000 | 1.000000 | 1.000000 | 0/16 |
| DEV/H | zh (16) | 0.656250 | 0.531250 | 0.562500 | 1/16 |
| DEV/H | mixed (16) | 1.000000 | 0.875000 | 1.000000 | 0/16 |
| DEV/S | en (16) | 1.000000 | 1.000000 | 1.000000 | 0/16 |
| DEV/S | zh (16) | 0.468750 | 0.375000 | 0.437500 | 8/16 |
| DEV/S | mixed (16) | 0.875000 | 0.625000 | 0.750000 | 0/16 |
| DEV/R0 | en (16) | 0.750000 | 1.000000 | 0.500000 | 0/16 |
| DEV/R0 | zh (16) | 0.718750 | 0.645833 | 0.625000 | 0/16 |
| DEV/R0 | mixed (16) | 0.968750 | 0.968750 | 0.937500 | 0/16 |
| DEV/R1 | en (16) | 0.875000 | 1.000000 | 0.750000 | 0/16 |
| DEV/R1 | zh (16) | 0.968750 | 0.937500 | 0.937500 | 0/16 |
| DEV/R1 | mixed (16) | 0.968750 | 1.000000 | 0.937500 | 0/16 |
| challenge/A1 | en (16) | 0.875000 | 0.875000 | 0.750000 | 1/16 |
| challenge/A1 | zh (16) | 0.250000 | 0.281250 | 0.187500 | 11/16 |
| challenge/A1 | mixed (16) | 1.000000 | 1.000000 | 1.000000 | 0/16 |
| challenge/D | en (16) | 0.968750 | 0.906250 | 0.937500 | 0/16 |
| challenge/D | zh (16) | 0.750000 | 0.833333 | 0.562500 | 2/16 |
| challenge/D | mixed (16) | 0.968750 | 0.968750 | 0.937500 | 0/16 |
| challenge/H | en (16) | 0.968750 | 0.937500 | 0.937500 | 0/16 |
| challenge/H | zh (16) | 0.750000 | 0.822917 | 0.562500 | 2/16 |
| challenge/H | mixed (16) | 1.000000 | 1.000000 | 1.000000 | 0/16 |
| challenge/S | en (16) | 0.812500 | 0.833333 | 0.687500 | 0/16 |
| challenge/S | zh (16) | 0.062500 | 0.062500 | 0.062500 | 13/16 |
| challenge/S | mixed (16) | 0.906250 | 0.875000 | 0.875000 | 1/16 |
| challenge/R0 | en (16) | 1.000000 | 0.968750 | 1.000000 | 0/16 |
| challenge/R0 | zh (16) | 0.812500 | 0.937500 | 0.625000 | 1/16 |
| challenge/R0 | mixed (16) | 0.906250 | 0.937500 | 0.875000 | 1/16 |
| challenge/R1 | en (16) | 0.968750 | 1.000000 | 0.937500 | 0/16 |
| challenge/R1 | zh (16) | 0.937500 | 1.000000 | 0.875000 | 1/16 |
| challenge/R1 | mixed (16) | 1.000000 | 1.000000 | 1.000000 | 0/16 |

Challenge R1 evidence controls: complete 3/3, partial 0/3, missing 1/3,
contradiction 0/3, different-condition nonconflict 3/3. The literal directive
parser treats the leading condition/context as part of its proposition key;
“Separately, during ...” and “During the same ...” produce different keys. Thus
even these explicitly linked audit instructions escape its veto. This is a
public counterexample, not a hidden formal diagnosis. No runtime source/config/model/
scoring changes follow this Challenge. No second execution is performed.

R1 DEV/Challenge startup 1.808/1.723s; p95 search+diagnostics+verification
5.817/7.896s; peak RSS 1.759/1.769 billion bytes (single CPU thread per model).
Wall-time includes source validation and diagnostic trace generation; native
validation/full suites overlapped portions of the comparison, so these are
observed local costs, not isolated benchmark claims. Warm deterministic repeat
checks used caches and were not separately timed. Supplemental dual-Python
native probes measured parent CPU user 3.60–3.68s and system .44–.57s for model
load, two unsupported flows and two scoped searches; CLI child CPU excluded.
The pair asset set is about 296MB plus existing E5 assets, outside Git. No new
mandatory dependency, optional runtime version or dependency lock change.

Final local Python3.11/3.12 each: Core848 passed/1 Docker skip, Demo955 passed,
focused113 passed, Ruff/format/strict mypy/build/wheel-sdist payload parity/clean
install PASS. Native per interpreter:6 supported multilingual flows,2 unsupported
refusals,4 Controller approval/refusal/invalid/missing controls,2 compaction
flows,13/13 public business cases,10/10 fault controls, long pair token windows
and installed optional R1 Chinese CLI/readback PASS. All searches in each
actual flow share the same compact method identity. Model decisions are Mock;
no live model reasoning or independence claim is made. Unauthorized effects0
is supported by these finite business/security controls, not inferred from
retrieval metrics.
