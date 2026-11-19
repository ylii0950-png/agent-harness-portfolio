# Evidence correctness recovery — 2026-11-03

PUBLIC ENGINEERING / local only. Baseline `983ceb7`; runtime candidate `5214742`,
installed verification commit `ddfda91`. No new locked Challenge or formal scoring.
Historical failures and their source identities remain unchanged.

## Actual failures and replacement

- Request extraction mixed three English request families with full-phrase matching.
  `policy_query.py` now compiles source-owned entity/concept aliases into the same
  typed obligations for English, Chinese and mixed requests. AND/OR precedence,
  predicates, roles and explicit negation survive planning. Unknown text blocks proof.
- Source compilation rejected unknown unit names and conflated a request for a
  rule with literal equality. Exact Decimal quantities now distinguish known unit
  conversions, publisher-bound aliases and incomparable nominal units. Claim
  scope, operator, modality, role and canonical definition/source spans are retained.
- Binding previously needed one claim to prove each constraint and treated opposite
  modalities too broadly. Joint bounds, conditional coverage without gaps, requested
  domain anchors and scoped contradiction partitions replace those shortcuts.
  Bindings retain supporting qualifiers; unused OR facts do not become mandatory reads.
  Missing facts give INSUFFICIENT; unresolved conditions/conflicts give NOT_ASSESSABLE.

The actual proof path uses Evidence **3.0.0**, Source Selection **2.0.0**, Assertion
Search View **1.0.0**, and the existing R1 **3.0.1** with pinned E5/BGE assets.
No new model/dependency. `policy_query.py` replaces the old parser in the existing
engine; it is not a parallel verifier. Full current-run kb_get reads remain required.
Cache entries cannot supply read receipts. Frozen M3 facts/rules remain separately
locked: receipt basis distinguishes `source_assertion` from `frozen_m3_documentary`.
Neither proof, scores nor metadata grant WRITE or establish customer/invoice facts.

## Measured performance and DEV quality

Same two allowed DEV requests, before/after profiling (seconds; not a rerun of the
historical 33.49s Challenge). Short source-derived passages remove repeated metadata
encoding; source catalog/claim parsing uses bounded per-retriever memoization.

| Stage | Before EN / ZH | After EN / ZH |
|---|---:|---:|
| Query parsing | .0014 / .0003 | .0088 / .0013 |
| Source catalog | .0776 / .0769 | .0766 / .0770 |
| E5 generation | 1.882 / .0233 | .0357 / .0452 |
| BGE reranking | 6.306 / 8.077 | .1163 / .4668 |
| Claim compilation | .0035 / .0047 | .0030 / .0048 |
| Evidence verification | .0797 / .0862 | .0046 / .0104 |

Both existing DEV2 and new developer-authored **development** DEV3 have 24
positives (8/language; 12 single/12 multi), 12 negatives and 10 evidence controls.
Final candidate: Candidate Recall, Recall@3, MRR@3 and full coverage **1.0** in
both sets/all language groups; positive false rejects **0/24**, negative false
accepts **0/12**, semantic controls **10/10** each. Conflict controls require an
actual conflict, not an unrelated Unknown matching the expected status.

Native Python3.12 final p95 query+proof: **.5543s / .6397s**; startup **2.415s / 2.195s**;
peak RSS **1,699,004,416 / 1,747,959,808 bytes**; one inference thread.
Full evaluator CPU user/system: **8.902/.594s / 9.324/.408s** (includes startup,
determinism checks and controls). Limits were fixed before native evaluation:
p95<=15s, startup<=10s, RSS<=3GiB, per-language Recall>=.90/MRR>=.80,
zero false accepts and <=1 false reject. All DEV gates pass.

These are metadata-constrained source-view metrics, not raw unrestricted R1
retrieval scores. Raw R1 default algorithms/config/assets are unchanged; the
prospective view and source-selection identities explicitly record changed inputs.
DEV3 source fixtures and scoring labels are separate; Oracle never enters runtime.

## Verification, identity and remaining gate

Exact installed Python **3.11.17 / 3.12.15** each: focused **169**, Core **848 passed,
1 unavailable-Docker skip**, Demo **1011 passed**. Ruff/format, strict mypy,
Core/Demo wheel/sdist source payload parity and clean installs/CLI all pass.
Each native interpreter also passes 6 bilingual approval/refusal flows, 4 Controller
controls, 2 compaction flows, 13 business cases, 10 fault/security checks, 3 full-read
source-assertion Agent flows, 2 unsupported refusals and installed R1 CLI/readback.
Unauthorized effects: **0**. Actual Controller samples .255–.352s; fresh-source QA
Agent samples .797–.956s. These small integration samples are not a p95 estimate.

117-module Source SHA-256:
`eb5c5220fd067ef2b2800becd02c456f365e8456fa7005e096b2afd04f3abc54`.
DEV3 input SHA-256:
`32d1fbd06ed8c5c9f3fa501cd78b9574eb99b700a6a6207c314dfe1b3636398c`.
Scorer is prospective `public-relevance-evidence-scorer-3.0.0`; candidate/shortlist
recall now excludes read-plan supplements, and semantic controls require real proof.
Raw logs/identities remain in
`artifacts/evidence-contract-20261102/recovery-20261103/`.

Core, A1, policy resources, Gold, dependency/config files, formal evaluator/protocol
and consumed histories are unchanged. All 46 local sync copies retain their hashes.
Shared source bytes and module inventory did change: any future release must bind
its actual commit, 117-module source hash, Evidence/Selection/View identities and
prospective public scorer/helper hashes. Existing M2/M5 locks cannot stand in for
this candidate; the independent custodian must separately confirm compatibility
and rebind future release identity. No existing formal lock or authorization changed.

The finite contract requires publisher-owned schemas/assertions and vocabulary.
It does not infer arbitrary natural-language facts or undeclared synonym/unit
relationships. Unknowns and bounded proof overflow fail closed. DEV success does
not establish unseen generalization. A new independently authored locked public
Challenge is **NOT_RUN** under this directive; independent acceptance preparation
therefore remains unproven. No push, Actions, paid request or Formal M2/M5 run.
