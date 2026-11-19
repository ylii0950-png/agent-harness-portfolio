# Proposed held-out retrieval evaluation v2 protocol

2026-10-24. **Design only: no new held-out queries, Gold, isolation service or
formal score has been created. Requires separate Product Owner authorization.**
M2 v1 failed its quality gate. Its exposed results remain historical evidence,
not a fresh blind test of M2R. Improved development metrics cannot close that gate.

## Responsibilities and contamination

Propose a Product Owner-appointed evaluation maintainer, separate from the
retriever implementer, to author and retain the new synthetic queries and evidence
annotations. This role and access separation are **not currently established**.
The maintainer must work from the frozen policies and business contracts, without
looking at candidate rankings while authoring or labeling. An independent second
review checks policy meaning, missing complementary evidence and reachability.
The implementer must not author/review Gold or see queries before code freeze.
If the same Codex executor creates Gold, implements retrieval and sees answers,
report an openly developed evaluation; never call it an independent blind test.

The engineer has already seen v1 sealed failures and the PO's resulting generic
research directions. This background contamination cannot be undone. Disclose it
even with fresh queries. Shared fixed policies and rule families mean the new
evaluation is neither a new business domain nor a test of model generalization.

## Version and fixed inputs

Proposed namespace `lqd-heldout-retrieval-2.0.0`, using the unchanged M1 corpus,
rules, synthetic identities, clock and policy snapshots; no alteration of M1
BusinessGold or old queries. New query records are a **separate retrieval-only
artifact**, not replacements for M1 case/query truth. They must be newly written
and screened against all 12 dev and previously exposed queries for duplicates
and near paraphrases. Screening cannot prove statistical independence.

Propose exactly **18 positive queries (6 per business category)** and **6 negative
queries (2 per category)**. Positives must cover ordinary policies, exceptions,
multiple required policies, and explicit conflict snapshots, with the final
stratum counts fixed in the manifest before scoring. Do not create queries to
target a particular candidate's weaknesses or strengths. Freeze exact query
text/ID, category and snapshot (known filters, not classification predictions).
No partial dataset scoring. Negatives have their own denominator 6 and never
increase the 18-query positive ranking scores.

Versioned manifest pins corpus/index/chunking/normalizer/metric digests, query
count, split provenance, annotations, checksums, reviewer sign-off and budgets.
Keep new Gold outside public resources, wheels/sdists and model workspaces.
No new artifact is actually frozen by this proposal.

## Evidence and top-3 feasibility

For each positive q, freeze a nonempty set U_q of semantically distinct evidence
units, each anchored to unchanged authoritative source spans. One unit has OR
alternatives of AND bundles of stable Chunk IDs. Equivalent evidence belongs in
alternatives, not extra denominator units. Complementary policy requirements
remain separate units (or an AND bundle where one semantic unit spans chunks).
FAQ navigation never substitutes for policy authority. Conflict evidence must
cover both conflicting obligations; score cannot choose a winning policy.

Before ranking, mechanically verify every ID, exact span, active/snapshot scope,
alternative equivalence and bundle completeness. Independently compute the
smallest union of one alternative bundle per unit; require its size <=3 for
every positive. Reject unreachable queries before the immutable release, without
observing retrieval rankings. This is only a proposed gate, not a present proof
for an uncreated set. Do not hide inconvenient policies to obtain reachability.

## Fixed arithmetic

Let R_q be the first 3 distinct returned chunks, in order; fewer hits stay fewer.
Let G_q be the union of chunks in all annotated bundles. A unit is covered iff
at least one of its alternative bundles is wholly contained in R_q.

- Hit@3(q) = 1 if R_q intersects G_q, else 0.
- Recall@3(q) = number of covered units / |U_q|. This is **semantic-unit recall**;
  alternatives do not inflate the denominator. A partial AND bundle earns no unit.
- MRR@3(q) = 1/r for the earliest rank r<=3 in G_q; no hit gives 0. A relevant
  fragment can give MRR credit without completing an AND bundle; disclose this.
- Retrieval required-anchor Top3 coverage(q) = covered required units / total required units,
  where the required subset is fixed before scoring. It can differ from ranking
  recall if optional relevant units exist. Report full-coverage case count too.
  If no optional units exist, explicitly state the two coverage values coincide.
  This retrieval-only diagnostic is **not** the original M4/M5 runtime Required
  Evidence Coverage (REC), which requires actual evidence acquisition/verification
  by the Agent. Search hits alone do not establish runtime REC or task completion.

Main Hit, Recall and MRR are macro means over **all 18 positives**, including
errors/no hits as zeros. The inner Recall denominator is each query's fixed unit
count, not the number returned or union of equivalent chunks. Required coverage
uses all 18 required-evidence positive queries; each must have a nonempty required
subset. Report per-query units/bundles, hits, missing units and ranks. Also report
raw chunk recall only as a separate diagnostic, never the acceptance Recall.
The retrieval coverage report label must retain this distinction. Negative behavior
(no fabricated evidence, no inappropriate authority claims)
has denominator 6 and separate criteria; not a retrieval confidence calibration.

Proposed gates remain **Recall@3>=0.90, MRR@3>=0.80**, subject to PO acceptance of
the actual frozen dataset and reachability proof. Unit-normalized recall and
minimum-cover<=3 make a perfect structural score achievable, but do not guarantee
a lexical system can reach it. Do not relax annotations or thresholds after runs.

## Controls and comparability

Run unchanged v1 BM25 and v1 keyword in their original mixed-policy/FAQ mode as
explicit historical-semantics controls on the **same new queries**. Also report
authority-only flat BM25 (A1) and authority-only keyword as matched candidate-pool
controls. These are labeled new modes, not silent replacements of v1 baselines.
Use identical known category/time/snapshot inputs, frozen chunks and top3 for all.
Only the preselected locked M2R strategy is the primary new candidate. FAQ-only
navigation is descriptive and outside the authority acceptance metric.

The original 18-query v1 benchmark and new v2 queries have different labels and
distributions. Do not compare their scalar values as a controlled improvement
claim. Within v2, report that authority filtering changes both candidate pool and
BM25 DF/length statistics; weighting/diversity may change rankings further.

## Freeze, one run, preservation and approvals

1. PO appoints steward and approves creation protocol, counts/strata and gates.
2. Implementer freezes source, tests, strategy, parameters, input contracts and
   development report digests **before seeing new queries/labels**. No tuning.
3. Steward creates/reviews and seals the full new set; records access history and
   top3 proofs. Formal evaluation cannot start until all freeze checks pass.
4. PO separately approves release/execution of exactly one offline formal score
   run per frozen method. No model/OAuth/API is needed. No CI auto-evaluation.
5. Preserve complete success/failure/error outputs and run/input/code hashes.
   A failed metric is a failed gate. No selective reruns, Gold patching, answer
   changes or best-of-many reporting. A genuine implementation defect requires
   retaining original results and a new authorized version; previously seen data
   cannot become blind again.

This M2R task authorizes none of steps1–4. No new held-out creation or execution
has occurred. Before another task, the PO must approve stewardship, dataset
creation/sealing and the actual frozen execution protocol. M3 remains unauthorized.
