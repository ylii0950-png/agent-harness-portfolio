# M2R development experiment preregistration

2026-10-24. Declared **before implementing/scoring candidates**. Approved base:
f0f9297626c06e8eba83ca3a0f34b9158dfc048a (Draft PR #14, unmerged); main remains
6a55688172ba6bfac09d60e1c6982c94389e7743. Independent codex/phase3-m2r branch;
final Draft PR base must be codex/phase3-m2. No pushes until candidate stability,
full local checks and independent review; preserve global CI exactly.

## Truth, input boundary and contamination

M1 Corpus/Rules/BusinessGold/queries/anchors and M2 v1 chunks/Gold/config/reports
remain byte-identical. Existing788+216=1004 tests preserved. Only12 original
M1 dev queries are selection inputs. A mechanical dev-only copy of the12 records
from frozen M2 Gold is under tests/dev_gold/m2r-v1, with source digest and freeze;
no new answers, query rewrites or held-out set. New experiment runner accepts only
this dev schema and fixed D01–D12 set, no eval switch or sealed-Gold input.
Production ranker imports no Gold or case table. Old tests may reproduce **v1**
sealed results; no new strategy runs on exposed v1 sealed queries. Historical
sealed failures informed the PO's generic research directions and were seen by
this engineer; a blind independence claim is impossible. Do not train explicit
case/alias/ID patches. New held-out protocol is a design only, not a created set.

## Finite candidates (no adaptive additions)

All candidates keep k1=1.2,b=0.75, NFKC+casefold/[a-z0-9]+ and deduplicated query
terms. No stemming, synonym table, aliases or string-specific treatment. Modes
explicitly separate `authority` (policy/exception/conflict) from `faq_navigation`
(FAQ only); both reuse the same frozen time/category/snapshot filters. No policy
is dropped because of its recommendation or unfavorable business conclusion.

| ID | Authority scoring | Title/Heading/Body weights | Diversity lambda |
|---|---|---|---:|
| A1 | v1 flat concatenated BM25, FAQ removed | n/a |0|
| A2 | weighted sum of per-field BM25 |2/2/1|0|
| A3 | weighted sum of per-field BM25 |1/1/2|0|
| A4 | same field score as A2 |2/2/1|0.25|
| A5 | same field score as A2 |2/2/1|0.50|

Controls: unchanged v1 BM25 and v1 keyword with their original mixed candidate
pool. Five candidates ×12 queries only; no parameter sweep beyond these five.
Same dev evidence units/metric formulas and top3. Authority changes candidate/DF
population; field scoring changes statistical semantics. Results therefore have
an explicit new strategy/config identity and are not silently v1 scores.

Field formula: each field f has its own tf, df, length and average length across
the filtered candidate pool N. `score=sum_f w_f * BM25_f`; BM25_f uses exactly
v1's positive-IDF/saturation formula. This is a weighted sum, not a claim to a
standard BM25F implementation. Fields with zero average length contribute zero.
All declared weights positive; no body suppression. Empty/no-match returns none.

Diversity is deterministic greedy MMR-style ordering over positive lexical
candidates only. `gain(c,S)=base(c)/max_base - lambda*max(Jaccard(tokens(c),tokens(s)))`
for s in S (redundancy0 if S empty). Tokens are title+heading+body, same English
normalizer. Pick highest gain, tie by Chunk ID, until k. No document quota,
Gold coverage hint, forced policy insertion or catalog fill. Retain score metadata;
score is neither confidence nor an authority decision. Catalog conflict status and
both policies remain visible irrespective of rank, not automatically resolved.

## Selection/reporting rule

Order candidates by dev Macro Recall, then MRR, then Hit (descending); ties use
A1..A5 declaration order, preferring simpler earlier strategies. Select an
improvement only if Recall improves over original v1 BM25, or Recall equals it
and MRR improves. If none improves, record no recommendation; retain candidates
without claiming success. This small dev comparison has no significance claim.
No candidate selection uses v1 sealed scores. Preserve every candidate's per-query
Hit/Recall/MRR/covered/missing units/Top3 and improvements AND regressions.

Group report: authority performance over all12; exception-required and conflict
subsets derived from dev evidence metadata only, with explicit denominators.
FAQ navigation runs the same12 queries but reports only hit availability/metadata/
valid source refs, not invented FAQ relevance Recall/MRR (no FAQ Gold exists).
It is not part of selection and cannot substitute FAQ for authoritative evidence.

After comparison, freeze chosen explicit strategy/config, source-code digests,
input/Gold/plan digest and tests. Later regression reruns reproduce fixed results;
no new strategy tuning or formal v2 held-out execution. Any genuine engineering
fix must retain earlier evidence and explain recomputation, not add candidates.

## Validation/publication limits

New tests: hand arithmetic, empty/tie/error cases, metadata modes, retained
exceptions/conflicts, Gold-free diversity, generic synthetic metamorphic queries,
old bounds/path/citation/version safety and distribution Gold exclusion. Never
weaken originals. Full3.11/3.12 core+Demo, Ruff, strict mypy, both builds and clean
local-wheel installs before one grouped push and stacked Draft PR. No CI file
change, quota purchases, paid runner, paid API, credential or model access.
Remote checks may be quota-blocked: report actual status without bypass or claim.
No merge of either PR, no M3 and no newly created/scored held-out benchmark.
