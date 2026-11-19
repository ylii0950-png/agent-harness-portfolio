# Phase 3 M2F — one controlled formal offline evaluation

2026-10-25, Asia/Kuala_Lumpur. **M2 Quality Acceptance: FAIL**: positive retrieval
thresholds PASS, but the preregistered negative behavior assertions fail4/6 for
fixed A1. Engineering completion and a COMPLETED ledger are distinct from quality
acceptance. No automatic merge, strategy reselection or M3 follows.

## Identity and scope

Original repository/origin verified, not a clone/new worktree. M2F branches from
PR17 fb53d115aef25e08e6b9f4cf1e0cbc0e0326a602 on codex/phase3-m2f. Formal code
HEAD9b9411bafa0330c96dd2bcd54c0a20578c7e58b7; later delivery commits add public
results/docs only. Original bf89220 history, main ce5df29 and PR14/15/17 preserved.
School connection and CLI ylii0950-png; Private and eight strict app-bound main
checks/admin enforcement/no force-push/delete reverified. $0 hard stop retained;
no purchases or paid runners. Charged/remaining Actions minutes remain unknown.

Three new optional Demo modules only: exact-fraction scorer, owner-only persistent
ledger and trusted local Operator. No core production/security, A1, v1 ranker,
Gold/corpus/chunker/config, thresholds, CI or dependency changes. All283 protected
PR17 files byte-identical, excluding five explicitly updated top-level authority/
progress documents. Original192-input exact CI compatibility verifier still passes.
The original fixed protocol and independent review are retained with the archived evaluation record.

## Actual execution gates and retained diagnostics

macOS arm64 installed-package Python3.11.17 and3.12.15 each: **788 Core +453 Demo
=1241 passed**,0 failures/skips. All1203 originals retained;38 public synthetic
regressions. Ruff lint/format, strict mypy Core81/Demo45, both packages' sdist/wheel
and fresh --no-index/--no-deps two-wheel installs PASS. Clean public environments
have no cryptography, load no private data and validate retrieval/citations.
Independent separate reviewer52 tests PASS plus Ruff/types/installed-byte review.
[20 recorded gates and log digests](evidence/m2f/preflight.json).

Public synthetic tests actually cover four concurrent processes/one claim, all
terminal repeat refusals, SIGKILL and KeyboardInterrupt, missing lock, unsafe
permissions/symlinks, key/cipher/claim damage, record swaps/gaps, version/config/
Gold mismatch, missing gate/code/log mismatch, complete transcript and no-rank
recovery. Earlier synthetic failures remain locally retained, not relabeled PASS.
First real custody preflight stopped before claim on an overly strict four-file
archive assumption. Fixed enumeration retains/scans historical archives too;
current pairs remain mandatory. Full checks and independent review reran after
this sole engineering correction. No ranking was involved in that failed preflight.

Trusted preflight authenticated reviewed Gold and backup in memory, checked all
8 frozen ciphertexts, frozen journal prefix and no pending seals. Raising ranker
sentinels observed0 attempts. Exact-content scan:20 current/historical artifacts,
2348 blobs/files/members;0 complete-query/Gold/key/ciphertext leaks at pre-run
checkpoint. Final publication scans are separately recorded below/in PR delivery.
No prior formal ledger existed; retained M2E journal/history recorded no formal
run. This checks known recorded history, not impossible proof of no unlogged
same-uid activity. Original Vault17 files/5 dirs retain identical byte/permission
fingerprint before/after the complete run; its journal was not appended.

## Formal results — fixed release, all three methods

Release lqd-heldout-retrieval-2.0.0; top3, known category/snapshot;18 positives and
6 negatives for each method. A1 authority BM25 differs in candidate pool from v1
mixed BM25/keyword. No catalog supplementation, extra query, retry, new cutoff or
after-result tuning. Full raw report and all72 method/case results stay encrypted
locally; [public exact fractions and all negative statuses](evidence/m2f/formal-summary.json).

| Method | Hit@3 | Macro Evidence Recall@3 | MRR@3 | Auxiliary Chunk Recall@3 | Retrieval Required Evidence Coverage | Full Evidence Case Rate | Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 |1.000000|1.000000|0.916667 (11/12)|1.000000|1.000000|1.000000|2/6|
| v1 BM25 |1.000000|0.953704 (103/108)|0.916667 (11/12)|0.953704|0.953704|0.888889 (16/18)|2/6|
| v1 keyword |1.000000|1.000000|0.972222 (35/36)|1.000000|1.000000|1.000000|3/6|

All methods: errors0,not_run0. Fixed A1 passes Recall>=.90 and MRR>=.80. Keyword
has higher observed MRR; this does not authorize selecting another method after
the held-out run or demonstrate A1 superiority. Required coverage here concerns
18 retrieval annotation targets, not original20-case Agent runtime REC or all
business workflow evidence. Coverage and Full Evidence Case Rate are distinct.

| Negative ID / kind | A1 | v1 BM25 | keyword |
|---|---|---|---|
| N01 no_policy_evidence |FAIL|FAIL|FAIL|
| N02 faq_not_authority |FAIL|FAIL|FAIL|
| N03 unverifiable_fact |PASS|PASS|PASS|
| N04 execution_not_authorized |FAIL|FAIL|PASS|
| N05 expired_only |PASS|PASS|PASS|
| N06 no_policy_evidence |FAIL|FAIL|FAIL|

IDs have M2E- prefix. N01/N06 fail the declared empty-hit assertion: lexical
matching returns evidence despite no annotated answer. N02/N04 A1 failures mean
incomplete annotated rationale coverage, not proven unsafe business execution.
FAQ hits are navigation diagnostics and never credited as authoritative units;
baselines do not fail merely for including FAQ alongside complete policy evidence.
All returned sources were applicable and citations valid; source validity alone
does not establish query relevance, private facts, approval or task success.

## Persistent evidence and recovery

Run ID **6bec8a95-790b-4be0-8adb-d785aeac0069**. Durable STARTED at
2026-10-25T03:34:10.128642Z; COMPLETED at03:34:10.863330Z (local11:34:10,+08).
[Public audit receipt](evidence/m2f/formal-audit.json). Full report SHA256:
7608afdcc7d7ab3c975d5ce750ff14ac5de71e975a6b93e301591d5d2405263d.

The historical ledger remains in restricted custody. It retained 74 immutable
encrypted records, an independent result key, an exclusive STARTED claim and
the terminal event. No output-directory switch or reset
flag. FAILED/STARTED/COMPLETED claims refuse another formal run. Saved-byte
recovery authenticated all74 records, exact Gold/config/source/result metadata,
ordinals, per-case arithmetic and final report digest; raising rank/input-loader
sentinels observed0 calls. A separate trusted arithmetic program independently
checked all72 saved rows (54 positive×6 metrics +18 negative assertions) and
macros with exact fractions, without retrieval or private plaintext output.
No second formal execution was attempted. Partial/corrupt-run behavior is tested
with public data, not induced on the actual completed release.

## Limits and next decision

Process-separated, non-blind, same macOS uid; not independent access isolation,
immutable/tamper-proof audit, statistically independent generalization or a real
enterprise result. Small fixed corpus has known shared evidence. Same-disk
recovery is not off-device backup. Gold/keys/full results never enter Git/CI,
public packages, normal worktree or model context. Model/OAuth/API calls0; no
inferred token/currency usage. No Agent business execution or safety evaluation.

M2 positive metric gate PASS; negative diagnostic suite FAIL; **overall M2 Quality
Acceptance FAIL** under the retained declared conditions. Keep all failed cases
and original v1 failures. Recommend PO review of missing rationale/abstention
limitations and a separately scoped future decision; do not retune this release,
rerun, replace A1, lower targets or begin M3 automatically.

Delivery is one stacked Draft PR targeting codex/phase3-m2e-integration. Later
safe merge dependency: PR14 → PR15 → PR17 → M2F, only with explicit PO authority,
retarget/main synchronization and fresh full checks at each step. Current quality
failure and future approval remain visible. Final-head hosted checks belong in
the PR body/final delivery; local results and PR17's old CI are not substitutes.
