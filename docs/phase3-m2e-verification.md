# Phase 3 M2E — Local-First creation, QA and technical freeze

2026-10-24. **No new retrieval score. No push, PR, hosted run, merge or M3.**
The PO reports Actions2000/2000 minutes exhausted, expected reset2026-11-01;
this report does not independently verify the account allowance/reset schedule.
Remote verification is DEFERRED, never success. Local Mac is not Ubuntu CI.

## Baseline and freeze before creation

Verified PR14 remains Draft/OPEN at f0f9297626c06e8eba83ca3a0f34b9158dfc048a;
PR15 Draft/OPEN at958a124111551a60779d6d8853f97dd49c8f264f with base
codex/phase3-m2; remote main6a55688172ba6bfac09d60e1c6982c94389e7743 unchanged.
Created local codex/phase3-m2e from approved M2R HEAD. Before new queries existed,
commit8dccc68 froze [creation/custody contract](phase3-m2e-contract.md) and
[retriever lock](../examples/knowledge_ticket_agent/verification/m2e/retriever-lock.json).
The lock pins192 original input files, A1/config, chunker/index/corpus, metric
version, dev experiments, tests and historical evidence. Automated regression
checks the lock's own SHA and every bound file; A1 was not changed or retuned.

Frozen source:
- Corpus lqd-data-1.0.0, manifest
  `52a13a1b0b0eae835fde7f27612b14dea57d8ecbf457aa21dd49d26127d856c9`.
- Index `e54a66cf15711628885acf289e6425c769e6b0f2af8f2463ac32c6b8ddfbf004`;
  chunker lqd-section-1.0.0,21 original chunks.
- A1 config `5dad0b47b42f6a1bb42fefcf3f6f592f8732970543edbfdd7a7c6949b13e2354`.
- Lock `fef2103c881461ec5178a35b0439b7a450726a448fe569c8c17f4bb56db3b8bd`.
- Existing M2/M2R scores remain historical failed-quality/development evidence;
  they are not this dataset's results. Historical28/26/10 request evidence intact.

## Roles and actual access boundary

Primary Codex task is frozen Retriever Implementer and generic maintenance author.
Separate `m2e_evaluation_steward` creates and keeps the24 questions/annotations;
`m2e_semantic_reviewer` independently checks all original policies and24 cases;
`m2e_tool_review` independently reviews generic source/test/security behavior
without reading private Gold. These are task roles on the **same macOS uid and
filesystem**, not separate authenticated OS accounts or an independently blind
stewardship service. The primary received aggregate/ID-level records, not the
new query/Gold plaintext. This convention is not an access-control guarantee.
Codex task contexts/history also do not constitute an isolation boundary.

**Independent Access Isolation: NOT ESTABLISHED.** Same-account processes can
read the local key and decrypt. Encryption and0700/0600 protect against accidental
publication and other ordinary users, not the Retriever Implementer's uid. A
branch/worktree/task label is not an ACL. Do not retrospectively call this set
blind after moving it elsewhere. A genuinely independent blind study requires
separate controlled authorship/access and potentially a future fresh version.

## Dataset, strata and retained revisions

Version **lqd-heldout-retrieval-2.0.0**; wholly fictional English from unchanged
Lantern Quay Desk policies.24 total:18 positives (6 Billing/6 Access/6 Export),
6 negatives (2 each category). Includes basic limitations, identity/provenance,
exceptions, multiple policy obligations, version applicability, explicit conflict,
low-trust instructions and simulated-only boundaries. Negative kinds explicitly
separate no policy evidence(2), expired-only(1), unverifiable facts(1), FAQ not
authority(1), and execution not authorized(1). The latter kinds do not mean the
retriever must return no policy: knowledge of a policy cannot prove a private
fact, grant approval or execute a business action.

Annotations include exact original source spans/version/section, source-reviewed
semantic units, OR alternatives/AND complements, stable mechanical Chunk IDs,
required targets and expected behavior. Independent policies remain separate.
Do not manufacture equivalent alternatives: actual admissible alternatives come
from original source meaning, not retrieval rank. These required targets are
single-query evidence, not every business workflow obligation or runtime REC.

The encrypted initial submission is immutable. Independent review requested
changes: P12 within-new semantic overlap; P15 old-query semantic overlap; P09
wording ambiguity; P17 missing obligation grounding; N02 metadata versus quoted
policy rationale; N02–N05 required-ID association. Steward revised before freeze,
with no ranks consulted. All24 were independently re-reviewed, not just changed
records; final **PASS_WITH_LIMITATIONS**, no remaining semantic blocker. Initial,
initial review, revision rationale and final review are separately retained.
No unreachable candidate was dropped; initial positives already all had m_q<=3.

## Duplicate/semantic correlation audit

Compared all24 against exactly12 original dev and18 exposed old positive queries,
plus within-new pairs. Frozen lexical alerts: normalized exact, token Jaccard
>=.65, character sequence ratio>=.85, or equal numeric-substitution template.
Final exact/threshold alerts0; independent semantic review checked paraphrases,
mechanical changes in names/numbers, intent and full evidence sets. The process
is not a semantic-similarity oracle or proof of statistical independence.

**6/18 positives share a complete evidence set with an old question**, and two
within-new positive pairs share complete evidence sets. Their support tasks differ;
this substantial fixed-corpus/rule correlation is retained, not hidden. No new
business domain, guaranteed model generalization or independent blind-test claim.
Complete comparisons/rationales stay encrypted, not in a developer-readable PR.
See [non-answer QA summary](../examples/knowledge_ticket_agent/verification/m2e/dataset-qa.json).

## Per-query structural proof (not retrieval quality)

Exact Anchor equality, authoritative active scope/version and every OR/AND mapping
are checked. Exhaustive union enumeration produces minimal witness sets and is
cross-checked with the original independent `minimum_cover` implementation.
Private freeze retains actual witnesses; public proof exposes only IDs/counts.

| Minimum distinct Chunks | Positive IDs | Count |
|---|---|---:|
|1|P01,P02,P03,P06,P07,P08,P12,P13|8|
|2|P04,P05,P09,P10,P11,P14,P15,P18|8|
|3|P16,P17|2|

All IDs have M2E- prefix. Every positive satisfies m_q<=3; each has one minimum
witness set in this corpus. This is structural feasibility, **not** a successful
Top3 retrieval result. Formal ranking, Hit/Recall/MRR and negative behavior
scores have **not** been run.

## Metrics frozen without scoring

Positive macro denominator18; negative behavior denominator6 separately.
Hit@3: any annotated related chunk. Macro Evidence Recall@3: complete semantic
units/units per query, then mean over18; OR equivalents do not inflate units and
partial AND bundles do not count. MRR@3: reciprocal of earliest related rank<=3,
zero for no hit. Raw Chunk Recall@3 is auxiliary only. Errors retain denominators.
Retrieval required-anchor Top3 coverage is saved separately from M4/M5 actual
kb_get/verified-evidence Runtime REC. Current targets all require their annotated
units, so these retrieval coverage definitions coincide; neither proves task
completion or full business workflow coverage. Proposed0.90 Recall/0.80 MRR
thresholds unchanged. No scorer or automatic execution was added for this set.

## Local encrypted custody, freeze and recovery

Complete queries/Gold are **not in the normal worktree, Git history or packages**.
Local location: `.git/m2e-steward-private/` in this repository. All private and
recovery directories0700/files0600, current uid owner, no symlinks/hardlinks.
Plaintext is decrypted in steward/reviewer memory; no plaintext file is retained.
The new randomly generated local evaluation key is not OAuth or another app's
credential and is never committed/printed. It remains accessible to the same uid.

Final reviewed Gold49,630 bytes; plaintext SHA256:
`f7a25aadfa0e60449bbe03933c8352536a03d30283d6ce5dcdaecfa6e6109651`.
[Public freeze manifest](../examples/knowledge_ticket_agent/verification/m2e/freeze-manifest.json)
pins whole-artifact/corpus/index/config/source digests, counts, structural proofs,
review/custody/backup metadata and explicit isolation/score limitations. Full
per-case mapping remains only in encrypted entries. Final release and backup
use AES-GCM with AAD=version+NUL+canonical filename; filename swaps fail. Historical
initial files used version-only AAD before review and remain archived submissions,
never silently upgraded or used as the final release. There is no fallback.

Backup `backup-reviewed.aesgcm` and owner-only recovery-key copy were actually
restored into a separate restricted local recovery directory; LocalVault decryption
returned the exact frozen Gold digest. Corrupted ciphertext and wrong filename
AAD were rejected. Originals were rehashed unchanged. This is **same disk**, not
off-device disaster recovery; deleting/corrupting all Git metadata can lose both
copies. Git clone/push cannot recover this intentionally untracked vault.

Journal uses serialized file locks, pinned directory/lock inodes and a hash chain.
Its freeze snapshot is the first18 events, no pending seals. Later reads append;
that snapshot is a prefix, not a falsely permanent tail. Bootstrap initial/review
access declarations were imported after the event with actual import timestamps,
not fabricated historic times. Actor strings/hash chains are process records,
not authenticated roles, tamper-proof OS audit or proof nobody else accessed data.
Failed/pending seals and partial/corrupt journals block validation; no silent
rewrite or approval broadening. Corrections require retained old versions and
new review/version proposals, never overwrite based on future score failures.

## Maintenance code and meaningful regression tests

Three new optional Demo modules only:
- evaluation_maintenance: strict schema, exact Anchors/scopes/version/config,
  OR/AND mapping, exhaustive witnesses, fixed18/6 and lexical duplicate alerts.
- evaluation_custody: explicit encrypted local operations, no key discovery,
  no other credential access, bounded no-follow reads, owner-only files, locked
  process journal, failure/pending visibility and immutable-by-tool new entries.
- evaluation_cli: public integrity smoke or explicit steward vault validation;
  generic redacted errors/ID-level output, no Gold text and no --score entry.

Existing locked cryptography50.0.2 is used only for maintenance; no new package
or dependency. Clean public CLI works with cryptography absent. Tests use visibly
artificial public fixtures, never load the real private Gold or model credentials.
Rank functions are replaced by raising sentinels during actual steward maintenance;
observed rank attempts0. No model request, OAuth/catalog/key fallback or paid API.
This is execution evidence, not an unbypassable same-user enforcement claim.

Independent code review found/fixed: journal concurrency, post-write failure
incorrectly clearing pending, missing filename AAD binding, late actor validation,
empty payload mismatch, UTF8 diagnostics and failing-test child cleanup. A real
Mac concurrent lock-file open ENOENT was reproduced in fresh environments; traceback
located transaction-time creation, not a model/Gold error. Lock creation was moved
to initialization; transactions pin/open the existing lock inode and reject
replacement. Preserved failure diagnostic and bounded follow-up logs, not only the
best run. Five primary and three independent bounded post-fix repetitions passed;
final reviewer ran all69 new tests. Constructor directory-swap negative test also
passes. No remaining code blocker. See [independent reviews](phase3-m2e-independent-review.md).

## Actual local engineering results

macOS arm64, installed non-editable packages; each interpreter:

| Check | Python3.11.17 | Python3.12.15 |
|---|---:|---:|
| Core pytest |788 passed|788 passed|
| Demo pytest |343 passed|343 passed|
| Unique total |1131 passed|1131 passed|
| Failures/skips |0/0|0/0|
| Ruff lint/format0.17.1 |PASS|PASS|
| Strict mypy2.4.0 |Core81/Demo37 PASS|Core81/Demo37 PASS|
| Core+Demo sdist/wheel |PASS|PASS|
| Fresh two-local-wheel install |PASS|PASS|

69 new M2E tests; all1062 prior tests unchanged. pytest9.1.1, uv0.13.5,
setuptools84.2.1; system Python untouched. Clean environments used --no-index
--no-deps and only built local wheels: public data CLI and maintenance public-check
pass, with cryptography absent. Private operations run in the locked dev environment.
239 protected preexisting files byte-identical, including all core/Demo source,
old tests/data/Gold/config/reports/journals and original CI. Source code freeze
verified separately from dataset structural CLI; do not conflate them.

[Local check manifest](../examples/knowledge_ticket_agent/verification/m2e/local-verification.json)
and committed non-sensitive pytest/failure-follow-up logs preserve exact results.
All raw local install/build/test logs and artifacts remain under ignored artifacts/;
build/install logs are not committed with machine-specific personal paths.
Four actual artifacts (335 members) were scanned for all24 complete query strings,
canonical new Gold and binary/hex/base64 key material: none found. Public policy
quotes are intentionally present and not Gold leakage. Filename exclusion alone
was not the only check. Final complete staged-increment scan is recorded separately.
Content scans do not prove every possible encoding/side channel absent.

## Deferred remote delivery and formal prerequisites

No push, PR creation, workflow dispatch, rerun, budget change, purchased minutes,
paid runner or modified workflow. **New Actions usage0 runs**, charged minutes not
measured. All new Ubuntu/macOS Python3.11/3.12 Core+Demo hosted checks remain
**DEFERRED_QUOTA / NOT_RUN**, no borrowed old-head successes. No Linux validation
was performed in a local VM/container. Existing4-environment8-job workflow remains.

After quota restoration, the deferred work is one grouped branch publication,
stacked Draft PR targeting codex/phase3-m2r, and current-HEAD Push/PR Core+Demo
matrices/build/clean-install checks. Preserve source/input/Gold freeze, recheck
baseline and publish only non-answer artifacts. This task installs no scheduler
and will not automatically push, create/merge PRs or execute formal evaluation.

Local technical integrity does **not** complete all formal-score prerequisites:
independent access isolation is NOT_ESTABLISHED, off-device custody not provided,
hosted CI deferred, separate formal-run approval absent. PO must choose an actual
independent custodian/account/environment and a future fresh blind protocol, or
explicitly accept a disclosed nonblind study and its limits. Moving current data
later cannot retroactively remove exposure. Neither option is selected here.

- Engineering: PASS (authorized Local-First scope; hosted verification deferred)
- Dataset QA: PASS (independent semantic task, with documented correlation limits)
- Freeze Integrity: PASS (local nonblind technical freeze only)
- Independent Access Isolation: NOT ESTABLISHED
- Formal Scoring: NOT RUN
- M2 Acceptance: STILL PENDING
- M3: NOT AUTHORIZED
