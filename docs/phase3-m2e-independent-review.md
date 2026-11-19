# M2E independent review evidence

2026-10-24, same macOS uid/filesystem. Task separation is engineering review,
not independent access isolation or blind scoring. No new ranks or model calls.

## Semantic review

Steward and semantic reviewer are separate Codex tasks. Reviewer independently
read original frozen policies/Rules/Manifest and all24 new records, then checked
against12 dev+18 exposed old questions and within-new semantics. Only after the
initial independent inspection did the reviewer consult the author's encrypted
risk analysis. Initial verdict REQUEST_CHANGES is retained encrypted.

Found P12 within-new semantic repetition, P15 old semantic overlap, P09 ambiguous
fact wording, P17 incomplete obligation grounding, N02 metadata/body rationale
confusion, and N02–N05 ID association errors. Steward retained initial and wrote
a separate reviewed entry. Reviewer rechecked all24, exact anchors/active versions,
independent-policy units, true negatives and minimum-cover mappings: final
PASS_WITH_LIMITATIONS, no remaining semantic blocker. Six positives share old
complete evidence sets; two within-new positive pairs share sets. Not a new domain
or statistically independent sample. Full record and comparisons remain encrypted;
public dataset-qa.json contains no question/answer mapping.

## Code/security review

A third task read generic maintenance code and synthetic tests only, not private
Gold. Review found and implementation fixed:

1. Unlocked journal append/creation races: serialize with local flock.
2. Files created before actor/audit validation: validate first, journal seal_begin
   and seal_complete; partial/failed writes remain unresolved, never frozen.
3. Incorrectly clearing pending for seal_failed: keep pending, add write-then-fail
   regression. Corrupt/partial journals fail closed and are not silently repaired.
4. Version-only AAD permits file swap: final entries bind canonical filename too;
   retained historical initial only, no automatic fallback.
5. Empty input mismatch, UTF8 diagnostics and failed-test subprocess cleanup.
6. Fresh Mac concurrency failed with ENOENT at transaction-time lock creation.
   Preserved diagnosis, initialized lock before workers, pin inode/open existing
   lock during transactions. Replacement fails; constructor rechecks directory
   identity/owner/0700 before leaf write. No guessing successful log means success.

Actual independent test evidence: fresh lock-fix environment68/68 new tests plus
3 bounded concurrency repetitions passed. Final installed environment, including
constructor directory-swap regression, **69/69 passed**. Primary's bounded5
post-fix repetitions also passed. Original failures retained in local logs.

Reviewed CLI has no score option; output is IDs/hashes/status only. Actual package
name exclusions are tested; Steward separately scanned every member's content
against the real hidden query/Gold/key bytes without exposing them to tool reviewer.
Precreation lock test verifies192 frozen inputs and pinned lock SHA. No core or
A1 changes; no new dependencies, CI edits, permission relaxation or automatic
run/publication. No remaining code blocker, subject to actual custody/semantic
attestations and explicitly deferred hosted checks.

## Limits

Review is not proof of unobservable same-user access, model safety, retrieval
quality, semantic completeness in every possible interpretation or tamper-proof
custody. Raw task contexts are not an ACL. Final local artifact/restore evidence
belongs to freeze-manifest.json; Linux/hosted CI and formal scoring are NOT_RUN.
