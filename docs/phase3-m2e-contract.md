# M2E creation contract, before query authorship

2026-10-24. PO authorized full M2E creation/review/freeze, then Local-First mode
because Actions allowance is exhausted. Base958a124111551a60779d6d8853f97dd49c8f264f;
PR14/15 stay Draft/unmerged, main6a55688172ba6bfac09d60e1c6982c94389e7743.
No push, new PR, remote run, purchased minutes, paid runner, formal scoring,
OAuth/model/catalog access, A1 tuning or M3. Mac checks are not Ubuntu CI.

## Frozen inputs and interface

The accompanying verification/m2e/retriever-lock.json freezes A1 code/config,
all old source/tests/Gold/reports, corpus/index/chunker and metric formulas before
new query creation. No changes to those files are permitted by this work package.
Maintenance code is a separate new optional Demo module, never a ranker change.

Version lqd-heldout-retrieval-2.0.0:18 positives (6 per business category),6
negatives (2 per category), all fictional English from unchanged source policies.
Known category and snapshot are inputs, not classification predictions. Each
record contains query_id, query, category, snapshot, positive, stratum,
units, required_unit_ids, negative_kind, expected_behavior, review_notes.
Each annotated unit has unit_id, anchors (exact original Anchor records), and
alternatives (OR of AND Chunk-ID bundles). An equivalent anchor is not an extra
unit; independent policy obligations stay separate. Required units here are
retrieval targets, not all business workflow evidence. Negatives may have source
anchors explaining a refusal/unknown fact, but are never scored as positives.
No-evidence differs from expired-only, unverified-fact, FAQ-not-authority and
execution-not-authorized: latter kinds do not imply retrieval must return empty.

Document root fields: version, synthetic, data_origin, corpus_version,
manifest_sha256, index_sha256, chunker_version, config_sha256, metrics_version,
cases. Strict schema and bounded sizes apply. IDs M2E-P01..18/M2E-N01..06.
All positive annotations bind active authoritative source spans and exact mechanical
chunk coverage; minimum union of one valid bundle per unit must be <=3. Keep
both policies in conflict targets. Semantic review checks query meaning separately
from mechanical parsing. No rankings are consulted for authorship or review.

Fixed scoring specification is preserved: positive macro denominator18; Hit is
any annotated relevant chunk, Recall covered semantic units/units per query,
MRR first related rank<=3, raw chunk recall auxiliary. OR equivalents do not
inflate unit denominator; partial AND bundles earn no unit. Retrieval required-
anchor Top3 coverage is separate from original Agent runtime REC. Negative
behavior denominator6 separately, with versioned kind-specific expectations.
Proposed thresholds0.90/0.80 unchanged. No scoring is implemented/executed here.

## Roles, custody and honest isolation

Primary task is frozen Retriever Implementer/maintenance-tool author and receives
only aggregate/ID-level QA. Separate Codex Evaluation Steward creates annotations;
Independent Reviewer compares all source meanings/duplicates/negative validity.
They share uid/filesystem; this is process separation, **NOT independent blindness**.
No privileged account/service or imaginary technical ACL is claimed.

Store complete new data only in an authenticated encrypted local vault beneath
`.git/m2e-steward-private/`, outside the normal Git worktree and all distributions.
Directory0700, files0600, owner-only key and append-only-by-tool event journal.
Use existing local cryptography development dependency, no new project dependency.
Do not print secret key, queries, Gold or decrypted contents in ordinary logs.
Decrypt into memory only. Retain immutable initial submission and reviewed release
as separate encrypted entries. Keep a local encrypted backup and test recovery.
Same-user keys/access remain possible; event labels are process audit, not proof
of unforgeable OS identities. Git cloning/pushing cannot restore private data;
local backup is not off-device disaster recovery. Report these limitations.

Ordinary commits contain generic maintenance tools, synthetic non-held-out tests,
opaque whole-artifact digests, counts, review/status and structural proof summaries;
no full new queries, per-case answer mappings or private vault/key. Prevent public
resource/ordinary worktree roots from being accepted as private vault locations.

## Gates and review

1. Commit this contract and retriever lock before data creation.
2. Steward submits authored initial data without ranking; log provenance and
   proposed coverage. Do not claim review of uncreated data.
3. Separate semantic review checks all24, original sources/versions, omitted or
   overbroad units, alternatives, negatives and overlap with original12dev+18eval.
   Lexical diagnostics are alerts, not semantic independence proof. Record exact,
   high lexical, mechanical numeric/name substitutions, semantic and evidence
   overlap risks; retain initial and corrective versions, never rank-fit.
4. Validate exact anchors/mapping and independent minimum-cover witnesses;
   keep initial unreachable cases and predeclared semantic admission rationale.
5. Freeze reviewed artifact+manifest/code/input hashes, journal position and
   local encrypted backup. Verify recovery/tamper rejection. No silent overwrite.
6. Run offline 3.11/3.12 core+Demo+new tests, Ruff, strict mypy, dual builds,
   clean installs, package/staged leakage checks and independent code review.

No ranking is an allowed stage. Formal scoring NOT_RUN; M2 acceptance STILL_PENDING;
M3 NOT_AUTHORIZED. Independent access isolation NOT_ESTABLISHED. Cryptographic
local freeze integrity is distinct from a genuinely independently held blind
release. Remote Linux/macOS CI and final publication remain pending until quota
recovers and the PO authorizes the deferred grouped push/PR/run.
