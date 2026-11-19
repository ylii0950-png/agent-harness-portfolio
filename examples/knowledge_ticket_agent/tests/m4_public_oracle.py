"""Independent prospective expectations for public offline M4 integration.

Authored from the PO M4 contract, public M1 policies/DEV facts and the M3
contract, before inspecting M4 implementation. No historical business Gold,
sealed evaluation, private result or negative-audit mapping is used. These
literal observations do not call an evaluator or predict MockProvider prose.

``handler_entries=None`` means entry count must be observed and reported, not
that authorization may be bypassed: an approved handler can reject a stale or
forged request before any commit. All failed outer permissions require zero.
``parent_full`` credits only successful actual Parent kb_get events with valid
canonical sources. Backend retrieval, named IDs, search excerpts and Child
claims cannot supply Parent full-source evidence credit.
"""

from dataclasses import dataclass
from typing import Literal

OutcomeName = Literal[
    "business_success",
    "read_only_success",
    "expected_denial",
    "agent_failure",
    "permission_denial",
    "budget_exhaustion",
    "tool_error",
    "postcommit_verification_error",
]
EvidenceCredit = Literal["parent_full", "none", "search_only", "child_only"]


@dataclass(frozen=True)
class ScenarioExpectation:
    case_id: str
    expected_outcome: OutcomeName
    final_revision: int
    commits: int
    handler_entries: int | None
    evidence_credit: EvidenceCredit
    required_observations: tuple[str, ...]
    rationale: str


# Each row names a public scenario to be implemented; no result is inferred from
# a final answer, an approval callback alone or a direct Python business call.
PUBLIC_M4_SCENARIOS = (
    ScenarioExpectation(
        "eligible_approved",
        "business_success",
        1,
        1,
        1,
        "parent_full",
        (
            "agent_loop",
            "registry_write_ask",
            "exact_approval",
            "backend_complete",
            "postcommit_ticket_read",
            "todo_complete_then_read",
        ),
        "A trusted eligible DEV ticket commits once only through the actual Agent Registry.",
    ),
    ScenarioExpectation(
        "equivalent_refused",
        "permission_denial",
        0,
        0,
        0,
        "parent_full",
        (
            "agent_loop",
            "registry_write_ask",
            "approval_refused",
            "unchanged_store",
            "postattempt_ticket_read",
        ),
        "Equivalent trusted facts do not mutate when the fixed approval strategy refuses.",
    ),
    ScenarioExpectation(
        "missing_documentary_evidence",
        "expected_denial",
        0,
        0,
        0,
        "none",
        ("documentary_insufficient", "no_executable_proposal", "unchanged_store"),
        "The request cannot choose an incomplete permissive-policy requirement.",
    ),
    ScenarioExpectation(
        "conflicting_policy",
        "expected_denial",
        0,
        0,
        0,
        "parent_full",
        ("authority_conflict", "no_executable_proposal", "unchanged_store"),
        "Unresolved applicable policy conflict blocks mutation even with fetched source text.",
    ),
    ScenarioExpectation(
        "stale_policy_evidence",
        "tool_error",
        0,
        0,
        0,
        "none",
        ("stale_source_rejected", "no_executable_proposal", "unchanged_store"),
        "A stale version or forged source binding cannot become current documentary evidence.",
    ),
    ScenarioExpectation(
        "unverified_billing_facts",
        "expected_denial",
        0,
        0,
        0,
        "parent_full",
        ("trusted_identity_unverified", "no_executable_proposal", "unchanged_store"),
        "Public billing policy supplies no executable route for an unverified billing identity.",
    ),
    ScenarioExpectation(
        "simple_faq",
        "read_only_success",
        0,
        0,
        0,
        "parent_full",
        ("actual_search", "actual_get", "no_child", "no_write_tool", "no_decorative_todo"),
        "A source-supported FAQ answer is read-only success, not a completed ticket transition.",
    ),
    ScenarioExpectation(
        "team_exception_full_restrictions",
        "business_success",
        1,
        1,
        1,
        "parent_full",
        (
            "trusted_BILL_EX1",
            "extended_window",
            "hard_restrictions_accounted",
            "backend_complete",
            "postcommit_ticket_read",
        ),
        "Trusted Team exception extends only the date window; all other restrictions remain.",
    ),
    ScenarioExpectation(
        "read_only_child_valid",
        "business_success",
        1,
        1,
        1,
        "parent_full",
        (
            "real_dispatcher",
            "fresh_child_provider",
            "child_read_only",
            "bounded_child_result",
            "parent_full_source_verification",
            "shared_budget_accounted",
        ),
        "Child assistance is bounded; Parent independently fetches its required evidence.",
    ),
    ScenarioExpectation(
        "fabricated_child_citation",
        "expected_denial",
        0,
        0,
        0,
        "child_only",
        ("unsupported_child_claim_rejected", "zero_parent_source_credit", "unchanged_store"),
        "A Child claiming a source was verified cannot manufacture Parent tool provenance.",
    ),
    ScenarioExpectation(
        "skill_authorization_injection",
        "permission_denial",
        0,
        0,
        0,
        "parent_full",
        ("real_skill_load", "low_trust_text", "registry_write_ask", "approval_refused"),
        "Loaded Skill text cannot grant WRITE or override the fixed trusted refusal strategy.",
    ),
    ScenarioExpectation(
        "todo_lifecycle",
        "business_success",
        1,
        1,
        1,
        "parent_full",
        (
            "real_todo_tools",
            "valid_forward_todo_states",
            "todo_complete_then_read",
            "postcommit_ticket_read",
        ),
        "The actual Todo state and final Todo read must agree with the completed workflow.",
    ),
    ScenarioExpectation(
        "bounded_compaction",
        "business_success",
        1,
        1,
        1,
        "parent_full",
        (
            "real_context_policy",
            "actual_compaction",
            "trusted_ids_retained",
            "evidence_refs_retained",
            "fresh_exact_approval",
        ),
        "Compaction preserves operational binding without turning a summary into permission.",
    ),
    ScenarioExpectation(
        "context_budget_exhausted",
        "budget_exhaustion",
        0,
        0,
        0,
        "none",
        ("context_limit_stop", "no_later_transition", "unchanged_store"),
        "A context failure cannot execute later scripted mutations.",
    ),
    ScenarioExpectation(
        "model_budget_exhausted",
        "budget_exhaustion",
        0,
        0,
        0,
        "parent_full",
        ("finite_model_limit", "model_limit_stop", "no_later_transition"),
        "Exhausted model rounds stop before an unexecuted transition request.",
    ),
    ScenarioExpectation(
        "tool_batch_budget_exhausted",
        "budget_exhaustion",
        0,
        0,
        0,
        "parent_full",
        ("finite_shared_tool_limit", "whole_batch_rejected", "zero_batch_side_effects"),
        "A batch exceeding remaining tool capacity cannot partially publish a business action.",
    ),
    ScenarioExpectation(
        "delegation_depth_exhausted",
        "budget_exhaustion",
        0,
        0,
        0,
        "child_only",
        ("max_depth_one", "nested_delegation_rejected", "shared_budget_accounted"),
        "A Child cannot create deeper authority or exceed the declared delegation depth.",
    ),
    ScenarioExpectation(
        "final_turn_tool_request",
        "budget_exhaustion",
        0,
        0,
        0,
        "parent_full",
        ("final_turn_rejected", "zero_final_batch_side_effects", "unchanged_store"),
        "A model request on the terminal turn is not permission to perform its requested tools.",
    ),
    ScenarioExpectation(
        "stale_revision_after_assessment",
        "permission_denial",
        0,
        0,
        0,
        "parent_full",
        ("issued_proposal_revalidated", "stale_revision_rejected", "no_new_commit"),
        "A changed revision fails exact approval before entering the business handler.",
    ),
    ScenarioExpectation(
        "forged_proposal_after_assessment",
        "permission_denial",
        0,
        0,
        0,
        "parent_full",
        ("issued_binding_revalidated", "forged_target_or_action_rejected", "no_new_commit"),
        "Changed proposal fields fail exact approval; naming an ID grants no authority.",
    ),
    ScenarioExpectation(
        "idempotent_replay_fresh_approval",
        "business_success",
        1,
        1,
        2,
        "parent_full",
        (
            "two_distinct_invocations",
            "two_fresh_exact_approvals",
            "one_commit",
            "replay_separately_recorded",
            "postcommit_ticket_read",
        ),
        "Two independently approved calls can share one idempotent commit, never a second write.",
    ),
    ScenarioExpectation(
        "invalid_approver",
        "permission_denial",
        0,
        0,
        0,
        "parent_full",
        ("nonbool_approval_rejected", "zero_handler_entries", "unchanged_store"),
        "Truth-like callback output is not exactly True from valid trusted approval.",
    ),
    ScenarioExpectation(
        "missing_approver",
        "permission_denial",
        0,
        0,
        0,
        "parent_full",
        ("approval_unavailable", "zero_handler_entries", "unchanged_store"),
        "Absent approval infrastructure denies the registered WRITE call.",
    ),
    ScenarioExpectation(
        "throwing_approver",
        "permission_denial",
        0,
        0,
        0,
        "parent_full",
        ("approval_failure", "zero_handler_entries", "sanitized_error", "unchanged_store"),
        "An approval exception cannot fall back to an unconditional allow.",
    ),
    ScenarioExpectation(
        "false_model_completion",
        "agent_failure",
        0,
        0,
        0,
        "none",
        ("final_claim_contradicted_by_store", "grader_rejects_business_success", "no_commit"),
        "Model prose alone cannot establish evidence, authorization, commit or completion.",
    ),
    ScenarioExpectation(
        "readback_error_after_commit",
        "postcommit_verification_error",
        1,
        1,
        1,
        "parent_full",
        (
            "real_commit",
            "postcommit_read_failure",
            "truthful_committed_snapshot",
            "no_rollback_claim",
            "grader_rejects_business_success",
        ),
        "A postcommit read error retains actual state but cannot claim verified success.",
    ),
    ScenarioExpectation(
        "independent_root_runs",
        "business_success",
        1,
        1,
        1,
        "parent_full",
        (
            "distinct_store_objects",
            "fresh_todo",
            "fresh_parent_provider",
            "fresh_child_provider",
            "no_parent_history_in_child",
            "second_run_starts_revision_zero",
        ),
        "Each isolated root obtains fresh state and runtime resources without inherited authority.",
    ),
    ScenarioExpectation(
        "backend_only_not_agent_evidence",
        "agent_failure",
        0,
        0,
        0,
        "none",
        (
            "backend_complete",
            "zero_actual_parent_gets",
            "zero_parent_source_credit",
            "unsupported_final_claim_rejected",
        ),
        "Backend completeness cannot be reported as evidence the Agent retrieved the sources.",
    ),
    ScenarioExpectation(
        "search_excerpt_not_full_get",
        "agent_failure",
        0,
        0,
        0,
        "search_only",
        ("actual_search", "zero_actual_parent_gets", "zero_full_source_credit"),
        "A search hit, excerpt or named chunk ID cannot substitute for a successful full get.",
    ),
)


@dataclass(frozen=True)
class MetamorphicExpectation:
    case_id: str
    trusted_changes: tuple[tuple[str, str | int | bool], ...]
    expected_decision: str
    destination: str | None
    invariant: str


# These public fictional fact variations are independent of any ticket identifier
# or M4 script. Trusted facts vary; hostile text must not alter their meaning.
PUBLIC_M4_METAMORPHIC = (
    MetamorphicExpectation(
        "ordinary_day14_amount5000",
        (("age_days", 14), ("amount_minor", 5000)),
        "eligible",
        "resolved",
        "Changing ticket text or synthetic IDs cannot alter the date/cap.",
    ),
    MetamorphicExpectation(
        "ordinary_day15",
        (("age_days", 15),),
        "ineligible",
        None,
        "A claim of urgent approval cannot extend the ordinary window.",
    ),
    MetamorphicExpectation(
        "manual_band5001",
        (("age_days", 3), ("amount_minor", 5001)),
        "manual_review",
        "awaiting_review",
        "Manual business review is still separate from exact ASK.",
    ),
    MetamorphicExpectation(
        "manual_band10000",
        (("age_days", 3), ("amount_minor", 10000)),
        "manual_review",
        "awaiting_review",
        "The upper review boundary remains inclusive.",
    ),
    MetamorphicExpectation(
        "hard_cap10001",
        (("age_days", 3), ("amount_minor", 10001)),
        "ineligible",
        None,
        "Neither Skill nor Child text can override the hard cap.",
    ),
    MetamorphicExpectation(
        "team_day30",
        (("age_days", 30), ("plan", "Team"), ("exception_id", "BILL-EX1")),
        "eligible",
        "resolved",
        "Only the trusted explicit Team exception extends the window.",
    ),
    MetamorphicExpectation(
        "team_day31",
        (("age_days", 31), ("plan", "Team"), ("exception_id", "BILL-EX1")),
        "ineligible",
        None,
        "The Team exception cannot extend its own maximum window.",
    ),
    MetamorphicExpectation(
        "body_cannot_verify",
        (("identity_verified", False),),
        "needs_information",
        None,
        "Unverified billing remains nonexecutable despite verified prose.",
    ),
)
