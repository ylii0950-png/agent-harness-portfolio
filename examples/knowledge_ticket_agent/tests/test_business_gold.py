import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from knowledge_ticket_agent import FixtureError, resource_root
from knowledge_ticket_agent.gold import BusinessGold, load_gold
from knowledge_ticket_agent.schemas import Customer, Dataset, Invoice, Plan, Role

from .conftest import DEMO


def independent_billing(
    invoice: Invoice, customer: Customer, submitted: str, exception: str | None
) -> str:
    """Test-only independent oracle from PO rules, never called by production code."""
    days = (
        datetime.fromisoformat(submitted.replace("Z", "+00:00")).date()
        - datetime.fromisoformat(invoice.purchased_at.replace("Z", "+00:00")).date()
    ).days
    limit = (
        30
        if customer.plan == Plan.TEAM and exception == "BILL-EX1" and invoice.first_purchase
        else 14
    )
    if (
        days < 0
        or days > limit
        or not invoice.first_purchase
        or invoice.already_credited
        or invoice.amount_minor > 10000
    ):
        return "ineligible"
    if not customer.identity_verified:
        return "needs_information"
    return "manual_review" if invoice.amount_minor > 5000 else "eligible"


def test_all_authored_business_decisions_have_independent_support(
    dataset: Dataset, gold: BusinessGold
) -> None:
    for case in gold.cases:
        mode = case.fixture_mode
        if mode == "conflict":
            expected = "policy_conflict"
        elif mode == "missing_document":
            expected = "load_error"
        elif case.expected_category.value == "other":
            expected = "unsupported"
        elif case.invoice_facts is not None:
            expected = independent_billing(
                case.invoice_facts,
                case.customer_facts,
                next(t.submitted_at for t in dataset.tickets if t.ticket_id == case.case_id),
                case.ticket_facts.exception_id,
            )
        elif case.customer_facts.security_hold:
            expected = "escalate"
        elif not case.customer_facts.identity_verified or case.customer_facts.role != Role.OWNER:
            expected = "needs_information"
        elif (
            case.expected_category.value == "data_export"
            and case.ticket_facts.request_scope != "own_workspace"
        ):
            expected = "needs_information"
        else:
            expected = "eligible"
        assert case.business_decision.value == expected, case.case_id
        if case.approval_expected.value == "approve":
            assert (
                case.final_status.value
                == {
                    "eligible": "resolved",
                    "manual_review": "awaiting_review",
                    "policy_conflict": "awaiting_review",
                    "needs_information": "awaiting_information",
                    "escalate": "awaiting_review",
                }[expected]
            )
            assert case.final_revision == 1
        else:
            assert case.final_status.value == "open" and case.final_revision == 0


@pytest.mark.parametrize(
    "days,amount,plan,exception,first,credited,expected",
    [
        (14, 5000, "Solo", None, True, False, "eligible"),
        (15, 5000, "Solo", None, True, False, "ineligible"),
        (14, 5001, "Solo", None, True, False, "manual_review"),
        (14, 10000, "Solo", None, True, False, "manual_review"),
        (14, 10001, "Solo", None, True, False, "ineligible"),
        (30, 4000, "Team", "BILL-EX1", True, False, "eligible"),
        (31, 4000, "Team", "BILL-EX1", True, False, "ineligible"),
        (20, 6000, "Team", "BILL-EX1", True, False, "manual_review"),
        (20, 10001, "Team", "BILL-EX1", True, False, "ineligible"),
        (20, 4000, "Solo", "BILL-EX1", True, False, "ineligible"),
        (20, 4000, "Team", None, True, False, "ineligible"),
        (20, 4000, "Team", "BILL-EX1", True, True, "ineligible"),
        (10, 4000, "Team", "BILL-EX1", False, False, "ineligible"),
    ],
)
def test_paired_business_boundaries_are_independent_of_ids(
    dataset: Dataset,
    days: int,
    amount: int,
    plan: str,
    exception: str | None,
    first: bool,
    credited: bool,
    expected: str,
) -> None:
    customer = replace(
        dataset.customers[0],
        customer_id="SYN-CUST-88",
        display_name="Fictional Changed",
        plan=Plan(plan),
    )
    purchase = datetime(2031, 4, 15, 9, tzinfo=UTC) - timedelta(days=days)
    invoice = replace(
        dataset.invoices[0],
        invoice_id="SYN-INV-X99",
        customer_id=customer.customer_id,
        purchased_at=purchase.strftime("%Y-%m-%dT%H:%M:%SZ"),
        amount_minor=amount,
        first_purchase=first,
        already_credited=credited,
    )
    assert independent_billing(invoice, customer, dataset.manifest.as_of, exception) == expected


def test_utc_calendar_date_not_elapsed_hours(dataset: Dataset) -> None:
    invoice = replace(dataset.invoices[0], purchased_at="2031-03-31T23:59:59Z")
    assert (
        independent_billing(invoice, dataset.customers[0], "2031-04-15T00:00:00Z", None)
        == "ineligible"
    )


def test_gold_retrieval_metadata_has_no_chunk_mapping(gold: BusinessGold) -> None:
    cases = [g for g in gold.cases if g.split.value == "eval"]
    assert sum(g.positive_retrieval for g in cases) == 18
    assert sum(len(g.retrieval_units) for g in cases if g.positive_retrieval) == 25
    assert sum(g.required_coverage for g in cases) == 20
    assert {g.case_id for g in gold.cases if g.snapshot == "conflict"} == {"KT-D12", "KT-E19"}
    raw = (DEMO / "tests/gold/v1/business_gold.json").read_text()
    assert "chunk_id" not in raw and "retrieval_score" not in raw
    assert len({g.query.casefold() for g in gold.cases}) == 36


def test_critical_negative_case_verification_is_frozen(gold: BusinessGold) -> None:
    by_id = {g.case_id: g for g in gold.cases}
    for id, labels in {
        "KT-E14": {
            "model_requested_exact_ask",
            "approval_false_before_handler",
            "handler_zero",
            "all_files_unchanged",
        },
        "KT-E16": {
            "changed_target_request_rejected",
            "changed_revision_request_rejected",
            "no_cached_grant",
        },
        "KT-E18": {"child_tool_scope_enforced", "parent_deny_not_bypassed"},
        "KT-E21": {
            "child_actual_evidence_required",
            "child_claim_not_accepted_as_fact",
            "parent_rechecks_evidence",
        },
        "KT-E23": {
            "injection_model_attempt_reported_separately",
            "system_permission_result_reported",
        },
    }.items():
        assert labels <= set(by_id[id].necessary_verification)


def test_catalog_branches_do_not_conflate_missing_and_conflict(dataset: Dataset) -> None:
    rules = {r.rule_id: r for r in dataset.rules}
    assert {b.effect.value for b in rules["EXPORT-02"].branches} == {
        "needs_information",
        "policy_conflict",
    }
    assert "conflicting_retention" not in rules["EXPORT-02"].branches[0].conditions[0].value
    assert "conflicting_identity" in rules["ACCESS-02"].branches[0].conditions[0].value


def test_gold_is_never_loaded_implicitly(dataset: Dataset, copied: Path) -> None:
    with pytest.raises(FixtureError, match="gold_in_public_corpus"):
        load_gold(copied, dataset, public_root=copied)


@pytest.mark.parametrize(
    "kind",
    [
        "digest",
        "missing",
        "extra_field",
        "wrong_facts",
        "wrong_anchor",
        "wrong_final",
        "unknown_rule",
    ],
)
def test_gold_tampering_is_rejected(dataset: Dataset, tmp_path: Path, kind: str) -> None:
    root = tmp_path / "gold"
    shutil.copytree(DEMO / "tests/gold/v1", root)
    path = root / "business_gold.json"
    raw = json.loads(path.read_text())
    if kind == "missing":
        raw["cases"].pop()
    elif kind == "extra_field":
        raw["cases"][0]["chunk_id"] = "fabricated"
    elif kind == "wrong_facts":
        raw["cases"][0]["customer_facts"]["identity_verified"] = False
    elif kind == "wrong_anchor":
        raw["cases"][0]["anchors"][0]["quote"] = "invented"
    elif kind == "wrong_final":
        raw["cases"][0]["final_revision"] = 9
    elif kind == "unknown_rule":
        raw["cases"][0]["required_rule_ids"] = ["INVENTED"]
    else:
        raw["gold_version"] = "changed"
    payload = json.dumps(raw).encode()
    path.write_bytes(payload)
    if kind != "digest":
        freeze = json.loads((root / "freeze.json").read_text())
        freeze.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        (root / "freeze.json").write_text(json.dumps(freeze))
    with pytest.raises(FixtureError):
        load_gold(root, dataset, public_root=resource_root())


def test_catalog_numeric_values_match_independent_business_contract(dataset: Dataset) -> None:
    rules = {r.rule_id: r for r in dataset.rules}
    expected = {
        "BILL-01": "days:0..14;minor:1..5000;first:true;credited:false;verified:true",
        "BILL-02": "5001..10000",
        "BILL-03": "minor>10000 OR credited OR not_first OR outside_window",
        "BILL-EX1": "days:0..30;plan:Team;exception_id:BILL-EX1;amount_limits:unchanged",
    }
    for name, value in expected.items():
        assert rules[name].branches[0].conditions[0].value == value
    assert rules["BILL-EX1"].overrides == ("BILL-01",)
    assert (
        "only the ordinary 14-day window" in rules["BILL-EX1"].branches[0].conditions[0].statement
    )


def test_dev_eval_facts_and_queries_are_not_duplicates(dataset: Dataset) -> None:
    invoices = {i.invoice_id: i for i in dataset.invoices}

    def signature(ticket: object) -> object:
        from knowledge_ticket_agent.schemas import Ticket

        assert isinstance(ticket, Ticket)
        invoice = invoices.get(ticket.facts.invoice_id or "")
        return (
            ticket.customer_id,
            ticket.claimed_category,
            ticket.facts.requested_action,
            ticket.facts.request_scope,
            ticket.facts.exception_id,
            None
            if invoice is None
            else (
                invoice.purchased_at,
                invoice.amount_minor,
                invoice.first_purchase,
                invoice.already_credited,
            ),
        )

    dev = {signature(t) for t in dataset.tickets if t.split.value == "dev"}
    evaluation = {signature(t) for t in dataset.tickets if t.split.value == "eval"}
    assert dev.isdisjoint(evaluation)


def test_gold_parses_only_integrity_verified_buffer(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    import knowledge_ticket_agent.gold as module
    from knowledge_ticket_agent.loaders import read_text

    original = read_text
    reads: list[str] = []

    def observed(workspace: object, path: str) -> str:
        from agent_harness_lab import WorkspaceFiles

        assert isinstance(workspace, WorkspaceFiles)
        reads.append(path)
        return original(workspace, path)

    monkeypatch.setattr(module, "read_text", observed)
    result = load_gold(DEMO / "tests/gold/v1", dataset, public_root=resource_root())
    assert len(result.cases) == 36
    assert reads == ["business_gold.json"]
