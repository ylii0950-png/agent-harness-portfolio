"""Product flow tests: actual source acquisition, exact approval, state and CLI."""

import json
from dataclasses import replace

import pytest
from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.agent_grading import grade_run
from knowledge_ticket_agent.business_demo import main, run_ticket, summarize
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category


@pytest.mark.parametrize(
    "ticket,approval,followups,snapshot,outcome,status,revision",
    (
        ("KT-D01", "approve", 6, "normal", "business_success", "resolved", 1),
        ("KT-D01", "approve", 0, "normal", "expected_denial", "open", 0),
        ("KT-D08", "refuse", 6, "normal", "permission_denial", "open", 0),
        ("KT-D05", "approve", 6, "normal", "business_success", "resolved", 1),
        ("KT-D06", "approve", 6, "normal", "business_success", "awaiting_review", 1),
        ("KT-D11", "approve", 6, "normal", "business_success", "awaiting_information", 1),
        ("KT-D12", "approve", 8, "conflict", "expected_denial", "open", 0),
    ),
)
def test_actual_product_flow(
    ticket: str,
    approval: str,
    followups: int,
    snapshot: str,
    outcome: str,
    status: str,
    revision: int,
) -> None:
    data = load_dataset()
    run = run_ticket(data, ticket, approval=approval, followups=followups, snapshot=snapshot)
    grade = grade_run(run)
    assert grade.issues == () and grade.outcome == outcome
    assert (grade.final_status, grade.final_revision) == (status, revision)
    assert grade.commits == revision and grade.handler_entries == revision
    assert grade.readback_verified is bool(revision)
    assert grade.parent_model_calls <= 12 and grade.parent_tool_calls <= 28
    assert grade.child_model_calls == 0
    report = summarize(run, "test")
    assert report["verification"] == "PASS"
    searches = [r for r in run.app.registry.records if r.call.name == "kb_search"]
    first = json.loads(searches[0].result.content)["result"]
    original = StrategyRetriever(data, config=strategy("A1"))
    query = next(t.body for t in data.tickets if t.ticket_id == ticket)
    expected = original.search(
        query, category=Category(run.app.category), snapshot=snapshot, top_k=3
    )
    assert [h["chunk_id"] for h in first["hits"]] == [h.chunk_id for h in expected.hits]
    assert "A1" in first["method"]
    if followups == 0:
        assert len(searches) == 1 and grade.agent_full_source_chunks < grade.required_policy_chunks
        assert report["evidence"]["status"] == "insufficient"
        assert not run.app.approval_calls and run.app.initial == run.app.workflow.store.snapshot
    elif snapshot == "normal":
        assert grade.agent_full_source_chunks == grade.required_policy_chunks
        assert report["evidence"]["status"] == "answerable"
        assert all(r.parent_ordinal is None for r in run.app.registry.records)


def test_missing_invoice_never_becomes_an_approved_credit() -> None:
    data = load_dataset()
    tickets = tuple(
        replace(t, facts=replace(t.facts, invoice_id=None)) if t.ticket_id == "KT-D01" else t
        for t in data.tickets
    )
    run = run_ticket(replace(data, tickets=tickets), "KT-D01", approval="approve")
    report = summarize(run, "missing")
    assert report["decision"] == "load_error"
    assert report["commits"] == report["handler_entries"] == 0
    assert not run.app.approval_calls and report["unchanged_state_verified"]


def test_cli_json_has_actual_traces_and_custom_default_refuses(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(("--ticket", "KT-D01", "--json"))
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 1
    row = reports[0]
    assert row["execution"] == "DETERMINISTIC_MOCK"
    assert row["approval"] == "approval_denied" and row["commits"] == 0
    assert row["trace"] and row["tool_records"] and row["business_audit"]
    assert row["after"]["revision"] == 0 and row["verification"] == "PASS"


def test_readable_three_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    main(())
    text = capsys.readouterr().out
    assert "[valid]" in text and "[insufficient]" in text and "[denied]" in text
    assert text.count("Verification: PASS") == 3
    assert "committed state read back" in text and "state verified unchanged" in text
    assert '"tool_records"' not in text


@pytest.mark.parametrize(
    "options,error",
    (
        (("--retrieval-variant", "R1"), "semantic_options_require_vnext_profile"),
        (("--model-directory", "unused"), "semantic_options_require_vnext_profile"),
        (("--pair-model-directory", "unused"), "semantic_options_require_vnext_profile"),
        (
            ("--retrieval-profile", "vnext", "--pair-model-directory", "unused"),
            "pair_model_directory_requires_R1",
        ),
    ),
)
def test_incompatible_retrieval_options_fail_before_any_demo(
    capsys: pytest.CaptureFixture[str], options: tuple[str, ...], error: str
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main((*options, "--json"))
    output = capsys.readouterr()
    assert stopped.value.code == 2 and error in output.err and not output.out


@pytest.mark.parametrize(
    "ticket,limit", (("KT-E01", 6), ("UNKNOWN", 6), ("KT-D01", -1), ("KT-D01", True))
)
def test_invalid_target_or_budget_fails_before_execution(ticket: str, limit: int) -> None:
    with pytest.raises(ValueError):
        run_ticket(load_dataset(), ticket, followups=limit)


def test_budget_stop_is_not_a_verified_business_result() -> None:
    from unittest.mock import patch

    from agent_harness_lab.runtime import RunLimits

    # Exercise the real Agent's budget stop, not a forged grader return value.
    with patch("knowledge_ticket_agent.business_demo.RunLimits", return_value=RunLimits(3, 28)):
        run = run_ticket(load_dataset(), "KT-D01", approval="approve")
    report = summarize(run, "limited")
    assert report["outcome"] == "budget_exhaustion"
    assert report["verification"] == "FAILED"
    assert report["commits"] == 0 and report["unchanged_state_verified"]


def test_explicit_cli_refusal_and_evidence_cap_override_scenarios(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(("--scenario", "valid", "--approval", "refuse", "--json"))
    row = json.loads(capsys.readouterr().out)[0]
    assert row["approval"] == "approval_denied" and row["handler_entries"] == row["commits"] == 0
    main(("--scenario", "valid", "--followups", "0", "--json"))
    row = json.loads(capsys.readouterr().out)[0]
    assert row["reason"] == "agent_evidence_missing" and row["approval"] == "not_requested"
    assert row["unchanged_state_verified"]


def test_missing_trusted_facts_cli_fails_closed_without_mutating_resources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = load_dataset()
    main(("--scenario", "missing-facts", "--json"))
    row = json.loads(capsys.readouterr().out)[0]
    assert row["decision"] == "load_error"
    assert row["approval"] == "not_requested"
    assert row["handler_entries"] == row["commits"] == 0
    assert row["unchanged_state_verified"] and row["verification"] == "PASS"
    assert load_dataset() == before
