"""Deterministic proof contracts; no semantic model, ranking scores or challenge."""

from dataclasses import replace

import pytest
from knowledge_ticket_agent.evidence_contract import (
    Constraint,
    EvidenceVerdict,
    Modality,
    Operator,
    Scope,
    Value,
    Verdict,
    compatible,
    implies,
    scope_entails,
)
from knowledge_ticket_agent.relevance_evidence import verify_support
from knowledge_ticket_agent.schemas import Category

from .evidence_contract_helpers import fixture, policy_sections


def check(
    sections: list[dict[str, str]], query: str = "spectral 0 startup and spectral 0 closeout"
) -> EvidenceVerdict:
    data, r, citations = fixture(sections)
    return verify_support(data, r, query, citations, category=Category.OTHER)


def test_real_full_source_binds_every_obligation_and_spans() -> None:
    verdict = check(policy_sections())
    assert verdict.status is Verdict.SUPPORTED
    assert len(verdict.bindings) == 3 and not verdict.missing and not verdict.unknown
    for b in verdict.bindings:
        assert b.claims and all(c.span.quote and len(c.span.quote_sha256) == 64 for c in b.claims)
    assert {b.requirement.key for b in verdict.bindings} == {
        "start.limit",
        "record.measurement",
        "review.role",
    }


@pytest.mark.parametrize("key", ["start.limit", "record.measurement", "review.role"])
def test_deleting_any_required_fact_is_insufficient(key: str) -> None:
    verdict = check(policy_sections(missing=key))
    assert verdict.status is Verdict.INSUFFICIENT
    assert any(key in keys for _, keys in verdict.missing)


def test_topic_background_is_not_fact_coverage() -> None:
    sections = policy_sections()
    for s in sections[1:]:
        s["body"] = "@note Spectral examination closeout and startup information.\n"
    verdict = check(sections)
    assert verdict.status is Verdict.INSUFFICIENT and not verdict.bindings


@pytest.mark.parametrize(
    "query",
    [
        "spectral 0 startup and spectral 0 closeout",
        "光谱0启动条件以及光谱0完成记录",
        "spectral 0 startup 与 光谱0完成记录",
    ],
)
def test_equivalent_language_requirements_produce_same_binding(query: str) -> None:
    verdict = check(policy_sections(), query)
    assert verdict.status is Verdict.SUPPORTED
    assert {b.requirement.key for b in verdict.bindings} == {
        "start.limit",
        "record.measurement",
        "review.role",
    }


def test_same_condition_opposition_and_different_condition_are_distinct() -> None:
    assert check(policy_sections(opposite=True)).status is Verdict.NOT_ASSESSABLE
    result = check(policy_sections(opposite=True, other_condition=True))
    assert result.status is Verdict.SUPPORTED and not result.conflicts


@pytest.mark.parametrize(
    "query",
    [
        "spectral 0 startup if temperature > 99 c",
        "spectral 0 startup unless operator is not authorized",
        "光谱0启动条件，但是不得忽略复核员",
        "spectral 0 startup and private password",
    ],
)
def test_unparsed_conditions_negations_roles_or_intents_never_pass(query: str) -> None:
    result = check(policy_sections(), query)
    assert result.status is Verdict.NOT_ASSESSABLE and result.unknown
    assert result.unknown[0].startswith("unparsed_query_constraints")


def test_canonical_source_tampering_and_partial_reads_are_rejected() -> None:
    data, r, citations = fixture(policy_sections())
    for forged in [
        replace(citations[0], quote="tampered"),
        replace(citations[0], index_sha256="0" * 64),
        replace(citations[0], start_char=citations[0].start_char + 1),
    ]:
        with pytest.raises(ValueError):
            verify_support(data, r, "spectral 0 startup", (forged,), category=Category.OTHER)
    with pytest.raises(ValueError):
        verify_support(data, r, "spectral 0 startup", citations, category=Category.BILLING)
    assert (
        verify_support(data, r, "spectral 0 startup", (), category=Category.OTHER).status
        is Verdict.INSUFFICIENT
    )


def test_unrecognized_source_or_unresolved_reference_maps_to_not_assessable() -> None:
    sections = policy_sections()
    sections[1]["body"] = "The owner must invent an unknown astronomical precondition.\n"
    assert check(sections).status is Verdict.NOT_ASSESSABLE
    sections = policy_sections()
    sections[1]["body"] = sections[1]["body"].replace('"cycle"', '"same absent context"')
    assert check(sections).status is Verdict.NOT_ASSESSABLE


def test_multi_source_one_obligation_and_one_source_many_obligations() -> None:
    sections = policy_sections()
    separate = check(sections)
    sections[1]["body"] += sections[2]["body"] + sections[3]["body"]
    sections[2]["body"] = sections[3]["body"] = "@note No normative assertion.\n"
    together = check(sections)
    assert separate.status is together.status is Verdict.SUPPORTED
    assert len({c.span.chunk_id for b in separate.bindings for c in b.claims}) == 3
    assert len({c.span.chunk_id for b in together.bindings for c in b.claims}) == 1


def test_property_values_units_and_irrelevant_source_changes_do_not_define_answers() -> None:
    for seed in range(10):
        sections = policy_sections(seed=seed)
        result = check(sections, f"spectral {seed} startup and spectral {seed} closeout")
        assert result.status is Verdict.SUPPORTED
        assert any(
            c.value.text == f"quality steward {seed}" for b in result.bindings for c in b.claims
        )
        for key in ("start.limit", "record.measurement", "review.role"):
            assert (
                check(
                    policy_sections(seed=seed, missing=key),
                    f"spectral {seed} startup and spectral {seed} closeout",
                ).status
                is Verdict.INSUFFICIENT
            )


def test_numeric_implication_requires_entire_condition_not_only_overlap() -> None:
    for threshold in range(1, 10):
        source = (Constraint("temperature", Operator.LE, Value("number", str(threshold), "c")),)
        smaller = (
            Constraint("temperature", Operator.EQ, Value("number", str(threshold - 1), "c")),
        )
        larger = (Constraint("temperature", Operator.EQ, Value("number", str(threshold + 1), "c")),)
        assert implies(smaller, source) is True
        assert implies(larger, source) is False
        assert implies((), source) is False
        assert compatible(smaller, larger) is False
    assert Value("number", "60", "min").canonical() == Value("number", "1", "h").canonical()
    a = Scope("operator", "start", "engine", "dry", "day")
    assert scope_entails(a, replace(a, time="night")) is False


def test_query_or_does_not_turn_into_and_or_let_unknown_hide() -> None:
    sections = policy_sections(missing="review.role")
    assert check(sections, "spectral 0 startup or spectral 0 closeout").status is Verdict.SUPPORTED
    assert (
        check(sections, "spectral 0 startup and spectral 0 closeout").status is Verdict.INSUFFICIENT
    )
    assert check(sections, "spectral 0 startup or secret password").status is Verdict.NOT_ASSESSABLE
    assert check(sections, "private password").status is Verdict.NOT_ASSESSABLE
    assert (
        check(
            sections, "(spectral 0 startup or spectral 0 closeout) and spectral 0 signatory"
        ).status
        is Verdict.INSUFFICIENT
    )


def test_schema_or_can_use_alternative_source_facts() -> None:
    from knowledge_ticket_agent.evidence_contract import FactRequirement
    from knowledge_ticket_agent.policy_assertions import SchemaDeclaration

    from .evidence_contract_helpers import declaration

    sections = policy_sections(missing="review.role")
    sections[0]["body"] += declaration(
        "schema",
        SchemaDeclaration(
            "cycle",
            "alternative",
            ("spectral 0 alternate proof",),
            ((FactRequirement("review.role"),), (FactRequirement("record.measurement"),)),
        ),
    )
    assert check(sections, "spectral 0 alternate proof").status is Verdict.SUPPORTED
    assert (
        check(policy_sections(missing="record.measurement"), "spectral 0 closeout").status
        is Verdict.INSUFFICIENT
    )


def test_ambiguous_same_context_is_unknown_instead_of_dropping_conditions() -> None:
    from knowledge_ticket_agent.policy_assertions import ContextDeclaration

    from .evidence_contract_helpers import SCOPE, declaration

    sections = policy_sections()
    sections[0]["body"] += declaration(
        "context", ContextDeclaration("other cycle", replace(SCOPE, time="nighttime"))
    )
    sections[1]["body"] = sections[1]["body"].replace('"cycle"', '"上述情况"')
    assert check(sections).status is Verdict.NOT_ASSESSABLE


def test_unread_claims_cannot_satisfy_requirements() -> None:
    data, r, citations = fixture(policy_sections())
    only_start = tuple(c for c in citations if c.section_id in {"contract", "fact_0"})
    result = verify_support(
        data, r, "spectral 0 startup and spectral 0 closeout", only_start, category=Category.OTHER
    )
    assert result.status is Verdict.INSUFFICIENT
    assert not any(b.requirement.key == "review.role" for b in result.bindings)


def test_explicit_source_aliases_mean_equivalent_rules_not_conflicting_values() -> None:
    from knowledge_ticket_agent.evidence_contract import Modality
    from knowledge_ticket_agent.policy_assertions import ClaimDeclaration, TermDeclaration

    from .evidence_contract_helpers import declaration

    sections = policy_sections()
    sections[0]["body"] += declaration(
        "term", TermDeclaration("reviewer", ("quality steward 0", "质量复核员"))
    )
    sections[3]["body"] += declaration(
        "claim",
        ClaimDeclaration(
            "cycle", "review.role", Operator.EQ, Value("text", "质量复核员"), Modality.REQUIRED
        ),
    )
    result = check(sections)
    assert result.status is Verdict.SUPPORTED and not result.conflicts
    assert all(
        c.value.text == "reviewer"
        for b in result.bindings
        if b.requirement.key == "review.role"
        for c in b.claims
    )


def test_expired_and_faq_sources_are_not_trusted_claims() -> None:
    from knowledge_ticket_agent.citations import cite
    from knowledge_ticket_agent.schemas import DocKind

    data, r, _ = fixture(policy_sections())
    faq = next(c for c in r.index.chunks if c.kind is DocKind.FAQ)
    with pytest.raises(ValueError):
        verify_support(
            data, r, "spectral 0 startup", (cite(r, faq.chunk_id),), category=Category.OTHER
        )
    legacy = next(c for c in r.index.chunks if c.kind is DocKind.LEGACY)
    with pytest.raises(ValueError):
        cite(r, legacy.chunk_id, category=Category.OTHER)


@pytest.mark.parametrize("compact", [False, True])
def test_actual_agent_full_reads_proof_and_no_write_capability(compact: bool) -> None:
    import json

    from knowledge_ticket_agent.agent_mock import scripted_calls
    from knowledge_ticket_agent.agent_tools import PolicyEvidenceSession
    from knowledge_ticket_agent.retrieval import lexical_rank
    from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever

    from agent_harness_lab.context import ContextPolicy
    from agent_harness_lab.providers import MockProvider, ModelResponse
    from agent_harness_lab.runtime import Agent, RunLimits

    from .test_retrieval_relevance import FakePair
    from .test_retrieval_vnext import EqualEncoder

    class LexicalPair(FakePair):
        def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]:
            return tuple(
                dict(lexical_rank((("doc", p),), q[:256])).get("doc", 0.0) for q, p in pairs
            )

    data, _, citations = fixture(policy_sections())
    r = RelevanceRetriever(data, EqualEncoder(), pair=LexicalPair())
    query = "spectral 0 startup and spectral 0 closeout"
    session = PolicyEvidenceSession(data, r, query, category=Category.OTHER)
    calls = [scripted_calls("search", (("kb_search", {"query": query}),))]
    for n, c in enumerate(c for c in citations if c.section_id != "contract"):
        calls.append(
            scripted_calls(
                f"read-{n}",
                (("kb_get", {"chunk_id": c.chunk_id}),),
            )
        )
    calls.extend((scripted_calls("proof", (("evidence_verify", {}),)), ModelResponse("DONE")))
    result = Agent(
        MockProvider(calls),
        session.registry,
        limits=RunLimits(10, 20),
        context_policy=ContextPolicy(30000, 9000, 1, 96) if compact else None,
    ).run(query)
    if compact:
        assert any(s.compacted_results > 0 for s in result.context_stats)
    from agent_harness_lab.contracts import ToolResult

    receipts = [
        json.loads(m.content)
        for m in result.messages
        if isinstance(m, ToolResult) and m.name == "evidence_verify" and m.ok
    ]
    assert receipts and receipts[-1]["status"] == "SUPPORTED"
    assert receipts[-1]["retrieval_method"] == r.method_identity
    assert set(session.registry.names) == {"kb_search", "kb_get", "evidence_verify"}
    assert len(session.reads) == 3
    repeat = Agent(
        MockProvider(
            [scripted_calls("new-run-proof", (("evidence_verify", {}),)), ModelResponse("DONE")]
        ),
        session.registry,
        limits=RunLimits(3, 3),
    ).run(query)
    again = [
        json.loads(m.content)
        for m in repeat.messages
        if isinstance(m, ToolResult) and m.name == "evidence_verify" and m.ok
    ]
    assert again[-1]["status"] == "INSUFFICIENT" and not session.reads


def test_forged_search_summary_or_model_claim_cannot_replace_full_reads() -> None:
    import json

    from knowledge_ticket_agent.agent_mock import scripted_calls
    from knowledge_ticket_agent.agent_tools import PolicyEvidenceSession
    from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever

    from agent_harness_lab.contracts import ToolCall, ToolResult
    from agent_harness_lab.providers import MockProvider, ModelResponse
    from agent_harness_lab.runtime import Agent, RunLimits

    from .test_retrieval_relevance import FakePair
    from .test_retrieval_vnext import EqualEncoder

    data, _, citations = fixture(policy_sections())
    session = PolicyEvidenceSession(
        data,
        RelevanceRetriever(data, EqualEncoder(), pair=FakePair()),
        "spectral 0 startup",
        category=Category.OTHER,
    )
    rejected = session.registry.invoke(
        ToolCall("fake", "kb_get", json.dumps({"chunk_id": citations[1].chunk_id}))
    )
    assert not rejected.ok and not session.reads
    result = Agent(
        MockProvider([scripted_calls("proof", (("evidence_verify", {}),)), ModelResponse("DONE")]),
        session.registry,
        limits=RunLimits(3, 3),
    ).run("Model says all facts exist")
    receipts = [
        json.loads(m.content)
        for m in result.messages
        if isinstance(m, ToolResult) and m.name == "evidence_verify" and m.ok
    ]
    assert receipts[-1]["status"] == "INSUFFICIENT" and not session.reads


def test_numeric_normalization_is_idempotent_and_never_rounds_conditions() -> None:
    for value in (
        Value("number", "1000000000000", "day"),
        Value("number", "0.000000001", "mm"),
        Value("number", "1.123456789012345678", "h"),
    ):
        assert value.canonical().canonical() == value.canonical()
    assert (
        compatible(
            (Constraint("size", Operator.EQ, Value("number", "1", "mm")),),
            (Constraint("size", Operator.NE, Value("number", "0.001", "m")),),
        )
        is False
    )
    with pytest.raises(ValueError):
        Value("number", "1e999999999", "h").canonical()


def test_source_json_duplicate_keys_fail_closed() -> None:
    sections = policy_sections()
    sections[1]["body"] = sections[1]["body"].replace(
        '"key":"start.limit"', '"key":"start.limit","key":"review.role"'
    )
    assert check(sections).status is Verdict.NOT_ASSESSABLE


@pytest.mark.parametrize(
    "query",
    [
        "What are the startup conditions for spectral 0 and completion records for spectral 0?",
        "请说明光谱0的启动条件以及完成记录",
        "who reviews spectral 0?",
        "光谱0由谁复核？",
    ],
)
def test_finite_natural_query_grammar_uses_entities_not_complete_question_templates(
    query: str,
) -> None:
    result = check(policy_sections(), query)
    assert result.status is Verdict.SUPPORTED


def test_long_query_preserves_unsupported_middle_condition() -> None:
    query = "Please " * 100 + "spectral 0 startup if temperature > 99 c and spectral 0 closeout"
    result = check(policy_sections(), query)
    assert result.status is Verdict.NOT_ASSESSABLE and any("99" in u for u in result.unknown)


def test_weaker_permission_or_negative_value_does_not_prove_required_fact() -> None:
    sections = policy_sections()
    sections[3]["body"] = sections[3]["body"].replace(
        '"modality":"required"', '"modality":"permitted"'
    )
    assert check(sections).status is Verdict.INSUFFICIENT
    sections = policy_sections()
    sections[3]["body"] = sections[3]["body"].replace('"operator":"eq"', '"operator":"ne"')
    assert check(sections).status is Verdict.INSUFFICIENT
    sections[3]["body"] = sections[3]["body"].replace('"operator":"ne"', '"operator":"gt"')
    assert check(sections).status is Verdict.NOT_ASSESSABLE


def test_unread_potential_counterclaim_prevents_premature_supported() -> None:
    data, r, citations = fixture(policy_sections(opposite=True))
    without_opposition = tuple(c for c in citations if c.section_id != "opposition")
    result = verify_support(
        data, r, "spectral 0 startup", without_opposition, category=Category.OTHER
    )
    assert result.status is Verdict.INSUFFICIENT
    assert any(name == "source_closure" for name, _ in result.missing)
    assert (
        verify_support(data, r, "spectral 0 startup", citations, category=Category.OTHER).status
        is Verdict.NOT_ASSESSABLE
    )


def test_deictic_reference_cannot_bind_a_future_context() -> None:
    sections = policy_sections()
    context, rest = sections[0]["body"].split("\n", 1)
    sections[0]["body"] = rest
    sections[1]["body"] = (
        sections[1]["body"].replace('"cycle"', '"during the same cycle"') + context + "\n"
    )
    assert check(sections).status is Verdict.NOT_ASSESSABLE


def test_impossible_conditions_never_prove_a_vacuous_obligation() -> None:
    import json

    sections = policy_sections()
    context, rest = sections[0]["body"].split("\n", 1)
    value = json.loads(context[9:])
    condition = {
        "field": "temperature",
        "operator": "eq",
        "value": {"kind": "number", "text": "30", "unit": "c"},
    }
    value["scope"]["conditions"] = [condition, {**condition, "operator": "ne"}]
    sections[0]["body"] = "@context " + json.dumps(value) + "\n" + rest
    assert check(sections).status is Verdict.NOT_ASSESSABLE


def test_source_scope_aliases_normalize_without_erasing_conditions() -> None:
    from knowledge_ticket_agent.policy_assertions import TermDeclaration

    from .evidence_contract_helpers import declaration

    sections = policy_sections(opposite=True)
    sections[0]["body"] += declaration(
        "term", TermDeclaration("operator", ("operator on duty", "当班操作员"))
    )
    sections[-1]["body"] = sections[-1]["body"].replace(
        '"subject":"operator"', '"subject":"当班操作员"'
    )
    assert check(sections).status is Verdict.NOT_ASSESSABLE
    sections[-1]["body"] = sections[-1]["body"].replace('"time":"daytime"', '"time":"nighttime"')
    assert check(sections).status is Verdict.SUPPORTED


def test_source_read_plan_is_metadata_bound_not_score_or_answer_coverage() -> None:
    from knowledge_ticket_agent.relevance_evidence import select_sources
    from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever

    from .test_retrieval_relevance import FakePair
    from .test_retrieval_vnext import EqualEncoder

    for seed in (0, 19, 87):
        data, _, _ = fixture(policy_sections(seed=seed, opposite=True))
        r = RelevanceRetriever(data, EqualEncoder(), pair=FakePair())
        query = f"spectral {seed} startup and spectral {seed} closeout"
        selection = select_sources(data, r, query, category=Category.OTHER)
        assert {h.section_id for h in selection.result.hits} == {"fact_0", "fact_1", "fact_2"}
        assert len(selection.read_dependencies) == 4  # Includes the opposite declaration.
        assert selection.result.method != r.method_identity
        from knowledge_ticket_agent.citations import cite

        citations = tuple(
            cite(r, h.chunk_id, category=Category.OTHER) for h in selection.result.hits
        )
        assert (
            verify_support(data, r, query, citations, category=Category.OTHER).status
            is Verdict.INSUFFICIENT
        )
        complete = tuple(cite(r, i, category=Category.OTHER) for i in selection.read_dependencies)
        assert (
            verify_support(data, r, query, complete, category=Category.OTHER).status
            is Verdict.NOT_ASSESSABLE
        )


def test_source_concepts_and_word_order_are_language_independent() -> None:
    from knowledge_ticket_agent.evidence_contract import FactRequirement
    from knowledge_ticket_agent.policy_assertions import SchemaDeclaration, TermDeclaration

    from .evidence_contract_helpers import declaration

    for seed in range(4):
        sections = policy_sections(seed=seed)
        lines = [
            line for line in sections[0]["body"].splitlines() if not line.startswith("@schema ")
        ]
        concept = f"packet_{seed}"
        sections[0]["body"] = (
            "\n".join(lines)
            + "\n"
            + declaration(
                "schema",
                SchemaDeclaration(
                    "cycle",
                    concept,
                    (f"handover bundle {seed}", f"移交资料{seed}"),
                    ((FactRequirement("record.measurement"), FactRequirement("review.role")),),
                ),
            )
            + declaration(
                "term", TermDeclaration(concept, (f"turnover forms {seed}", f"交接文件{seed}"))
            )
        )
        queries = [
            f"Please list turnover forms {seed} for spectral {seed}.",
            f"What are the handover bundle {seed} of spectral {seed}?",
            f"请列出光谱{seed}的交接文件{seed}。",
            f"请说明 turnover forms {seed} for 光谱{seed}。",
        ]
        results = [check(sections, q) for q in queries]
        assert all(v.status is Verdict.SUPPORTED for v in results)
        assert len({frozenset(b.requirement.key for b in v.bindings) for v in results}) == 1
        for field_name in ("record.measurement", "review.role"):
            modified = [dict(s) for s in sections]
            for section in modified[1:]:
                if '"key":"' + field_name + '"' in section["body"]:
                    section["body"] = "@note Background only.\n"
            assert all(check(modified, q).status is Verdict.INSUFFICIENT for q in queries)


def test_quantity_types_are_exact_and_nominal_units_do_not_disappear() -> None:
    pairs = [
        (Value("number", "2", "hours"), Value("number", "120", "min")),
        (Value("number", "1", "kg"), Value("number", "1000", "g")),
        (Value("number", "3", "metres"), Value("number", "300", "cm")),
    ]
    for a, b in pairs:
        assert a.canonical() == b.canonical()
    for n in range(12):
        a = Value("number", str(n), f"lot_{n}")
        assert a.canonical().canonical() == a.canonical()
        assert (
            compatible((Constraint("q", Operator.GE, a),), (Constraint("q", Operator.LT, a),))
            is False
        )
        assert (
            compatible(
                (Constraint("q", Operator.EQ, a),),
                (Constraint("q", Operator.EQ, Value("number", str(n), f"other_{n}")),),
            )
            is None
        )
    assert Value("number", "1", "M").canonical() != Value("number", "1", "m").canonical()


def test_publisher_units_bind_definitions_without_inventing_conversions() -> None:
    from knowledge_ticket_agent.policy_assertions import UnitDeclaration

    from .evidence_contract_helpers import declaration

    sections = policy_sections()
    sections[0]["body"] += declaration(
        "unit", UnitDeclaration("sample_slots", ("slots", "样本位", "sample slots"))
    )
    sections[1]["body"] = sections[1]["body"].replace('"unit":"c"', '"unit":"slots"')
    verdict = check(sections, "spectral 0 startup start.limit = 60 样本位")
    assert verdict.status is Verdict.SUPPORTED
    assert any(c.value.unit == "unit:sample_slots" for b in verdict.bindings for c in b.claims)
    assert (
        check(sections, "spectral 0 startup start.limit = 60 sample slots").status
        is Verdict.SUPPORTED
    )
    assert check(sections, "spectral 0 startup start.limit = 60 s").status is Verdict.NOT_ASSESSABLE
    sections[0]["body"] += declaration("unit", UnitDeclaration("kg", ("g",)))
    assert check(sections).status is Verdict.NOT_ASSESSABLE


def test_numeric_predicates_and_source_bounds_are_not_topic_matching() -> None:
    from knowledge_ticket_agent.policy_assertions import ClaimDeclaration

    from .evidence_contract_helpers import declaration

    for value in range(2, 7):
        sections = policy_sections()
        sections[1]["body"] = declaration(
            "claim",
            ClaimDeclaration(
                "cycle",
                "start.limit",
                Operator.LE,
                Value("number", str(value), "kg"),
                Modality.REQUIRED,
            ),
        )
        assert (
            check(sections, f"spectral 0 startup start.limit <= {value * 1000} g").status
            is Verdict.SUPPORTED
        )
        assert (
            check(sections, f"spectral 0 startup start.limit <= {value - 1} kg").status
            is Verdict.INSUFFICIENT
        )
        assert (
            check(sections, f"spectral 0 startup start.limit not > {value} kg").status
            is Verdict.SUPPORTED
        )
        assert (
            check(sections, f"spectral 0 startup start.limit == {value} M").status
            is Verdict.NOT_ASSESSABLE
        )


def test_joint_numeric_proof_requires_both_bounds_and_records_all_sources() -> None:
    from knowledge_ticket_agent.evidence_contract import FactRequirement
    from knowledge_ticket_agent.policy_assertions import ClaimDeclaration, SchemaDeclaration

    from .evidence_contract_helpers import declaration

    sections = policy_sections()
    sections[0]["body"] += declaration(
        "schema",
        SchemaDeclaration(
            "cycle",
            "quantity",
            ("exact quantity", "精确数量"),
            ((FactRequirement("quantity", functional=True),),),
        ),
    )
    for i, op in enumerate((Operator.GE, Operator.LE)):
        sections.append(
            {
                "id": f"bound_{i}",
                "heading": "Quantity bound",
                "body": declaration(
                    "claim",
                    ClaimDeclaration(
                        "cycle", "quantity", op, Value("number", "8", "cells"), Modality.REQUIRED
                    ),
                ),
            }
        )
    result = check(sections, "spectral 0 exact quantity quantity = 8 cells")
    assert result.status is Verdict.SUPPORTED
    assert len(result.bindings[0].claims) == 2
    assert (
        check(sections[:-1], "spectral 0 exact quantity quantity = 8 cells").status
        is Verdict.INSUFFICIENT
    )
    sections.append(
        {
            "id": "ban",
            "heading": "Prohibition",
            "body": declaration(
                "claim",
                ClaimDeclaration(
                    "cycle",
                    "quantity",
                    Operator.EQ,
                    Value("number", "8", "cells"),
                    Modality.FORBIDDEN,
                ),
            ),
        }
    )
    conflict = check(sections, "spectral 0 exact quantity")
    assert conflict.status is Verdict.NOT_ASSESSABLE
    assert any(c.additional for c in conflict.conflicts)


def test_prohibition_tightens_a_bound_instead_of_always_conflicting() -> None:
    from knowledge_ticket_agent.policy_assertions import ClaimDeclaration

    from .evidence_contract_helpers import declaration

    sections = policy_sections()
    sections[1]["body"] = declaration(
        "claim",
        ClaimDeclaration(
            "cycle", "start.limit", Operator.LE, Value("number", "10", "kg"), Modality.REQUIRED
        ),
    )
    sections.append(
        {
            "id": "extra",
            "heading": "Additional bound",
            "body": declaration(
                "claim",
                ClaimDeclaration(
                    "cycle",
                    "start.limit",
                    Operator.GE,
                    Value("number", "5", "kg"),
                    Modality.FORBIDDEN,
                ),
            ),
        }
    )
    result = check(sections, "spectral 0 startup")
    assert result.status is Verdict.SUPPORTED and not result.conflicts
    assert result.bindings[0].qualifiers[0].modality is Modality.FORBIDDEN
    sections[-1]["body"] = (
        sections[-1]["body"]
        .replace('"operator":"ge"', '"operator":"le"')
        .replace('"text":"5"', '"text":"10"')
    )
    assert check(sections, "spectral 0 startup").status is Verdict.NOT_ASSESSABLE


def test_or_coverage_does_not_require_an_unused_fact_family() -> None:
    data, r, citations = fixture(policy_sections())
    available = tuple(c for c in citations if c.section_id in {"contract", "fact_0"})
    assert (
        verify_support(
            data, r, "spectral 0 startup or spectral 0 closeout", available, category=Category.OTHER
        ).status
        is Verdict.SUPPORTED
    )
    assert (
        verify_support(
            data,
            r,
            "(spectral 0 startup or spectral 0 closeout) and spectral 0 signatory",
            available,
            category=Category.OTHER,
        ).status
        is Verdict.INSUFFICIENT
    )
    assert (
        verify_support(
            data, r, "spectral 0 startup or password", available, category=Category.OTHER
        ).status
        is Verdict.NOT_ASSESSABLE
    )


def test_conditions_actions_and_roles_are_requests_not_business_facts() -> None:
    from knowledge_ticket_agent.policy_assertions import ContextDeclaration, TermDeclaration

    from .evidence_contract_helpers import SCOPE, declaration

    sections = policy_sections()
    header = sections[0]["body"].splitlines()
    header[0] = declaration(
        "context",
        ContextDeclaration(
            "cycle",
            replace(
                SCOPE,
                object="spectral examination 0",
                conditions=(Constraint("load", Operator.LE, Value("number", "10", "kg")),),
            ),
        ),
    ).strip()
    sections[0]["body"] = (
        "\n".join(header) + "\n" + declaration("term", TermDeclaration("load", ("负载",)))
    )
    for query in ("spectral 0 startup where load = 5 kg", "光谱0启动条件当负载等于5000 g"):
        assert check(sections, query).status is Verdict.SUPPORTED
    assert check(sections, "spectral 0 startup where load = 11 kg").status is Verdict.INSUFFICIENT
    assert (
        check(sections, "spectral 0 startup where scope.action = 'other'").status
        is Verdict.INSUFFICIENT
    )
    assert check(sections, "spectral 0 startup where load != 5 kg").status is Verdict.INSUFFICIENT
    assert (
        check(sections, "spectral 0 startup unless the customer is approved").status
        is Verdict.NOT_ASSESSABLE
    )


def test_cache_still_requires_canonical_full_reads_in_the_current_proof() -> None:
    from knowledge_ticket_agent.policy_assertions import compile_catalog, compile_claims

    data, r, citations = fixture(policy_sections())
    catalog = compile_catalog(data, r, category=Category.OTHER)
    assert catalog is compile_catalog(data, r, category=Category.OTHER)
    compile_claims(data, r, citations, catalog, category=Category.OTHER)
    assert (
        verify_support(data, r, "spectral 0 startup", (), category=Category.OTHER).status
        is Verdict.INSUFFICIENT
    )
    with pytest.raises(ValueError):
        compile_claims(
            data, r, (replace(citations[1], quote="forged"),), catalog, category=Category.OTHER
        )
    with pytest.raises(ValueError):
        compile_claims(data, r, citations, replace(catalog, unknown=()), category=Category.OTHER)


def test_assertion_ranking_view_is_bounded_and_cannot_replace_source_reads() -> None:
    from knowledge_ticket_agent.relevance_evidence import select_sources
    from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever

    from .test_retrieval_relevance import FakePair
    from .test_retrieval_vnext import EqualEncoder

    class ObservedPair(FakePair):
        calls: list[tuple[tuple[str, str], ...]] = []

        def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float, ...]:
            self.calls.append(pairs)
            return super().score(pairs)

    data, _, _ = fixture(policy_sections())
    pair = ObservedPair()
    r = RelevanceRetriever(data, EqualEncoder(), pair=pair)
    query = "spectral 0 startup and spectral 0 closeout"
    selected = select_sources(data, r, query, category=Category.OTHER)
    assert selected.ranking.result.method != r.method_identity
    assert len(selected.ranking.trace.legal) == 3
    assert pair.calls and all(len(batch) == 3 for batch in pair.calls)
    assert all(
        "@schema " not in text and "@claim " not in text
        for batch in pair.calls
        for _, text in batch
    )
    assert any("quality steward 0" in text for batch in pair.calls for _, text in batch)
    assert (
        verify_support(data, r, query, (), category=Category.OTHER).status is Verdict.INSUFFICIENT
    )
    pair.calls.clear()
    unknown = select_sources(data, r, "unrelated galactic weather", category=Category.OTHER)
    assert unknown.result.status == "not_assessable" and not unknown.result.hits
    assert not pair.calls
    assert r.search("first purchase credit", category=Category.BILLING).method == r.method_identity


def test_ranking_view_cannot_expand_authoritative_source_scope() -> None:
    from knowledge_ticket_agent.retrieval_relevance import RelevanceRetriever, RetrievalView

    from .test_retrieval_relevance import FakePair
    from .test_retrieval_vnext import EqualEncoder

    data, _, citations = fixture(policy_sections())
    r = RelevanceRetriever(data, EqualEncoder(), pair=FakePair())
    with pytest.raises(ValueError):
        r.explain(
            "request",
            category=Category.BILLING,
            view=RetrievalView("request", ((citations[0].chunk_id, "forged ranking text"),)),
        )


def test_coordinated_entities_are_not_silently_erased_from_obligations() -> None:

    sections = policy_sections()
    second = policy_sections(seed=1)
    sections.extend(
        {**s, "id": "other_" + s["id"], "body": s["body"].replace('"cycle"', '"other_cycle"')}
        for s in second
    )
    data, r, citations = fixture(sections)
    only_first = tuple(c for c in citations if not c.section_id.startswith("other_"))
    for query in (
        "spectral 0 and spectral 1 startup",
        "startup for spectral 0 and spectral 1",
        "光谱0和光谱1启动条件",
    ):
        assert check(sections, query).status is Verdict.SUPPORTED
        assert (
            verify_support(data, r, query, only_first, category=Category.OTHER).status
            is Verdict.INSUFFICIENT
        )
    assert (
        verify_support(
            data, r, "spectral 0 or spectral 1 startup", only_first, category=Category.OTHER
        ).status
        is Verdict.SUPPORTED
    )
    assert check(sections, "spectral 0 startup and spectral 1").status is Verdict.NOT_ASSESSABLE
    assert check(sections, "spectral 0 startup; spectral 1 closeout").status is Verdict.SUPPORTED


def test_query_conjunction_keeps_every_numeric_and_categorical_constraint() -> None:
    from knowledge_ticket_agent.policy_assertions import ContextDeclaration

    from .evidence_contract_helpers import SCOPE, declaration

    sections = policy_sections()
    first, rest = sections[0]["body"].split("\n", 1)
    sections[0]["body"] = (
        declaration(
            "context",
            ContextDeclaration(
                "cycle",
                replace(
                    SCOPE,
                    object="spectral examination 0",
                    conditions=(
                        Constraint("load", Operator.GE, Value("number", "0", "kg")),
                        Constraint("load", Operator.LE, Value("number", "10", "kg")),
                    ),
                ),
            ),
        )
        + rest
    )
    assert (
        check(sections, "spectral 0 startup where load >= 5 kg and load <= 7 kg").status
        is Verdict.SUPPORTED
    )
    assert (
        check(sections, "spectral 0 startup where load != 5 kg and load = 5 kg").status
        is Verdict.NOT_ASSESSABLE
    )
    assert (
        check(
            sections, "spectral 0 startup where scope.action = 'other' and scope.action = 'process'"
        ).status
        is Verdict.NOT_ASSESSABLE
    )


def test_development_positive_negative_and_control_contract_without_models() -> None:
    import copy

    from knowledge_ticket_agent import load_dataset
    from knowledge_ticket_agent.citations import cite
    from knowledge_ticket_agent.policy_assertions import compile_catalog
    from knowledge_ticket_agent.policy_query import plan_obligations
    from knowledge_ticket_agent.retrieval_profiles import A1Retriever

    from .evidence_contract_helpers import development_inputs
    from .relevance_corpus import build

    corpus, requests, oracle = development_inputs()
    data, _ = build(load_dataset(), corpus, replace_sections=True)
    r = A1Retriever(data)
    catalog = compile_catalog(data, r, category=Category.OTHER)
    assert not catalog.unknown
    for case in requests["cases"]:
        plan = plan_obligations(case["query"], catalog)
        keys = {req.key for o in plan.obligations for bundle in o.alternatives for req in bundle}
        identifiers = {
            loc.chunk_id
            for loc in catalog.claim_locations
            if loc.key in keys
            and any(scope_entails(o.scope, loc.scope) is True for o in plan.obligations)
        }
        citations = tuple(
            cite(r, identifier, category=Category.OTHER) for identifier in sorted(identifiers)
        )
        verdict = verify_support(data, r, case["query"], citations, category=Category.OTHER)
        if case["kind"] == "negative":
            assert verdict.status is not Verdict.SUPPORTED, case["id"]
        else:
            assert verdict.status is Verdict.SUPPORTED, (
                case["id"],
                verdict.unknown,
                verdict.missing,
            )
    expected = {c["id"]: c for c in oracle["evidence_cases"]}
    for case in requests["evidence_cases"]:
        altered = copy.deepcopy(corpus)
        for change in case["source_overrides"]:
            next(
                s
                for s in altered["topics"][change["topic"]]["sections"]
                if s["id"] == change["section"]
            )["body"] = change["body"]
        data, refs = build(load_dataset(), altered, replace_sections=True)
        r = A1Retriever(data)
        citations = tuple(
            cite(
                r,
                next(
                    c.chunk_id
                    for c in r.index.chunks
                    if c.doc_ref == refs[source["topic"]] and c.section_id == source["section"]
                ),
                category=Category.OTHER,
            )
            for source in case["available"]
        )
        verdict = verify_support(data, r, case["query"], citations, category=Category.OTHER)
        assert verdict.status == expected[case["id"]]["expected"], (
            case["id"],
            verdict.unknown,
            verdict.missing,
        )
        assert not verdict.unknown
        if expected[case["id"]]["reason"] == "conflict":
            assert verdict.conflicts
        else:
            assert not verdict.conflicts


def test_query_cannot_retarget_a_publisher_goal_to_other_entity_facts() -> None:
    sections = policy_sections()
    sections.extend(
        {**s, "id": "other_" + s["id"], "body": s["body"].replace('"cycle"', '"other_cycle"')}
        for s in policy_sections(seed=1)
    )
    assert (
        check(sections, "spectral 0 startup where scope.object = 'spectral examination 1'").status
        is Verdict.INSUFFICIENT
    )
    assert check(sections, "spectral 0 startup; spectral 1 closeout").status is Verdict.SUPPORTED


def test_policy_scope_union_covers_a_case_only_without_gaps() -> None:
    from knowledge_ticket_agent.evidence_contract import scope_covered

    from .evidence_contract_helpers import SCOPE

    for boundary in range(1, 6):
        low = replace(
            SCOPE,
            conditions=(Constraint("load", Operator.LE, Value("number", str(boundary), "kg")),),
        )
        high = replace(
            SCOPE,
            conditions=(Constraint("load", Operator.GT, Value("number", str(boundary), "kg")),),
        )
        assert scope_covered(SCOPE, (low, high)) is True
        assert scope_covered(SCOPE, (low,)) is False
        gap = replace(
            SCOPE,
            conditions=(Constraint("load", Operator.GT, Value("number", str(boundary + 1), "kg")),),
        )
        assert scope_covered(SCOPE, (low, gap)) is False
