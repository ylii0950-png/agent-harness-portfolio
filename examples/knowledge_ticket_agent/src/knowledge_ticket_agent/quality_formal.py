"""Prospective independent M2 input/scoring contract; never grants formal authority.

Only the existing restricted Operator may stamp a formal result after claiming a
run. Pure evaluation and every rehearsal always return formal NOT_RUN.
"""

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from knowledge_ticket_agent.evidence import EvidenceUnit, minimum_cover, score
from knowledge_ticket_agent.quality_evidence import AnswerSpec, resolve_policy
from knowledge_ticket_agent.retrieval_profiles import QUERY_VERSION, A1Retriever
from knowledge_ticket_agent.retrieval_strategies import strategy
from knowledge_ticket_agent.rules import validate_rule_contract
from knowledge_ticket_agent.schemas import Category, Dataset, Split
from knowledge_ticket_agent.strict import require

SUITE = "lqd-m2-independent-1.0.0"
CONTRACT = "lqd-policy-quality-formal-1.0.0"
DENOMINATORS = (12, 8, 8)


@dataclass(frozen=True)
class RankingInput:
    case_id: str
    query: str
    category: Category
    snapshot: str


@dataclass(frozen=True)
class SectionTarget:
    doc_ref: str
    section_id: str


@dataclass(frozen=True)
class DocumentaryInput:
    case_id: str
    query: str
    ticket_id: str
    category: Category
    intent: str
    sections: tuple[SectionTarget, ...]
    snapshot: str


@dataclass(frozen=True)
class QualityInputs:
    version: str
    data: Dataset
    ranking: tuple[RankingInput, ...]
    documentary: tuple[DocumentaryInput, ...]
    negative: tuple[DocumentaryInput, ...]


@dataclass(frozen=True)
class RankingGold:
    case_id: str
    units: tuple[EvidenceUnit, ...]


@dataclass(frozen=True)
class DecisionGold:
    case_id: str
    status: str


@dataclass(frozen=True)
class QualityGold:
    version: str
    contract: str
    suite: str
    retriever_config_sha256: str
    ranking: tuple[RankingGold, ...]
    documentary: tuple[DecisionGold, ...]
    negative: tuple[DecisionGold, ...]
    top_k: int = 3
    denominators: tuple[int, ...] = DENOMINATORS


def validate(inputs: QualityInputs, gold: QualityGold) -> None:
    validate_rule_contract(inputs.data)
    require(
        bool(inputs.version)
        and bool(gold.version)
        and gold.contract == CONTRACT
        and gold.suite == SUITE
        and gold.top_k == 3
        and gold.denominators == DENOMINATORS
        and gold.retriever_config_sha256 == strategy("A1").sha256,
        "M2_contract_version_or_config",
    )
    groups: tuple[tuple[RankingInput | DocumentaryInput, ...], ...] = (
        inputs.ranking,
        inputs.documentary,
        inputs.negative,
    )
    expected: tuple[tuple[RankingGold | DecisionGold, ...], ...] = (
        gold.ranking,
        gold.documentary,
        gold.negative,
    )
    require(tuple(len(g) for g in groups) == DENOMINATORS, "M2_denominators")
    ids = tuple(c.case_id for group in groups for c in group)
    require(
        len(set(ids)) == 28 and all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", i) for i in ids),
        "M2_case_ids",
    )
    for actual, oracle_group in zip(groups, expected, strict=True):
        require(
            tuple(c.case_id for c in actual) == tuple(c.case_id for c in oracle_group),
            "M2_Gold_case_binding",
        )
    require(
        all(c.status == "answerable" for c in gold.documentary)
        and all(c.status in {"insufficient", "not_assessable"} for c in gold.negative),
        "M2_expected_behavior",
    )
    retriever = A1Retriever(inputs.data)
    chunks = {c.chunk_id for c in retriever.index.chunks}
    sections = {(c.doc_ref, c.section_id) for c in retriever.index.chunks}
    for group in groups:
        for case in group:
            require(
                bool(case.query.strip())
                and len(case.query) <= 4000
                and case.snapshot in {"normal", "conflict"},
                "M2_query_or_snapshot",
            )
    for case in (*inputs.documentary, *inputs.negative):
        require(
            any(
                t.ticket_id == case.ticket_id and t.split is Split.DEV for t in inputs.data.tickets
            ),
            "M2_documentary_context",
        )
        require(
            len(case.sections) <= 16
            and len(set(case.sections)) == len(case.sections)
            and all(
                0 < len(s.doc_ref) <= 128 and 0 < len(s.section_id) <= 128 for s in case.sections
            )
            and (
                case in inputs.negative
                or bool(case.sections)
                and all((s.doc_ref, s.section_id) in sections for s in case.sections)
            ),
            "M2_section_contract",
        )
    for case, oracle in zip(inputs.ranking, gold.ranking, strict=True):
        require(case.category is not Category.OTHER and bool(oracle.units), "M2_positive_units")
        require(
            all(b and set(b) <= chunks for u in oracle.units for b in u.alternatives),
            "M2_Gold_chunk_binding",
        )
        require(minimum_cover(oracle.units) <= 3, "M2_top3_feasibility")


def evaluate(
    inputs: QualityInputs,
    gold: QualityGold,
    *,
    on_case: Callable[[int, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    validate(inputs, gold)
    retriever = A1Retriever(inputs.data)
    rows: list[dict[str, Any]] = []
    for group_name in ("ranking", "documentary", "negative"):
        for task, oracle in zip(
            getattr(inputs, group_name), getattr(gold, group_name), strict=True
        ):
            ordinal = len(rows) + 1
            if on_case:
                on_case(ordinal, None)
            row: dict[str, Any] = {"case_id": task.case_id, "kind": group_name}
            try:
                if group_name == "ranking":
                    found = retriever.search(
                        task.query, category=task.category, snapshot=task.snapshot, top_k=3
                    )
                    row.update(
                        status="observed",
                        metrics=asdict(score(oracle.units, tuple(h.chunk_id for h in found.hits))),
                    )
                else:
                    observed = resolve_policy(
                        inputs.data,
                        task.ticket_id,
                        task.query,
                        AnswerSpec(
                            task.category,
                            task.intent,
                            tuple((s.doc_ref, s.section_id) for s in task.sections),
                            task.snapshot,
                        ),
                        retriever=retriever,
                    )
                    safe = (
                        observed.approval_calls == observed.commits == 0
                        and observed.unchanged_state_verified
                    )
                    row.update(
                        status="observed",
                        observed=asdict(observed),
                        passed=observed.decision.status == oracle.status and safe,
                    )
            except Exception:
                row.update(status="error", passed=False, error="M2_case_execution_error")
            rows.append(row)
            if on_case:
                on_case(ordinal, row)
    recall = sum(r.get("metrics", {}).get("recall", 0.0) for r in rows[:12]) / 12
    mrr = sum(r.get("metrics", {}).get("reciprocal_rank", 0.0) for r in rows[:12]) / 12
    passed = (
        all(r["status"] == "observed" for r in rows)
        and recall >= 0.90
        and mrr >= 0.80
        and all(r["passed"] for r in rows[12:])
    )
    return {
        "contract": CONTRACT,
        "suite": SUITE,
        "input_version": inputs.version,
        "gold_version": gold.version,
        "denominators": DENOMINATORS,
        "top_k": 3,
        "query_handling": QUERY_VERSION,
        "recall_at_3": recall,
        "mrr_at_3": mrr,
        "documentary_passed": sum(r.get("passed", False) for r in rows[12:20]),
        "negative_passed": sum(r.get("passed", False) for r in rows[20:]),
        "engineering_contract": "PASS" if passed else "FAIL",
        "cases": rows,
        "scope": "PUBLIC_REHEARSAL_NOT_FORMAL",
        "revised_formal_quality": "NOT_RUN",
        "historical_M2": "FAIL_UNCHANGED",
    }
