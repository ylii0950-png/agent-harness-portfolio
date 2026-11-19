"""Reproducible public DEV evidence diagnostics, never sealed retrieval scoring."""

import json
from dataclasses import asdict

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.business_evaluation import public_manifest
from knowledge_ticket_agent.evidence import score
from knowledge_ticket_agent.quality_evidence import AnswerSpec, resolve_policy
from knowledge_ticket_agent.retrieval_strategies import StrategyRetriever, strategy
from knowledge_ticket_agent.schemas import Category

# Trusted public explanation tasks. These are application obligations, not a
# model-inferred intent or facts. Ranking metrics below never select these sections.
EXPLANATIONS = (
    (
        "KT-D01",
        Category.BILLING,
        "Explain first purchase credit eligibility and exclusions",
        (("POL-BILL-01@1.0", "eligibility"), ("POL-BILL-02@1.0", "restrictions")),
    ),
    (
        "KT-D01",
        Category.BILLING,
        "Describe manual review and hard billing restrictions",
        (("POL-BILL-01@1.0", "manual"), ("POL-BILL-02@1.0", "restrictions")),
    ),
    (
        "KT-D01",
        Category.BILLING,
        "Explain the Team window and exclusions together",
        (("EX-BILL-TEAM@1.0", "team_window"), ("POL-BILL-02@1.0", "restrictions")),
    ),
    (
        "KT-D08",
        Category.ACCESS,
        "Describe verification and security hold policy",
        (("POL-ACCESS-01@1.0", "verification"), ("POL-ACCESS-02@1.0", "security")),
    ),
    (
        "KT-D08",
        Category.ACCESS,
        "Explain access verification and escalation routing",
        (("POL-ACCESS-01@1.0", "verification"), ("POL-ESCALATE-01@1.0", "routing")),
    ),
    (
        "KT-D10",
        Category.EXPORT,
        "Explain export scope and security hold restrictions",
        (("POL-EXPORT-01@1.0", "scope"), ("POL-ACCESS-02@1.0", "security")),
    ),
    (
        "KT-D10",
        Category.EXPORT,
        "Describe export missing requirements and escalation",
        (("POL-EXPORT-02@1.0", "missing"), ("POL-ESCALATE-01@1.0", "routing")),
    ),
    (
        "KT-D10",
        Category.EXPORT,
        "Explain export scope and missing information",
        (("POL-EXPORT-01@1.0", "scope"), ("POL-EXPORT-02@1.0", "missing")),
    ),
)


def main() -> None:
    data = load_dataset()
    ret = StrategyRetriever(data, config=strategy("A1"))
    metrics = []
    for case in public_manifest(data).cases[:12]:
        found = ret.search(
            case.task.query,
            category=Category(case.oracle.category),
            snapshot=case.task.snapshot,
            top_k=3,
        )
        metrics.append(score(case.retrieval_units, tuple(h.chunk_id for h in found.hits)))
    rows = []
    for ticket, category, query, sections in EXPLANATIONS:
        result = resolve_policy(
            data, ticket, query, AnswerSpec(category, "policy_explanation", sections), retriever=ret
        )
        rows.append(asdict(result))
    print(
        json.dumps(
            {
                "scope": "PUBLIC_DEV_NOT_FORMAL_NOT_BLIND",
                "initial_A1_top3_denominator": 12,
                "initial_A1_top3": {
                    field: sum(getattr(m, field) for m in metrics) / 12
                    for field in ("hit", "recall", "reciprocal_rank")
                },
                "specified_explanation_denominator": len(rows),
                "specified_explanations": rows,
                "answerable": sum(r["decision"]["status"] == "answerable" for r in rows),
                "formal_M2_acceptance": "FAIL_UNCHANGED",
                "formal_M5": "NOT_RUN",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
