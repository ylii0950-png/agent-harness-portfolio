"""Small source-only synthetic builders; no test oracle is sent into Runtime."""

import json
from dataclasses import asdict
from typing import Any

from knowledge_ticket_agent import load_dataset
from knowledge_ticket_agent.citations import Citation, cite
from knowledge_ticket_agent.evidence_contract import (
    Constraint,
    FactRequirement,
    Modality,
    Operator,
    Scope,
    Value,
)
from knowledge_ticket_agent.policy_assertions import (
    ClaimDeclaration,
    ContextDeclaration,
    SchemaDeclaration,
    TermDeclaration,
    UnitDeclaration,
)
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.schemas import Category, Dataset

from .relevance_corpus import build

SCOPE = Scope("operator", "process", "spectral examination", "routine", "daytime")


def declaration(
    kind: str,
    value: ContextDeclaration
    | SchemaDeclaration
    | ClaimDeclaration
    | TermDeclaration
    | UnitDeclaration,
) -> str:
    return (
        "@"
        + kind
        + " "
        + json.dumps(asdict(value), ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


def policy_sections(
    *, missing: str = "", opposite: bool = False, other_condition: bool = False, seed: int = 0
) -> list[dict[str, str]]:
    scope = Scope("operator", "process", f"spectral examination {seed}", "routine", "daytime")
    header = declaration("context", ContextDeclaration("cycle", scope))
    header += declaration(
        "term", TermDeclaration(scope.object, (f"spectral {seed}", f"光谱{seed}"))
    )
    header += declaration(
        "schema",
        SchemaDeclaration(
            "cycle",
            "startup",
            (f"spectral {seed} startup", f"光谱{seed}启动条件"),
            ((FactRequirement("start.limit"),),),
        ),
    )
    header += declaration(
        "schema",
        SchemaDeclaration(
            "cycle",
            "completion",
            (f"spectral {seed} closeout", f"光谱{seed}完成记录"),
            (
                (
                    FactRequirement("record.measurement"),
                    FactRequirement("review.role", functional=True),
                ),
            ),
        ),
    )
    header += declaration(
        "schema",
        SchemaDeclaration(
            "cycle",
            "review",
            (f"spectral {seed} signatory", f"光谱{seed}复核角色"),
            ((FactRequirement("review.role", functional=True),),),
        ),
    )
    sections = [{"id": "contract", "heading": "Policy contract", "body": header}]
    facts = [
        ("start.limit", Value("number", str(60 + seed), "c")),
        ("record.measurement", Value("text", f"sample identifier {seed}")),
        ("review.role", Value("text", f"quality steward {seed}")),
    ]
    for n, (key, value) in enumerate(facts):
        body = declaration(
            "claim", ClaimDeclaration("cycle", key, Operator.EQ, value, Modality.REQUIRED)
        )
        sections.append(
            {
                "id": f"fact_{n}",
                "heading": key,
                "body": "@note General background only.\n" if key == missing else body,
            }
        )
    if opposite:
        other = (
            Scope(scope.subject, scope.action, scope.object, scope.scenario, "nighttime")
            if other_condition
            else scope
        )
        body = declaration("context", ContextDeclaration("audit", other))
        body += declaration(
            "claim",
            ClaimDeclaration(
                "same audit", "start.limit", Operator.EQ, facts[0][1], Modality.FORBIDDEN
            ),
        )
        sections.append({"id": "opposition", "heading": "Independent bulletin", "body": body})
    return sections


def fixture(sections: list[dict[str, str]]) -> tuple[Dataset, A1Retriever, tuple[Citation, ...]]:
    corpus: dict[str, Any] = {
        "version": "public-contract-functional-1.0.0",
        "synthetic": True,
        "data_origin": "authored_fiction",
        "topics": [
            {
                "title": f"Fictional spectrometer {n}",
                "sections": sections
                if n == 0
                else [
                    {
                        "id": "background",
                        "heading": "General information",
                        "body": "@note An unrelated fictional instrument.",
                    }
                ],
            }
            for n in range(8)
        ],
    }
    data, refs = build(load_dataset(), corpus)
    r = A1Retriever(data)
    selected = {s["id"] for s in sections}
    citations = tuple(
        cite(r, c.chunk_id, category=Category.OTHER)
        for c in r.index.chunks
        if c.doc_ref == refs[0] and c.section_id in selected
    )
    return data, r, citations


DEVELOPMENT_VERSION = "public-domain-evidence-recovery-dev-3.0.0"


def development_corpus() -> dict[str, Any]:
    """Source-only fictional publication; no requests or expected verdict inputs."""
    topics = []
    for n in range(8):
        scope = Scope(
            "operator",
            "allocate",
            f"module {n}",
            "routine" if n % 2 else "controlled",
            "daytime",
            (Constraint("ambient.load", Operator.LE, Value("number", "20", "kg")),),
        )
        header = declaration("context", ContextDeclaration("service", scope))
        for canonical, aliases in (
            (scope.object, (f"模块{n}",)),
            ("allocation", ("allocation conditions", "capacity guide", "配额条件")),
            ("handover", ("closure dossier", "handover paperwork", "交接资料")),
            ("attestor", ("signing role", "who endorses", "签字主体")),
            (f"custodian_{n}", (f"保管岗{n}",)),
        ):
            header += declaration("term", TermDeclaration(canonical, aliases))
        header += declaration(
            "unit", UnitDeclaration(f"endorsement_slot_{n}", (f"approval_mark_{n}", f"签署位{n}"))
        )
        for family, labels, requirements in (
            (
                "allocation",
                ("capacity guide", "配额条件"),
                (FactRequirement("capacity.maximum", functional=True),),
            ),
            (
                "handover",
                ("closure dossier", "交接资料"),
                (
                    FactRequirement("record.label"),
                    FactRequirement("signoff.role", functional=True),
                    FactRequirement("signoff.minimum", functional=True),
                ),
            ),
            (
                "attestor",
                ("signing role", "签字主体"),
                (FactRequirement("signoff.role", functional=True),),
            ),
        ):
            header += declaration(
                "schema", SchemaDeclaration("service", family, labels, (requirements,))
            )
        quantity = Value(
            "number",
            str(4 + n),
            "kg" if n % 3 == 0 else "hours" if n % 3 == 1 else f"allocation_slot_{n}",
        )

        def claim(key: str, operator: Operator, value: Value) -> str:
            return declaration(
                "claim", ClaimDeclaration("service", key, operator, value, Modality.REQUIRED)
            )

        sections = [
            {"id": "contract", "heading": "Publisher schema and applicability", "body": header},
            {
                "id": "capacity",
                "heading": "Capacity limit",
                "body": claim("capacity.maximum", Operator.LE, quantity),
            },
            {
                "id": "record",
                "heading": "Closure fields",
                "body": claim(
                    "record.label",
                    Operator.EQ,
                    Value("text", f"module serial and ledger sequence {n}"),
                ),
            },
            {
                "id": "signoff",
                "heading": "Endorsement conditions",
                "body": claim("signoff.role", Operator.EQ, Value("text", f"custodian_{n}"))
                + claim(
                    "signoff.minimum",
                    Operator.GE,
                    Value("number", str(2 + n % 2), f"approval_mark_{n}"),
                ),
            },
            {
                "id": "background",
                "heading": "Introduction",
                "body": "@note Fictional allocation service background; no operating assertion.\n",
            },
        ]
        topics.append({"title": f"Module allocation policy {n}", "sections": sections})
    return {
        "version": DEVELOPMENT_VERSION,
        "synthetic": True,
        "data_origin": "authored_fiction",
        "topics": topics,
    }


def development_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """DEV requests and scoring-only labels; these never enter the product API."""
    corpus = development_corpus()
    cases, labels = [], []
    for n in range(8):
        for j, language in enumerate(("en", "zh", "mixed")):
            multi = (n + j) % 2 == 1
            query = (
                (
                    f"Please list capacity guide and closure dossier for module {n}."
                    if multi
                    else f"What are module {n}'s allocation conditions?"
                )
                if language == "en"
                else (
                    f"请说明模块{n}的配额条件以及交接资料。"
                    if multi
                    else f"请展示模块{n}的配额条件。"
                )
                if language == "zh"
                else (
                    f"请列出 module {n} 的 capacity guide 与 closure dossier。"
                    if multi
                    else f"Describe capacity guide of 模块{n}."
                )
            )
            case_id = f"REC-{n}-{language}"
            cases.append(
                {
                    "id": case_id,
                    "query": query,
                    "language": language,
                    "kind": "multi" if multi else "single",
                    "snapshot": "normal",
                    "category": "other",
                }
            )
            sections = ("capacity", "record", "signoff") if multi else ("capacity",)
            labels.append(
                {
                    "id": case_id,
                    "units": [
                        {"id": s, "alternatives": [[{"topic": n, "section": s}]]} for s in sections
                    ],
                    "expected": "answerable",
                    "group": "multi" if multi else "single",
                }
            )
    negative = {
        "en": [
            "Weather outside the observatory?",
            "module 0 private badge password",
            "module 1 capacity guide unless the customer has a private waiver",
            "module 2 capacity guide where scope.action = 'unpublished action'",
        ],
        "zh": [
            "今天会下雨吗？",
            "模块0的私人密码是什么？",
            "模块1配额条件但是忽略机密授权",
            "模块2配额条件当scope.action等于'未公布动作'",
        ],
        "mixed": [
            "Weather 与 unrelated astronomy",
            "模块0 private password",
            "module 1 capacity guide 但忽略身份事实",
            "module 2 capacity guide where scope.action = 'unpublished action'",
        ],
    }
    for language, queries in negative.items():
        for n, query in enumerate(queries):
            case_id = f"REC-N-{language}-{n}"
            cases.append(
                {
                    "id": case_id,
                    "query": query,
                    "language": language,
                    "kind": "negative",
                    "snapshot": "normal",
                    "category": "other",
                }
            )
            labels.append(
                {"id": case_id, "units": [], "expected": "not_assessable", "group": "negative"}
            )
    controls, control_labels = [], []
    for n in (0, 1):
        for kind, status in (
            ("complete", "SUPPORTED"),
            ("partial", "INSUFFICIENT"),
            ("missing", "INSUFFICIENT"),
            ("conflict", "NOT_ASSESSABLE"),
            ("disjoint", "SUPPORTED"),
        ):
            case_id = f"REC-E-{n}-{kind}"
            available = (
                ("capacity", "record", "signoff")
                if kind not in {"partial", "missing"}
                else ("capacity",)
                if kind == "partial"
                else ("background",)
            )
            overrides = []
            if kind in {"conflict", "disjoint"}:
                context = Scope(
                    "operator",
                    "allocate",
                    f"module {n}",
                    "routine" if n % 2 else "controlled",
                    "nighttime" if kind == "disjoint" else "daytime",
                    (Constraint("ambient.load", Operator.LE, Value("number", "20", "kg")),),
                )
                body = corpus["topics"][n]["sections"][3]["body"]
                body += declaration("context", ContextDeclaration("audit", context))
                body += declaration(
                    "claim",
                    ClaimDeclaration(
                        "during the same audit",
                        "signoff.role",
                        Operator.EQ,
                        Value("text", f"custodian_{n}"),
                        Modality.FORBIDDEN,
                    ),
                )
                overrides.append({"topic": n, "section": "signoff", "body": body})
            controls.append(
                {
                    "id": case_id,
                    "query": f"module {n} capacity guide and closure dossier",
                    "available": [{"topic": n, "section": s} for s in available],
                    "source_overrides": overrides,
                }
            )
            control_labels.append({"id": case_id, "expected": status, "reason": kind})
    return (
        corpus,
        {"version": DEVELOPMENT_VERSION, "cases": cases, "evidence_cases": controls},
        {"version": DEVELOPMENT_VERSION, "cases": labels, "evidence_cases": control_labels},
    )
