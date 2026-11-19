"""Public source-only builder; accepts no request, expected answer or oracle."""

import hashlib
import json
from dataclasses import asdict, replace
from typing import Any

from knowledge_ticket_agent.loaders import markdown
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind, SectionSpec

from .vnext_public_evaluation import digest

VERSION = "public-relevance-corpus-2.0.0"


def build(
    base: Dataset, corpus: dict[str, Any], *, replace_sections: bool = False
) -> tuple[Dataset, list[str]]:
    assert set(corpus) == {"version", "synthetic", "data_origin", "topics"}
    assert corpus["synthetic"] is True and corpus["data_origin"] == "authored_fiction"
    refs = sorted(d.ref for d in base.manifest.documents if d.kind == DocKind.POLICY)
    assert len(corpus["topics"]) == len(refs) == 8
    policies = []
    for policy in base.documents:
        metadata = policy.metadata
        sections = [(s.section_id, s.heading, s.body) for s in policy.sections]
        if metadata.ref in refs:
            topic = corpus["topics"][refs.index(metadata.ref)]
            metadata = replace(metadata, title=topic["title"])
            if replace_sections:
                # Retain M1 reference identifiers as non-normative QA anchors.
                sections = [
                    (i, h, "@note Schema anchor only; no domain assertion.") for i, h, _ in sections
                ]
            sections.extend((s["id"], s["heading"], s["body"]) for s in topic["sections"])
        if metadata.kind not in {DocKind.LEGACY, DocKind.CONFLICT}:
            metadata = replace(
                metadata, scope=tuple(dict.fromkeys((*metadata.scope, Category.OTHER)))
            )
        assert len({i for i, _, _ in sections}) == len(sections)
        metadata = replace(metadata, sections=tuple(SectionSpec(i, h) for i, h, _ in sections))
        text = f"# {metadata.title}\nSynthetic: true; data_origin: authored_fiction\n\n"
        text += "".join(f"## {i} | {h}\n{body}\n\n" for i, h, body in sections)
        policies.append(markdown(metadata, text))
    # The new QA corpus cannot execute M3: its manifest/source lock differs.
    # Retained M1 rule IDs are schema scaffolding, not operating assertions.
    rules = base.rules
    if replace_sections:
        rules = tuple(
            replace(
                rule,
                branches=tuple(
                    replace(
                        branch,
                        conditions=tuple(
                            replace(c, statement="@note Schema anchor only; no domain assertion.")
                            if c.policy_ref in refs
                            else c
                            for c in branch.conditions
                        ),
                    )
                    for branch in rule.branches
                ),
            )
            for rule in rules
        )
    raw = {p.metadata.path: p.text.encode() for p in policies}
    if replace_sections:
        raw["rules.json"] = json.dumps(
            {
                "synthetic": True,
                "data_origin": "authored_fiction",
                "rules": [asdict(r) for r in rules],
            },
            sort_keys=True,
        ).encode()
    manifest = replace(
        base.manifest,
        corpus_version=corpus["version"] + ":" + VERSION,
        documents=tuple(p.metadata for p in policies),
        resources=tuple(
            replace(r, sha256=hashlib.sha256(raw[r.path]).hexdigest(), bytes=len(raw[r.path]))
            if r.path in raw
            else r
            for r in base.manifest.resources
        ),
    )
    return replace(
        base,
        manifest=manifest,
        manifest_sha256=digest(asdict(manifest)),
        documents=tuple(policies),
        rules=rules,
    ), refs
