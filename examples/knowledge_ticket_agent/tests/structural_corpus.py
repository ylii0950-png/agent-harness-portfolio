"""Public corpus construction only; no query/label/Gold enters this builder."""

import hashlib
from dataclasses import asdict, replace
from typing import Any

from knowledge_ticket_agent.loaders import markdown
from knowledge_ticket_agent.schemas import Dataset, SectionSpec

from .vnext_public_evaluation import digest, source_corpus

VERSION = "lqd-structural-corpus-1.1.0"


def build(base: Dataset, fixture: dict[str, Any]) -> tuple[Dataset, list[str]]:
    data, refs = source_corpus(base, fixture)
    policies = []
    for policy in data.documents:
        if policy.metadata.ref not in refs:
            policies.append(policy)
            continue
        title = policy.metadata.title
        extra = (
            (
                "st_overview",
                "Service overview",
                f"{title} is a fictional service process. This introduction describes the purpose "
                f"and general context of {title}. It explains the service topic without stating "
                "an approving role, a permitted duration or required attachments.",
            ),
            (
                "st_routing",
                "Administrative routing",
                f"{title} requests follow a documented workflow. This overview mentions approval "
                f"conditions and supporting records for {title}, but supplies none of those "
                "conditions or records. The controlling Conditions and Service record "
                "sections provide the details.",
            ),
        )
        metadata = replace(
            policy.metadata,
            sections=(*policy.metadata.sections, *(SectionSpec(i, h) for i, h, _ in extra)),
        )
        text = policy.text + "".join(f"## {i} | {h}\n{body}\n\n" for i, h, body in extra)
        policies.append(markdown(metadata, text))
    raw = {p.metadata.path: p.text.encode() for p in policies}
    manifest = replace(
        data.manifest,
        corpus_version=fixture["version"] + ":" + VERSION,
        documents=tuple(p.metadata for p in policies),
        resources=tuple(
            replace(r, sha256=hashlib.sha256(raw[r.path]).hexdigest(), bytes=len(raw[r.path]))
            if r.path in raw
            else r
            for r in data.manifest.resources
        ),
    )
    return replace(
        data, manifest=manifest, manifest_sha256=digest(asdict(manifest)), documents=tuple(policies)
    ), refs
