"""M1 static integrity checks only, not a ticket-processing business rules engine."""

import re
import unicodedata
from collections import Counter
from dataclasses import astuple

from knowledge_ticket_agent.loaders import utc
from knowledge_ticket_agent.schemas import Dataset, DocKind, Split, Status
from knowledge_ticket_agent.strict import require


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_dataset(data: Dataset) -> None:
    m = data.manifest
    require(m.synthetic is True and m.data_origin == "authored_fiction", "synthetic_label")
    require(m.schema_version == 1 and m.profile in {"seed", "full"}, "manifest_version")
    require(utc(m.as_of).isoformat() == "2031-04-15T12:00:00+00:00", "fixed_clock")
    require(bool(m.corpus_version) and bool(m.ruleset_version), "data_version")
    records = (
        *m.documents,
        *data.customers,
        *data.staff,
        *data.invoices,
        *data.tickets,
        *data.rules,
    )
    require(
        all(r.synthetic is True and r.data_origin == "authored_fiction" for r in records),
        "synthetic_label",
    )
    require(tuple(p.metadata for p in data.documents) == m.documents, "document_metadata_binding")
    docs = {p.metadata.ref: p for p in data.documents}
    customers = {c.customer_id: c for c in data.customers}
    invoices = {i.invoice_id: i for i in data.invoices}
    tickets = {t.ticket_id: t for t in data.tickets}
    rules = {r.rule_id: r for r in data.rules}
    for size, actual in (
        (len(data.documents), len(docs)),
        (len(data.customers), len(customers)),
        (len(data.invoices), len(invoices)),
        (len(data.tickets), len(tickets)),
        (len(data.rules), len(rules)),
        (len(data.staff), len({s.actor_id for s in data.staff})),
    ):
        require(size == actual, "duplicate_id")
    require(len({d.path for d in m.documents}) == len(m.documents), "duplicate_document_path")
    require(
        {d.path for d in m.documents} == {r.path for r in m.resources if r.path.endswith(".md")},
        "document_inventory",
    )
    require(
        {r.path for r in m.resources if not r.path.endswith(".md")}
        == {
            "customers.json",
            "staff.json",
            "invoices.json",
            "rules.json",
            "tickets/dev.json",
            "tickets/eval.json",
        },
        "data_inventory",
    )
    actual_counts = (
        len(docs),
        len(customers),
        len(data.staff),
        len(invoices),
        sum(t.split == Split.DEV for t in data.tickets),
        sum(t.split == Split.EVAL for t in data.tickets),
        len(rules),
    )
    require(astuple(m.counts) == actual_counts, "manifest_counts")
    expected_ids = {f"KT-D{i:02}" for i in range(1, 5 if m.profile == "seed" else 13)} | (
        set() if m.profile == "seed" else {f"KT-E{i:02}" for i in range(1, 25)}
    )
    require(set(tickets) == expected_ids, "ticket_id_set")
    kinds = Counter(d.kind for d in m.documents)
    if m.profile == "full":
        require(
            kinds
            == {
                DocKind.POLICY: 8,
                DocKind.LEGACY: 2,
                DocKind.EXCEPTION: 2,
                DocKind.CONFLICT: 2,
                DocKind.FAQ: 6,
            },
            "full_document_counts",
        )
        require(
            (len(customers), len(data.staff), actual_counts[4], actual_counts[5], len(rules))
            == (6, 3, 12, 24, 9),
            "full_data_counts",
        )
    else:
        require(
            (
                len(docs),
                len(customers),
                len(data.staff),
                actual_counts[4],
                actual_counts[5],
                len(rules),
            )
            == (2, 1, 1, 4, 0, 3),
            "seed_counts",
        )
    for d in m.documents:
        require(
            bool(re.fullmatch(r"[A-Z0-9-]+", d.doc_id))
            and bool(re.fullmatch(r"\d+\.\d+", d.version)),
            "document_id",
        )
        require(
            bool(d.title.strip()) and bool(d.policy_family) and bool(d.scope), "document_metadata"
        )
        start = utc(d.effective_from)
        end = utc(d.effective_to) if d.effective_to else None
        require(end is None or start < end, "effective_interval")
        require(len(set(d.supersedes)) == len(d.supersedes), "duplicate_supersedes")
        if d.kind == DocKind.LEGACY:
            require(end is not None and end <= utc(m.as_of), "legacy_active")
        else:
            require(
                start <= utc(m.as_of) and (end is None or utc(m.as_of) < end), "inactive_document"
            )
        for ref in d.supersedes:
            require(ref in docs and ref != d.ref, "supersedes_reference")
            old = docs[ref].metadata
            require(
                old.policy_family == d.policy_family and old.kind == DocKind.LEGACY,
                "supersedes_family",
            )
            require(
                old.effective_to is not None and utc(old.effective_to) <= start,
                "supersedes_overlap",
            )
        require(all(r in rules for r in d.rule_ids), "document_rule_reference")
        require(
            all(
                ref in docs
                and docs[ref].metadata.kind not in {DocKind.FAQ, DocKind.LEGACY, DocKind.CONFLICT}
                for ref in d.source_refs
            ),
            "source_reference",
        )
        require(d.kind != DocKind.FAQ or bool(d.source_refs), "faq_authority")
    replaced = {ref for d in m.documents for ref in d.supersedes}
    require(
        all(d.ref in replaced for d in m.documents if d.kind == DocKind.LEGACY), "orphan_legacy"
    )
    snapshots = {s.name: s.documents for s in m.snapshots}
    require(
        len(snapshots) == len(m.snapshots) and set(snapshots) == {"normal", "conflict"},
        "snapshot_names",
    )
    conflict_refs = {d.ref for d in m.documents if d.kind == DocKind.CONFLICT}
    normal_refs = set(docs) - conflict_refs
    require(
        set(snapshots["normal"]) == normal_refs and set(snapshots["conflict"]) == set(docs),
        "snapshot_isolation",
    )
    require(
        all(len(set(s.documents)) == len(s.documents) for s in m.snapshots), "snapshot_duplicates"
    )
    if conflict_refs:
        conflict = [docs[r].metadata for r in conflict_refs]
        require(
            len({d.policy_family for d in conflict}) == 1
            and all(not d.supersedes for d in conflict),
            "conflict_structure",
        )
    for c in data.customers:
        require(
            bool(re.fullmatch(r"SYN-CUST-\d{2}", c.customer_id))
            and bool(re.fullmatch(r"SYN-TENANT-\d{2}", c.tenant_id)),
            "identity_id",
        )
        require(c.display_name.startswith("Fictional "), "fictional_identity")
    for i in data.invoices:
        require(bool(re.fullmatch(r"SYN-INV-[DE][0-9]{2}", i.invoice_id)), "invoice_id")
        require(
            i.customer_id in customers and i.tenant_id == customers[i.customer_id].tenant_id,
            "invoice_relationship",
        )
        require(0 < i.amount_minor <= 1000000 and i.unit == "DEMO_CREDIT", "amount_units")
        require(utc(i.purchased_at) <= utc(m.as_of), "future_invoice")
    for t in data.tickets:
        require(
            t.customer_id in customers and t.tenant_id == customers[t.customer_id].tenant_id,
            "ticket_relationship",
        )
        require(t.facts.trusted_identity_ref == t.customer_id, "identity_reference")
        require(
            t.revision == 0 and t.status == Status.OPEN and t.last_action_id is None,
            "initial_state",
        )
        require(0 < len(t.subject) <= 160 and 0 < len(t.body) <= 4000, "ticket_text")
        require(utc(t.submitted_at) <= utc(m.as_of), "future_ticket")
        require(
            bool(
                re.fullmatch(
                    r"KT-D(?:0[1-9]|1[0-2])"
                    if t.split == Split.DEV
                    else r"KT-E(?:0[1-9]|1[0-9]|2[0-4])",
                    t.ticket_id,
                )
            ),
            "split_id",
        )
        if t.facts.invoice_id is not None:
            require(t.facts.invoice_id in invoices, "invoice_reference")
            require(invoices[t.facts.invoice_id].customer_id == t.customer_id, "invoice_customer")
        require(t.facts.exception_id in (None, "BILL-EX1"), "exception_reference")
        require(
            t.facts.requested_action
            in (None, "credit_guidance", "access_guidance", "export_guidance"),
            "requested_action",
        )
        require(t.facts.request_scope in (None, "own_workspace", "unknown"), "request_scope")
    require(len({normalized(t.body) for t in data.tickets}) == len(tickets), "duplicate_query")
    for actor in data.staff:
        require(bool(re.fullmatch(r"SYN-STAFF-\d{2}", actor.actor_id)), "staff_id")
        require(
            set(actor.allowed_ticket_ids) <= set(tickets)
            and len(set(actor.allowed_ticket_ids)) == len(actor.allowed_ticket_ids),
            "staff_ticket_scope",
        )
        require(
            bool(actor.allowed_actions)
            and set(actor.allowed_actions)
            <= {"resolved", "awaiting_information", "awaiting_review", "read"},
            "staff_action_scope",
        )
    for rule in data.rules:
        require(
            rule.version == m.ruleset_version
            and bool(rule.branches)
            and all(b.conditions for b in rule.branches)
            and bool(rule.scope),
            "rule_metadata",
        )
        require(
            set(rule.overrides) <= set(rules) and rule.rule_id not in rule.overrides,
            "rule_override",
        )
        for condition in (c for b in rule.branches for c in b.conditions):
            require(
                condition.operator in {"eq", "between", "le", "guard", "transition"},
                "condition_operator",
            )
            require(bool(condition.field) and bool(condition.value), "condition_metadata")
            require(condition.policy_ref in docs, "rule_policy_reference")
            policy = docs[condition.policy_ref]
            require(
                rule.rule_id in policy.metadata.rule_ids and policy.metadata.kind != DocKind.FAQ,
                "rule_policy_link",
            )
            sections = {s.section_id: s.body for s in policy.sections}
            require(condition.section_id in sections and bool(condition.statement), "rule_section")
            require(condition.statement in sections[condition.section_id], "rule_statement")
