from collections import Counter
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from knowledge_ticket_agent import FixtureError, load_dataset
from knowledge_ticket_agent.loaders import utc
from knowledge_ticket_agent.schemas import Customer, Dataset, DocKind, Invoice, Synthetic
from knowledge_ticket_agent.strict import decode, parse_json
from knowledge_ticket_agent.validation import validate_dataset

from .conftest import DEMO


def test_minimal_seed_precedes_full_corpus() -> None:
    seed = load_dataset(DEMO / "tests/seed")
    assert (len(seed.documents), len(seed.tickets), len(seed.customers), len(seed.rules)) == (
        2,
        4,
        1,
        3,
    )
    assert {t.split.value for t in seed.tickets} == {"dev"}


def test_full_counts_and_immutable_state(dataset: Dataset) -> None:
    assert Counter(d.metadata.kind for d in dataset.documents) == {
        DocKind.POLICY: 8,
        DocKind.LEGACY: 2,
        DocKind.EXCEPTION: 2,
        DocKind.CONFLICT: 2,
        DocKind.FAQ: 6,
    }
    assert (
        len(dataset.customers),
        len(dataset.staff),
        len(dataset.invoices),
        len(dataset.tickets),
        len(dataset.rules),
    ) == (6, 3, 23, 36, 9)
    with pytest.raises(FrozenInstanceError):
        dataset.tickets[0].__setattr__("revision", 1)


@pytest.mark.parametrize(
    "text",
    [
        '{"synthetic":true,"synthetic":false,"data_origin":"authored_fiction"}',
        '{"outer":{"x":1,"x":2}}',
        "NaN",
        "Infinity",
        "-Infinity",
        "{",
        "[] trailing",
    ],
)
def test_strict_json_rejects_duplicate_keys_and_invalid_constants(text: str) -> None:
    with pytest.raises(FixtureError):
        parse_json(text)


@pytest.mark.parametrize(
    "raw",
    [
        {"synthetic": True},
        {"synthetic": True, "data_origin": "authored_fiction", "extra": 1},
        {"synthetic": 1, "data_origin": "authored_fiction"},
        {"synthetic": "true", "data_origin": "authored_fiction"},
        {"synthetic": True, "data_origin": None},
        [],
        None,
        {"synthetic": True, "data_origin": "\ud800"},
        {"synthetic": True, "data_origin": "\x00"},
    ],
)
def test_schema_is_exact_and_never_coerces(raw: object) -> None:
    with pytest.raises(FixtureError):
        decode(Synthetic, raw)


@pytest.mark.parametrize("value", [True, 1.0, "5000", None])
def test_minor_units_are_strict_integers(dataset: Dataset, value: Any) -> None:
    from dataclasses import asdict

    raw = asdict(dataset.invoices[0])
    raw["amount_minor"] = value
    with pytest.raises(FixtureError):
        decode(Invoice, raw)


@pytest.mark.parametrize("value", ["administrator", "OWNER", "", 1])
def test_role_enum_is_strict(dataset: Dataset, value: Any) -> None:
    from dataclasses import asdict

    raw = asdict(dataset.customers[0])
    raw["role"] = value
    with pytest.raises(FixtureError):
        decode(Customer, raw)


@pytest.mark.parametrize(
    "value",
    [
        "2031-04-15T12:00:00+01:00",
        "2031-04-15",
        "2031-02-30T00:00:00Z",
        "2031-04-15T12:00:00",
        "2031-04-15T12:00:60Z",
    ],
)
def test_clock_is_explicit_valid_utc(value: str) -> None:
    with pytest.raises(FixtureError):
        utc(value)


@pytest.mark.parametrize(
    "kind",
    [
        "customer_tenant",
        "invoice_customer",
        "invoice_tenant",
        "invoice_id",
        "invoice_amount",
        "invoice_unit",
        "future_invoice",
        "ticket_customer",
        "ticket_tenant",
        "ticket_identity",
        "unknown_invoice",
        "wrong_invoice_owner",
        "future_ticket",
        "revision",
        "status",
        "ticket_id",
        "duplicate_ticket",
        "duplicate_query",
        "staff_target",
        "staff_action",
        "legacy_overlap",
        "legacy_missing_end",
        "supersedes_family",
        "missing_rule",
        "rule_statement",
        "snapshot_conflict",
        "synthetic",
        "fixed_clock",
    ],
)
def test_cross_record_integrity_is_not_just_json_parsing(dataset: Dataset, kind: str) -> None:
    m = dataset.manifest
    if kind == "customer_tenant":
        changed = replace(
            dataset,
            customers=(
                replace(dataset.customers[0], tenant_id="SYN-TENANT-99"),
                *dataset.customers[1:],
            ),
        )
    elif kind.startswith("invoice_") or kind == "future_invoice":
        field, value = {
            "invoice_customer": ("customer_id", "SYN-CUST-99"),
            "invoice_tenant": ("tenant_id", "SYN-TENANT-99"),
            "invoice_id": ("invoice_id", ""),
            "invoice_amount": ("amount_minor", 0),
            "invoice_unit": ("unit", "USD"),
            "future_invoice": ("purchased_at", "2032-01-01T00:00:00Z"),
        }[kind]
        changed = replace(
            dataset,
            invoices=(
                replace(dataset.invoices[0], **{field: cast(Any, value)}),
                *dataset.invoices[1:],
            ),
        )
    elif kind in {
        "ticket_customer",
        "ticket_tenant",
        "ticket_identity",
        "unknown_invoice",
        "wrong_invoice_owner",
        "future_ticket",
        "revision",
        "status",
        "ticket_id",
    }:
        t = dataset.tickets[0]
        if kind == "ticket_identity":
            t = replace(t, facts=replace(t.facts, trusted_identity_ref="SYN-CUST-99"))
        elif kind == "unknown_invoice":
            t = replace(t, facts=replace(t.facts, invoice_id="SYN-INV-UNKNOWN"))
        elif kind == "wrong_invoice_owner":
            t = replace(
                t,
                facts=replace(
                    t.facts,
                    invoice_id=next(
                        i.invoice_id for i in dataset.invoices if i.customer_id != t.customer_id
                    ),
                ),
            )
        else:
            field, value = {
                "ticket_customer": ("customer_id", "SYN-CUST-99"),
                "ticket_tenant": ("tenant_id", "SYN-TENANT-99"),
                "future_ticket": ("submitted_at", "2032-01-01T00:00:00Z"),
                "revision": ("revision", 1),
                "status": ("last_action_id", "premature"),
                "ticket_id": ("ticket_id", "KT-EZZ"),
            }[kind]
            t = replace(t, **{field: cast(Any, value)})
        changed = replace(dataset, tickets=(t, *dataset.tickets[1:]))
    elif kind == "duplicate_ticket":
        changed = replace(dataset, tickets=(*dataset.tickets[:-1], dataset.tickets[0]))
    elif kind == "duplicate_query":
        changed = replace(
            dataset,
            tickets=(
                replace(dataset.tickets[0], body=dataset.tickets[1].body),
                *dataset.tickets[1:],
            ),
        )
    elif kind.startswith("staff_"):
        person = (
            replace(dataset.staff[0], allowed_ticket_ids=("KT-X",))
            if kind == "staff_target"
            else replace(dataset.staff[0], allowed_actions=("pay",))
        )
        changed = replace(dataset, staff=(person, *dataset.staff[1:]))
    elif kind in {"legacy_overlap", "legacy_missing_end", "supersedes_family", "missing_rule"}:
        docs = list(m.documents)
        if kind.startswith("legacy_"):
            idx = next(i for i, d in enumerate(docs) if d.kind == DocKind.LEGACY)
            docs[idx] = replace(
                docs[idx],
                effective_to=None if kind == "legacy_missing_end" else "2031-01-02T00:00:00Z",
            )
        else:
            docs[0] = replace(
                docs[0],
                **(
                    {"policy_family": "wrong"}
                    if kind == "supersedes_family"
                    else {"rule_ids": ("UNKNOWN",)}
                ),
            )
        changed = replace(
            dataset,
            manifest=replace(m, documents=tuple(docs)),
            documents=tuple(
                replace(p, metadata=d) for p, d in zip(dataset.documents, docs, strict=True)
            ),
        )
    elif kind == "rule_statement":
        rule = dataset.rules[0]
        branch = rule.branches[0]
        rule = replace(
            rule,
            branches=(
                replace(
                    branch, conditions=(replace(branch.conditions[0], statement="fabricated"),)
                ),
            ),
        )
        changed = replace(dataset, rules=(rule, *dataset.rules[1:]))
    elif kind == "snapshot_conflict":
        changed = replace(
            dataset,
            manifest=replace(
                m,
                snapshots=(
                    replace(m.snapshots[0], documents=m.snapshots[1].documents),
                    m.snapshots[1],
                ),
            ),
        )
    elif kind == "synthetic":
        changed = replace(
            dataset,
            customers=(replace(dataset.customers[0], synthetic=False), *dataset.customers[1:]),
        )
    else:
        changed = replace(dataset, manifest=replace(m, as_of="2032-04-15T12:00:00Z"))
    with pytest.raises(FixtureError):
        validate_dataset(changed)


def test_current_and_old_versions_are_not_conflated(dataset: Dataset) -> None:
    old = [d.metadata for d in dataset.documents if d.metadata.kind == DocKind.LEGACY]
    assert len(old) == 2 and all(
        utc(d.effective_to or "") <= utc(dataset.manifest.as_of) for d in old
    )
    assert len({d.metadata.ref for d in dataset.documents}) == 20


def test_manifest_metadata_cannot_diverge_from_loaded_document(dataset: Dataset) -> None:
    manifest = replace(
        dataset.manifest,
        documents=(
            replace(dataset.manifest.documents[0], title="Changed metadata"),
            *dataset.manifest.documents[1:],
        ),
    )
    with pytest.raises(FixtureError, match="document_metadata_binding"):
        validate_dataset(replace(dataset, manifest=manifest))
