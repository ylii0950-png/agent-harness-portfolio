"""Bounded, read-only public fixture loading through the existing file safety API."""

import hashlib
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import TypeVar

from agent_harness_lab import FileToolError, WorkspaceFiles
from knowledge_ticket_agent.schemas import (
    CustomersFile,
    Dataset,
    Document,
    InvoicesFile,
    Manifest,
    PolicyText,
    RulesFile,
    Section,
    StaffFile,
    TicketsFile,
)
from knowledge_ticket_agent.strict import FixtureError, decode, parse_json, require

T = TypeVar("T")
MAX_RESOURCES = 64
MAX_JSON_BYTES = 256 * 1024
MAX_DOCUMENT_BYTES = 32 * 1024


def resource_root() -> Path:
    """Installed wheel filesystem resources; zipped importers are not supported."""
    return Path(str(files("knowledge_ticket_agent").joinpath("resources/v1")))


def utc(value: str) -> datetime:
    require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value)), "utc_format")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise FixtureError("utc_date") from None


def inventory(root: Path) -> set[str]:
    require(root.is_dir() and not root.is_symlink(), "invalid_root")
    found: set[str] = set()
    entries = 0
    for item in root.rglob("*"):
        entries += 1
        require(entries <= 128 and not item.is_symlink(), "resource_boundary")
        if item.is_dir():
            continue
        require(item.is_file(), "resource_type")
        found.add(item.relative_to(root).as_posix())
        require(len(found) <= MAX_RESOURCES, "resource_count")
    return found


def read_text(workspace: WorkspaceFiles, path: str, limit: int = MAX_JSON_BYTES) -> str:
    try:
        return workspace.read_text(path, max_bytes=limit)
    except FileToolError:
        raise FixtureError("resource_read_failed") from None


def read_model(workspace: WorkspaceFiles, path: str, model: type[T]) -> T:
    return decode(model, parse_json(read_text(workspace, path)))


def markdown(metadata: Document, text: str) -> PolicyText:
    """Parse explicit document sections, never tokenize/chunk or execute content."""
    prefix = f"# {metadata.title}\nSynthetic: true; data_origin: authored_fiction\n\n"
    require(text.startswith(prefix), "markdown_header")
    matches = list(re.finditer(r"^## ([a-z][a-z0-9_-]{0,63}) \| ([^\n]+)\n", text, re.M))
    require(bool(matches) and matches[0].start() == len(prefix), "markdown_sections")
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].rstrip("\n")
        require(bool(body.strip()) and not re.search(r"^#+ ", body, re.M), "markdown_body")
        sections.append(Section(match[1], match[2], body, match.end(), match.end() + len(body)))
    require(len({s.section_id for s in sections}) == len(sections), "duplicate_section")
    require(
        tuple((s.section_id, s.heading) for s in sections)
        == tuple((s.section_id, s.heading) for s in metadata.sections),
        "section_manifest_mismatch",
    )
    return PolicyText(metadata, text, tuple(sections))


def load_dataset(root: Path | None = None) -> Dataset:
    selected = root if root is not None else resource_root()
    try:
        actual = inventory(selected)
        with WorkspaceFiles(selected, max_bytes=MAX_JSON_BYTES) as workspace:
            raw = read_text(workspace, "manifest.json")
            manifest = decode(Manifest, parse_json(raw))
            require(len(manifest.resources) < MAX_RESOURCES, "resource_count")
            paths = [r.path for r in manifest.resources]
            require(len(set(paths)) == len(paths), "duplicate_resource")
            require(actual == {"manifest.json", *paths}, "resource_inventory")
            payloads: dict[str, str] = {}
            for entry in manifest.resources:
                require(bool(re.fullmatch(r"[a-f0-9]{64}", entry.sha256)), "invalid_digest")
                limit = MAX_DOCUMENT_BYTES if entry.path.endswith(".md") else MAX_JSON_BYTES
                text = read_text(workspace, entry.path, limit)
                encoded = text.encode("utf-8")
                require(len(encoded) == entry.bytes, "resource_size_mismatch")
                require(hashlib.sha256(encoded).hexdigest() == entry.sha256, "resource_digest")
                payloads[entry.path] = text

            def record(path: str, model: type[T]) -> T:
                require(path in payloads, "missing_resource")
                return decode(model, parse_json(payloads[path]))

            customers = record("customers.json", CustomersFile)
            staff = record("staff.json", StaffFile)
            invoices = record("invoices.json", InvoicesFile)
            rules = record("rules.json", RulesFile)
            dev = record("tickets/dev.json", TicketsFile)
            evaluation = record("tickets/eval.json", TicketsFile)
            envelopes = (customers, staff, invoices, rules, dev, evaluation)
            require(
                all(e.synthetic is True and e.data_origin == "authored_fiction" for e in envelopes),
                "synthetic_label",
            )
            require(
                all(e.version == manifest.corpus_version for e in envelopes if e is not rules),
                "data_version",
            )
            require(rules.version == manifest.ruleset_version, "rules_version")
            documents = []
            for doc in manifest.documents:
                require(doc.path in payloads and doc.path.endswith(".md"), "document_resource")
                documents.append(markdown(doc, payloads[doc.path]))
            dataset = Dataset(
                manifest,
                hashlib.sha256(raw.encode()).hexdigest(),
                tuple(documents),
                customers.customers,
                staff.staff,
                invoices.invoices,
                dev.tickets + evaluation.tickets,
                rules.rules,
            )
            # Filename split is part of the contract, not merely a record label.
            require(
                all(t.split.value == "dev" for t in dev.tickets)
                and all(t.split.value == "eval" for t in evaluation.tickets),
                "split_file",
            )
            from knowledge_ticket_agent.validation import validate_dataset

            validate_dataset(dataset)
            return dataset
    except (OSError, UnicodeError, RecursionError):
        raise FixtureError("resource_unavailable") from None
