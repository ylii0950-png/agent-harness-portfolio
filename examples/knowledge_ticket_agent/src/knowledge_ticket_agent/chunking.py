"""Deterministic original-source spans; no query, Gold or ranking dependencies."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from knowledge_ticket_agent.loaders import markdown
from knowledge_ticket_agent.schemas import Dataset, DocKind, PolicyText
from knowledge_ticket_agent.strict import require
from knowledge_ticket_agent.validation import validate_dataset

CHUNKER_VERSION = "lqd-section-1.0.0"
MAX_WORDS = 200
OVERLAP_WORDS = 20
MAX_CHUNK_BYTES = 8192


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_ref: str
    section_id: str
    part: int
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    title: str
    heading: str
    text: str
    kind: DocKind
    source_path: str
    source_sha256: str


@dataclass(frozen=True)
class ChunkIndex:
    corpus_version: str
    manifest_sha256: str
    chunker_version: str
    chunks: tuple[Chunk, ...]
    sha256: str


def chunk_document(policy: PolicyText) -> tuple[Chunk, ...]:
    require(markdown(policy.metadata, policy.text) == policy, "section_integrity")
    digest = hashlib.sha256(policy.text.encode()).hexdigest()
    chunks: list[Chunk] = []
    for section in policy.sections:
        words = list(re.finditer(r"\S+", section.body))
        start = 0
        part = 1
        while start < len(words):
            end = min(start + MAX_WORDS, len(words))
            if end < len(words):
                boundaries = [
                    i + 1
                    for i in range(start + OVERLAP_WORDS, end)
                    if re.search(r"[.!?][\"')]*$", words[i][0])
                ]
                if boundaries:
                    end = boundaries[-1]
            first = section.start_char + words[start].start()
            last = section.start_char + words[end - 1].end()
            text = policy.text[first:last]
            require(len(text.encode()) <= MAX_CHUNK_BYTES, "chunk_size")
            identifier = (
                f"{CHUNKER_VERSION}:{policy.metadata.ref}:{section.section_id}:"
                f"part{part:02}:{digest}"
            )
            chunks.append(
                Chunk(
                    identifier,
                    policy.metadata.ref,
                    section.section_id,
                    part,
                    first,
                    last,
                    policy.text.count("\n", 0, first) + 1,
                    policy.text.count("\n", 0, last - 1) + 1,
                    policy.metadata.title,
                    section.heading,
                    text,
                    policy.metadata.kind,
                    policy.metadata.path,
                    digest,
                )
            )
            if end == len(words):
                break
            start = end - OVERLAP_WORDS
            part += 1
    return tuple(chunks)


def build_index(data: Dataset) -> ChunkIndex:
    validate_dataset(data)
    resources = {r.path: r for r in data.manifest.resources}
    chunks: list[Chunk] = []
    for policy in data.documents:  # Manifest-bound list, never directory discovery.
        encoded = policy.text.encode()
        resource = resources[policy.metadata.path]
        require(
            len(encoded) == resource.bytes
            and hashlib.sha256(encoded).hexdigest() == resource.sha256,
            "source_integrity",
        )
        chunks.extend(chunk_document(policy))
    ordered = tuple(sorted(chunks, key=lambda c: c.chunk_id))
    require(len({c.chunk_id for c in ordered}) == len(ordered), "duplicate_chunk")
    payload = {
        "corpus_version": data.manifest.corpus_version,
        "manifest_sha256": data.manifest_sha256,
        "chunker_version": CHUNKER_VERSION,
        "chunks": [asdict(c) for c in ordered],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return ChunkIndex(
        data.manifest.corpus_version, data.manifest_sha256, CHUNKER_VERSION, ordered, digest
    )
