"""Exact-source citation validity; never claims general semantic entailment."""

from dataclasses import dataclass

from knowledge_ticket_agent.chunking import build_index
from knowledge_ticket_agent.retrieval import Retriever
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind
from knowledge_ticket_agent.strict import require


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    index_sha256: str
    doc_ref: str
    section_id: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    quote: str


def cite(
    retriever: Retriever,
    chunk_id: str,
    *,
    snapshot: str = "normal",
    category: Category | None = None,
) -> Citation:
    chunk = retriever.get_chunk(chunk_id, snapshot=snapshot, category=category)
    return Citation(
        chunk_id,
        retriever.index.sha256,
        chunk.doc_ref,
        chunk.section_id,
        chunk.start_char,
        chunk.end_char,
        chunk.start_line,
        chunk.end_line,
        chunk.text,
    )


def verify_citation(
    citation: Citation,
    data: Dataset,
    retriever: Retriever,
    *,
    snapshot: str = "normal",
    category: Category | None = None,
    require_authority: bool = True,
) -> None:
    require(type(require_authority) is bool, "citation_purpose")
    require(
        citation.index_sha256 == retriever.index.sha256
        and data.manifest_sha256 == retriever.index.manifest_sha256,
        "citation_index",
    )
    require(build_index(data) == retriever.index, "citation_data_binding")
    chunk = retriever.get_chunk(citation.chunk_id, snapshot=snapshot, category=category)
    require(
        citation.doc_ref == chunk.doc_ref and citation.section_id == chunk.section_id,
        "citation_source",
    )
    require(not require_authority or chunk.kind != DocKind.FAQ, "faq_not_authority")
    require(
        type(citation.start_char) is int
        and type(citation.end_char) is int
        and chunk.start_char <= citation.start_char < citation.end_char <= chunk.end_char,
        "citation_span",
    )
    policy = next(d for d in data.documents if d.metadata.ref == chunk.doc_ref)
    require(
        citation.quote == policy.text[citation.start_char : citation.end_char], "citation_quote"
    )
    require(
        type(citation.start_line) is int
        and type(citation.end_line) is int
        and citation.start_line == policy.text.count("\n", 0, citation.start_char) + 1
        and citation.end_line == policy.text.count("\n", 0, citation.end_char - 1) + 1,
        "citation_lines",
    )
