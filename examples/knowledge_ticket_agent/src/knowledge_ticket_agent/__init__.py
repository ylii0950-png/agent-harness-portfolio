"""Optional, wholly fictional data contracts. No Agent or live model entry points."""

from knowledge_ticket_agent.loaders import load_dataset, resource_root
from knowledge_ticket_agent.schemas import Category, Dataset, DocKind, Split, Status
from knowledge_ticket_agent.strict import FixtureError

__all__ = [
    "Category",
    "Dataset",
    "DocKind",
    "FixtureError",
    "Split",
    "Status",
    "load_dataset",
    "resource_root",
]
