"""Retrieval module for LLMath - semantic search over mathematical corpora."""

from typing import TYPE_CHECKING

from .base import BaseRetriever, SearchResult

# Heavy imports (faiss, sentence-transformers) are lazy-loaded
if TYPE_CHECKING:
    from .faiss_retriever import NaturalProofsRetriever
    from .theorem_kb import TheoremKB, TheoremSnippet

__all__ = [
    "BaseRetriever",
    "SearchResult",
    "NaturalProofsRetriever",
    "TheoremKB",
    "TheoremSnippet",
]


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    if name == "NaturalProofsRetriever":
        from .faiss_retriever import NaturalProofsRetriever

        return NaturalProofsRetriever
    if name in ("TheoremKB", "TheoremSnippet"):
        from .theorem_kb import TheoremKB, TheoremSnippet

        if name == "TheoremKB":
            return TheoremKB
        return TheoremSnippet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
