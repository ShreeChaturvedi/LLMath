"""Retrieval module for LLMath - semantic search over mathematical corpora."""

from .base import BaseRetriever, SearchResult
from .faiss_retriever import NaturalProofsRetriever
from .theorem_kb import TheoremKB, TheoremSnippet

__all__ = [
    "BaseRetriever",
    "SearchResult",
    "NaturalProofsRetriever",
    "TheoremKB",
    "TheoremSnippet",
]
