"""Retrieval module for LLMath - semantic search over mathematical corpora."""

from .base import BaseRetriever, SearchResult
from .faiss_retriever import NaturalProofsRetriever

__all__ = ["BaseRetriever", "SearchResult", "NaturalProofsRetriever"]
