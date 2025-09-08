"""Abstract base classes for retrieval systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SearchResult:
    """A single search result from a retriever.

    Attributes:
        idx: Index of the result in the underlying dataset.
        score: Similarity score (higher is more relevant).
        text: The text content of the result.
        metadata: Optional additional metadata.
    """

    idx: int
    score: float
    text: str
    metadata: Optional[dict[str, Any]] = None


class BaseRetriever(ABC):
    """Abstract base class for retrieval systems.

    Subclasses must implement:
        - search(): Return top-k results for a query
        - get_row(): Access raw data by index
    """

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return top-k results matching the query.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by relevance.
        """
        pass

    @abstractmethod
    def get_row(self, idx: int) -> dict:
        """Access the raw dataset row by index.

        Args:
            idx: Index of the row to retrieve.

        Returns:
            Dictionary containing the raw row data.
        """
        pass

    def batch_search(
        self, queries: list[str], k: int = 5
    ) -> list[list[SearchResult]]:
        """Search for multiple queries.

        Default implementation calls search() for each query.
        Subclasses may override for batch optimization.

        Args:
            queries: List of query strings.
            k: Number of results per query.

        Returns:
            List of result lists, one per query.
        """
        return [self.search(query, k=k) for query in queries]
