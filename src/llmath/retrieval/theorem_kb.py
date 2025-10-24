"""Theorem knowledge base wrapper.

Provides a higher-level interface over the retriever for working with
theorem snippets, including title extraction and text formatting.
"""

import logging
from dataclasses import dataclass

from ..config import AgentConfig
from .faiss_retriever import NaturalProofsRetriever

logger = logging.getLogger(__name__)


@dataclass
class TheoremSnippet:
    """A formatted theorem snippet from the knowledge base.

    Attributes:
        idx: Index in the underlying dataset.
        score: Similarity score from retrieval.
        title: Title or name of the theorem/definition.
        snippet: Truncated text for display.
        full_text: Complete text of the entry.
    """

    idx: int
    score: float
    title: str
    snippet: str
    full_text: str


class TheoremKB:
    """Knowledge base interface over NaturalProofs.

    Wraps a retriever to provide formatted theorem snippets with
    titles and truncated text for display in prompts.

    Attributes:
        retriever: The underlying NaturalProofsRetriever.
        config: Agent configuration with snippet settings.
        ds: Direct access to the dataset.
        title_field: Name of the field used for titles.
    """

    def __init__(
        self,
        retriever: NaturalProofsRetriever,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the theorem knowledge base.

        Args:
            retriever: NaturalProofsRetriever instance.
            config: Agent configuration. Uses defaults if None.
        """
        self.retriever = retriever
        self.config = config or AgentConfig()
        self.ds = retriever.ds
        self.title_field = self._pick_title_field()

        if self.title_field is None:
            logger.info("No title-like field found; using generic entry labels.")
        else:
            logger.info("Using title field: `%s`", self.title_field)

    def _pick_title_field(self) -> str | None:
        """Select a title field from available columns."""
        cols = list(self.ds.column_names)
        candidates = ("title", "theorem", "name", "source", "section", "chapter")
        for c in candidates:
            if c in cols:
                return c
        return None

    def _make_snippet(self, text: str) -> str:
        """Create a truncated snippet from full text.

        Args:
            text: Full text to truncate.

        Returns:
            Truncated text, ending with "..." if cut off.
        """
        max_chars = self.config.max_snippet_chars
        t = text.strip().replace("\\n", "\n")

        if len(t) <= max_chars:
            return t

        cut = t[:max_chars]
        # Try to cut at a word boundary
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]

        return cut + " ..."

    def get_theorems(
        self,
        question: str,
        k: int | None = None,
    ) -> list[TheoremSnippet]:
        """Retrieve top-k theorem snippets for a question.

        Args:
            question: The math question to find theorems for.
            k: Number of theorems to retrieve. Uses config default if None.

        Returns:
            List of TheoremSnippet objects.
        """
        k = k or self.config.default_k
        logger.debug("Retrieving %s theorems for question.", k)
        raw = self.retriever.search(question, k=k)

        results: list[TheoremSnippet] = []
        for r in raw:
            row = self.ds[r.idx]

            if self.title_field is not None:
                title = str(row[self.title_field])
            else:
                title = f"Entry {r.idx}"

            snippet = self._make_snippet(r.text)

            results.append(
                TheoremSnippet(
                    idx=r.idx,
                    score=r.score,
                    title=title,
                    snippet=snippet,
                    full_text=r.text,
                )
            )

        return results

    def get_theorem_by_idx(self, idx: int) -> TheoremSnippet:
        """Get a specific theorem by its index.

        Args:
            idx: Index of the theorem in the dataset.

        Returns:
            TheoremSnippet for the specified entry.
        """
        row = self.ds[idx]

        if self.title_field is not None:
            title = str(row[self.title_field])
        else:
            title = f"Entry {idx}"

        text_field = self.retriever.text_field
        full_text = str(row[text_field])
        snippet = self._make_snippet(full_text)

        return TheoremSnippet(
            idx=idx,
            score=1.0,  # Direct lookup, no similarity score
            title=title,
            snippet=snippet,
            full_text=full_text,
        )
