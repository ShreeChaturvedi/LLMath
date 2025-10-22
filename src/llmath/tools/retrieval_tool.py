"""Tool wrapper for theorem retrieval."""

import logging

from .base import BaseTool, ToolResult
from ..retrieval.theorem_kb import TheoremKB

logger = logging.getLogger(__name__)


class RetrieveTool(BaseTool):
    """Retrieve relevant theorems from the knowledge base."""

    name = "retrieve"
    description = "Retrieve relevant theorems or definitions"

    def __init__(self, kb: TheoremKB, default_k: int = 3) -> None:
        self.kb = kb
        self.default_k = default_k

    def execute(self, query: str) -> ToolResult:
        """Retrieve theorem snippets for a query string."""
        try:
            theorems = self.kb.get_theorems(query, k=self.default_k)
            if not theorems:
                return ToolResult(success=True, output="No theorems found.")

            lines = []
            for i, t in enumerate(theorems, 1):
                lines.append(
                    f"[T{i}] (idx={t.idx}) (score={t.score:.3f}) {t.title}: {t.snippet}"
                )
            return ToolResult(success=True, output="\n".join(lines))
        except Exception as e:
            logger.exception("Retrieve tool failed.")
            return ToolResult(success=False, error=f"retrieve error: {e}")
