"""Parser for ReAct model outputs."""

import re
from dataclasses import dataclass


@dataclass
class ReActParseResult:
    """Parsed ReAct output fields."""

    thought: str | None
    tool: str | None
    answer: str | None
    raw_text: str


class ReActOutputParser:
    """Extract ReAct tags from model output."""

    _think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _tool_re = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
    _answer_re = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def _extract(self, pattern: re.Pattern[str], text: str) -> str | None:
        matches = pattern.findall(text)
        if not matches:
            return None
        return matches[-1].strip()

    def parse(self, text: str) -> ReActParseResult:
        """Parse model output into thought/tool/answer."""
        return ReActParseResult(
            thought=self._extract(self._think_re, text),
            tool=self._extract(self._tool_re, text),
            answer=self._extract(self._answer_re, text),
            raw_text=text,
        )
