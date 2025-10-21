"""State management for ReAct-style agent traces."""

from dataclasses import dataclass, field


@dataclass
class ReActStep:
    """Single ReAct step with optional thought, tool, observation, or answer."""

    thought: str | None = None
    tool: str | None = None
    observation: str | None = None
    answer: str | None = None
    raw_output: str | None = None

    def format(self) -> str:
        """Format step as tagged text for the prompt."""
        parts: list[str] = []
        if self.thought is not None:
            parts.append(f"<think>{self.thought}</think>")
        if self.tool is not None:
            parts.append(f"<tool>{self.tool}</tool>")
        if self.observation is not None:
            parts.append(f"<observe>{self.observation}</observe>")
        if self.answer is not None:
            parts.append(f"<answer>{self.answer}</answer>")
        return "\n".join(parts)


@dataclass
class ReActState:
    """Track question and history of ReAct steps."""

    question: str
    steps: list[ReActStep] = field(default_factory=list)

    def add_step(self, step: ReActStep) -> None:
        """Append a step to the trace."""
        self.steps.append(step)

    def build_context(self) -> str:
        """Build the full prompt context for the current trace."""
        parts = [f"Question:\n{self.question.strip()}"]
        for step in self.steps:
            formatted = step.format()
            if formatted:
                parts.append(formatted)
        return "\n".join(parts).strip() + "\n"
