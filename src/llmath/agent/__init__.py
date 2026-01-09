"""Agent module for LLMath - end-to-end math reasoning agent."""

from typing import TYPE_CHECKING, Any

from .parser import ReActOutputParser
from .state import ReActState, ReActStep

# Heavy imports (peft, transformers) are lazy-loaded
if TYPE_CHECKING:
    from .math_agent import AgentResult, MathAgent, create_math_agent
    from .react_agent import ReActAgent, ReActResult, create_react_agent


def create_agent(
    retriever: Any,
    mode: str = "manual",
    model_config: Any = None,
    generation_config: Any = None,
    agent_config: Any = None,
    react_config: Any = None,
) -> Any:
    """Factory to create manual or autonomous agents."""
    if mode == "manual":
        from .math_agent import create_math_agent

        return create_math_agent(
            retriever=retriever,
            model_config=model_config,
            generation_config=generation_config,
            agent_config=agent_config,
        )
    if mode == "autonomous":
        from .react_agent import create_react_agent

        return create_react_agent(
            retriever=retriever,
            model_config=model_config,
            generation_config=generation_config,
            agent_config=agent_config,
            react_config=react_config,
        )
    raise ValueError(f"Unknown agent mode: {mode}")


__all__ = [
    "MathAgent",
    "AgentResult",
    "create_math_agent",
    "ReActAgent",
    "ReActResult",
    "ReActOutputParser",
    "ReActState",
    "ReActStep",
    "create_react_agent",
    "create_agent",
]


def __getattr__(name: str) -> Any:
    """Lazy import for heavy dependencies."""
    if name in ("MathAgent", "AgentResult", "create_math_agent"):
        from .math_agent import AgentResult, MathAgent, create_math_agent

        return {"MathAgent": MathAgent, "AgentResult": AgentResult, "create_math_agent": create_math_agent}[name]
    if name in ("ReActAgent", "ReActResult", "create_react_agent"):
        from .react_agent import ReActAgent, ReActResult, create_react_agent

        return {"ReActAgent": ReActAgent, "ReActResult": ReActResult, "create_react_agent": create_react_agent}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
