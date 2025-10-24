"""Agent module for LLMath - end-to-end math reasoning agent."""

from .math_agent import AgentResult, MathAgent, create_math_agent
from .parser import ReActOutputParser
from .react_agent import ReActAgent, ReActResult, create_react_agent
from .state import ReActState, ReActStep


def create_agent(
    retriever,
    mode: str = "manual",
    model_config=None,
    generation_config=None,
    agent_config=None,
    react_config=None,
):
    """Factory to create manual or autonomous agents."""
    if mode == "manual":
        return create_math_agent(
            retriever=retriever,
            model_config=model_config,
            generation_config=generation_config,
            agent_config=agent_config,
        )
    if mode == "autonomous":
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
