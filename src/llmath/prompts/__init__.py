"""Prompts module for LLMath - prompt templates and orchestration."""

from .templates import SYSTEM_HEADER, ANSWER_INSTRUCTIONS, BASELINE_PROMPT_TEMPLATE
from .react_templates import (
    REACT_SYSTEM_PROMPT,
    build_react_system_prompt,
    build_react_user_prompt,
)
from .builder import build_math_prompt, build_baseline_prompt
from .orchestrator import ToolOrchestrator, OrchestratorResult, create_orchestrator

__all__ = [
    "SYSTEM_HEADER",
    "ANSWER_INSTRUCTIONS",
    "BASELINE_PROMPT_TEMPLATE",
    "REACT_SYSTEM_PROMPT",
    "build_react_system_prompt",
    "build_react_user_prompt",
    "build_math_prompt",
    "build_baseline_prompt",
    "ToolOrchestrator",
    "OrchestratorResult",
    "create_orchestrator",
]
