"""Prompts module for LLMath - prompt templates and orchestration."""

from .builder import build_baseline_prompt, build_math_prompt
from .orchestrator import OrchestratorResult, ToolOrchestrator, create_orchestrator
from .react_templates import (
    REACT_SYSTEM_PROMPT,
    build_react_system_prompt,
    build_react_user_prompt,
)
from .templates import ANSWER_INSTRUCTIONS, BASELINE_PROMPT_TEMPLATE, SYSTEM_HEADER

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
