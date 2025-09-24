"""Prompts module for LLMath - prompt templates and orchestration."""

from .templates import SYSTEM_HEADER, ANSWER_INSTRUCTIONS, BASELINE_PROMPT_TEMPLATE
from .builder import build_math_prompt, build_baseline_prompt
from .orchestrator import ToolOrchestrator, OrchestratorResult, create_orchestrator

__all__ = [
    "SYSTEM_HEADER",
    "ANSWER_INSTRUCTIONS",
    "BASELINE_PROMPT_TEMPLATE",
    "build_math_prompt",
    "build_baseline_prompt",
    "ToolOrchestrator",
    "OrchestratorResult",
    "create_orchestrator",
]
