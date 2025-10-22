"""ReAct-style autonomous agent for LLMath."""

from dataclasses import dataclass
import logging
from typing import Optional

from ..config import AgentConfig, GenerationConfig, ModelConfig, ReActConfig
from ..inference.deepseek import DeepSeekMathModel
from ..prompts.react_templates import build_react_system_prompt
from ..tools.registry import ToolRegistry, create_react_registry
from .parser import ReActOutputParser
from .state import ReActState, ReActStep

logger = logging.getLogger(__name__)


@dataclass
class ReActResult:
    """Result from a ReAct agent run."""

    question: str
    answer: str | None
    steps: list[ReActStep]
    iterations: int
    terminated_reason: str
    input_tokens: int = 0
    output_tokens: int = 0


class ReActAgent:
    """Autonomous ReAct agent that decides when to call tools."""

    def __init__(
        self,
        model: DeepSeekMathModel,
        tool_registry: ToolRegistry,
        config: Optional[ReActConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tools = tool_registry
        self.config = config or ReActConfig()
        self.parser = ReActOutputParser()
        self.system_prompt = system_prompt or build_react_system_prompt(
            self.tools.list_tools()
        )

    def run(self, question: str) -> ReActResult:
        """Run the ReAct loop for a single question."""
        state = ReActState(question=question)
        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(1, self.config.max_iterations + 1):
            context = state.build_context()
            logger.debug("ReAct iteration %s", iteration)
            gen_result = self.model.generate_step(
                context=context,
                system_prompt=self.system_prompt,
                max_new_tokens=self.config.max_tokens_per_step,
                stop_sequences=["</tool>", "</answer>"],
            )
            total_input_tokens += gen_result.input_tokens
            total_output_tokens += gen_result.output_tokens

            parsed = self.parser.parse(gen_result.text)
            step = ReActStep(
                thought=parsed.thought,
                tool=parsed.tool,
                answer=parsed.answer,
                raw_output=parsed.raw_text,
            )

            if parsed.tool:
                tool_result = self.tools.execute(parsed.tool)
                if tool_result.success:
                    observation = tool_result.output or ""
                else:
                    observation = f"[error] {tool_result.error}"
                step.observation = observation
                state.add_step(step)
                continue

            if parsed.answer:
                step.answer = parsed.answer
                state.add_step(step)
                return ReActResult(
                    question=question,
                    answer=parsed.answer,
                    steps=state.steps,
                    iterations=iteration,
                    terminated_reason="answer",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )

            step.observation = (
                "No <tool> or <answer> tag found. Respond with a tool call or final answer."
            )
            state.add_step(step)

        return ReActResult(
            question=question,
            answer=None,
            steps=state.steps,
            iterations=self.config.max_iterations,
            terminated_reason="max_iterations",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def __call__(self, question: str) -> str:
        """Convenience call returning only the final answer."""
        result = self.run(question)
        return result.answer or ""


def create_react_agent(
    retriever,
    model_config: Optional[ModelConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    agent_config: Optional[AgentConfig] = None,
    react_config: Optional[ReActConfig] = None,
) -> ReActAgent:
    """Factory for a configured ReAct agent."""
    model_config = model_config or ModelConfig()
    generation_config = generation_config or GenerationConfig()
    agent_config = agent_config or AgentConfig()
    react_config = react_config or ReActConfig()

    model = DeepSeekMathModel.from_config(model_config, generation_config)
    registry = create_react_registry(
        retriever=retriever,
        agent_config=agent_config,
        react_config=react_config,
    )

    return ReActAgent(model=model, tool_registry=registry, config=react_config)
