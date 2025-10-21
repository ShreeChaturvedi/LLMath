"""Integration tests for the ReAct agent loop."""

from dataclasses import dataclass

from llmath.agent.react_agent import ReActAgent
from llmath.config import ReActConfig
from llmath.inference.deepseek import GenerationResult
from llmath.tools.registry import create_default_registry


@dataclass
class DummyModel:
    outputs: list[str]

    def generate_step(self, context, system_prompt, max_new_tokens=256, stop_sequences=None):
        text = self.outputs.pop(0)
        return GenerationResult(
            text=text,
            raw_text=text,
            input_tokens=0,
            output_tokens=0,
        )


def test_react_agent_tool_then_answer():
    outputs = [
        "<think>Use a tool.</think><tool>simplify: x + x</tool>",
        "<think>Done.</think><answer>2*x</answer>",
    ]
    model = DummyModel(outputs=outputs)
    registry = create_default_registry()
    agent = ReActAgent(
        model=model,
        tool_registry=registry,
        config=ReActConfig(max_iterations=3, max_tokens_per_step=32),
    )

    result = agent.run("Simplify x + x.")
    assert result.answer == "2*x"
    assert result.terminated_reason == "answer"
    assert len(result.steps) == 2
