"""Math agent for end-to-end theorem-aware reasoning.

Combines retrieval, symbolic tools, and LLM generation into
a single pipeline for answering mathematical questions.
"""

import logging
from dataclasses import dataclass

from ..config import AgentConfig, GenerationConfig, ModelConfig
from ..inference.deepseek import DeepSeekMathModel
from ..prompts.orchestrator import OrchestratorResult, ToolOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Complete result from the math agent.

    Contains all intermediate results plus the final answer.

    Attributes:
        question: The original question.
        theorems: List of retrieved theorem snippets.
        sympy_context: List of formatted SymPy results.
        prompt: The complete prompt sent to the model.
        answer: The generated answer from the model.
        raw_answer: The raw model output before cleaning.
        input_tokens: Number of input tokens.
        output_tokens: Number of generated tokens.
    """

    question: str
    theorems: list
    sympy_context: list[str]
    prompt: str
    answer: str
    raw_answer: str
    input_tokens: int
    output_tokens: int

    def __str__(self) -> str:
        """Format result as a readable string."""
        lines = [
            "=" * 60,
            "QUESTION:",
            self.question,
            "",
            "RETRIEVED THEOREMS:",
        ]
        if self.theorems:
            for i, t in enumerate(self.theorems, 1):
                lines.append(f"  [T{i}] {t.title} (score={t.score:.3f})")
        else:
            lines.append("  None")

        lines.extend(["", "SYMPY CONTEXT:"])
        if self.sympy_context:
            for ctx in self.sympy_context:
                lines.append(f"  {ctx}")
        else:
            lines.append("  None")

        lines.extend(["", "ANSWER:", self.answer, "=" * 60])
        return "\n".join(lines)


class MathAgent:
    """End-to-end theorem-aware math reasoning agent.

    Orchestrates retrieval, symbolic computation, and LLM generation
    to answer mathematical questions with proper citations.

    Attributes:
        orchestrator: Tool orchestrator for retrieval and SymPy.
        model: DeepSeekMathModel for generation.
        config: Agent configuration settings.
    """

    def __init__(
        self,
        orchestrator: ToolOrchestrator,
        model: DeepSeekMathModel,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the math agent.

        Args:
            orchestrator: Configured tool orchestrator.
            model: DeepSeekMathModel for generation.
            config: Optional agent configuration.
        """
        self.orchestrator = orchestrator
        self.model = model
        self.config = config or AgentConfig()

    def run(
        self,
        question: str,
        sympy_expressions: list[str] | None = None,
        k: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResult:
        """Run the full agent pipeline on a question.

        Steps:
        1. Retrieve relevant theorems from NaturalProofs.
        2. Execute SymPy tools for symbolic computation.
        3. Build a context-rich prompt.
        4. Generate a proof-style answer with the model.

        Args:
            question: The mathematical question to answer.
            sympy_expressions: Optional list of tool commands
                (e.g., ["diff: x**2*sin(x)", "solve: x**2 - 1"]).
            k: Number of theorems to retrieve (default from config).
            max_new_tokens: Override for generation length.
            temperature: Override for sampling temperature.

        Returns:
            AgentResult with all intermediate and final results.
        """
        sympy_expressions = sympy_expressions or []
        k = k or self.config.default_k

        # Step 1-3: Orchestrate retrieval and tools
        logger.debug("Running tool orchestration.")
        orch_result: OrchestratorResult = self.orchestrator.solve_with_tools(
            question=question,
            k=k,
            sympy_expressions=sympy_expressions,
        )

        # Step 4: Generate answer
        logger.debug("Generating answer with model.")
        gen_result = self.model.generate(
            prompt=orch_result.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return AgentResult(
            question=question,
            theorems=orch_result.theorems,
            sympy_context=orch_result.sympy_context,
            prompt=orch_result.prompt,
            answer=gen_result.text,
            raw_answer=gen_result.raw_text,
            input_tokens=gen_result.input_tokens,
            output_tokens=gen_result.output_tokens,
        )

    def __call__(
        self,
        question: str,
        sympy_expressions: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Convenience method returning just the answer.

        Args:
            question: The mathematical question.
            sympy_expressions: Optional SymPy commands.
            **kwargs: Additional arguments for run().

        Returns:
            The generated answer text.
        """
        result = self.run(question, sympy_expressions, **kwargs)
        return result.answer


def create_math_agent(
    retriever,
    model_config=None,
    generation_config=None,
    agent_config=None,
) -> MathAgent:
    """Factory function to create a configured MathAgent.

    Creates all necessary components (orchestrator, model) from
    a retriever and optional configurations.

    Args:
        retriever: NaturalProofsRetriever instance.
        model_config: Optional ModelConfig for loading.
        generation_config: Optional GenerationConfig for inference.
        agent_config: Optional AgentConfig for agent settings.

    Returns:
        Configured MathAgent ready for use.
    """
    from ..prompts.orchestrator import create_orchestrator

    model_config = model_config or ModelConfig()
    generation_config = generation_config or GenerationConfig()
    agent_config = agent_config or AgentConfig()

    orchestrator = create_orchestrator(retriever, agent_config)
    model = DeepSeekMathModel.from_config(model_config, generation_config)

    return MathAgent(orchestrator, model, agent_config)
