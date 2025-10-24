"""Baseline model for comparison without tools or retrieval.

Provides a simple baseline that answers questions using only
the model's parametric knowledge, without RAG or SymPy tools.
"""

from ..inference.deepseek import DeepSeekMathModel

BASELINE_PROMPT_TEMPLATE = """You are a careful mathematical assistant.
Answer the following question rigorously and step by step, but DO NOT reference any external tools.

QUESTION:
{question}"""


class BaselineModel:
    """Baseline model wrapper for tool-free answers.

    Uses the same underlying model as the agent but without
    retrieval or symbolic computation support.

    Attributes:
        model: The DeepSeekMathModel for generation.
    """

    def __init__(self, model: DeepSeekMathModel) -> None:
        """Initialize the baseline model.

        Args:
            model: A configured DeepSeekMathModel instance.
        """
        self.model = model

    def answer(
        self,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """Generate an answer without tools or retrieval.

        Args:
            question: The mathematical question to answer.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more focused).

        Returns:
            The generated answer text.
        """
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=question.strip())
        return self.model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ).text

    def __call__(self, question: str, **kwargs) -> str:
        """Convenience method for answering questions.

        Args:
            question: The question to answer.
            **kwargs: Additional arguments for answer().

        Returns:
            The generated answer.
        """
        return self.answer(question, **kwargs)


def answer_without_tools(
    model: DeepSeekMathModel,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """Standalone function for baseline answers.

    Convenience function that creates a temporary BaselineModel
    and generates an answer.

    Args:
        model: The DeepSeekMathModel to use.
        question: The question to answer.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The generated answer text.
    """
    baseline = BaselineModel(model)
    return baseline.answer(question, max_new_tokens, temperature)
