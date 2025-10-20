"""Gradio web interface for the math agent.

Provides an interactive demo for the theorem-aware math assistant.
"""

from typing import Optional

import gradio as gr

from ..config import LLMathConfig
from ..retrieval import NaturalProofsRetriever
from ..retrieval.theorem_kb import TheoremKB
from ..prompts.orchestrator import create_orchestrator
from ..inference.deepseek import DeepSeekMathModel
from ..agent import MathAgent


# Global agent instance (initialized on first request)
_agent: Optional[MathAgent] = None
_config: Optional[LLMathConfig] = None


def _get_agent() -> MathAgent:
    """Get or create the global agent instance."""
    global _agent, _config
    if _agent is None:
        if _config is None:
            _config = LLMathConfig()

        print("Initializing retriever...")
        retriever = NaturalProofsRetriever(rebuild_index=False)

        print("Initializing model...")
        model = DeepSeekMathModel.from_config(
            _config.model,
            _config.generation,
        )

        print("Creating orchestrator...")
        orchestrator = create_orchestrator(retriever, _config.agent)

        print("Creating agent...")
        _agent = MathAgent(orchestrator, model, _config.agent)

    return _agent


def math_agent_ui(
    question: str,
    sympy_expression_hint: str,
) -> tuple[str, str, str]:
    """Process a question and return formatted results.

    Args:
        question: The math question to answer.
        sympy_expression_hint: Optional SymPy command hint.

    Returns:
        Tuple of (answer_md, theorems_md, sympy_md) as Markdown strings.
    """
    question = (question or "").strip()
    hint = (sympy_expression_hint or "").strip()

    if not question:
        return (
            "### Agent answer\n\nPlease enter a question.",
            "### Retrieved theorem snippets\n\n_None._",
            "### SymPy tool outputs\n\n_None._",
        )

    sympy_expressions = [hint] if hint else []

    agent = _get_agent()
    result = agent.run(
        question=question,
        sympy_expressions=sympy_expressions,
        k=5,
        max_new_tokens=256,
    )

    # Format answer as Markdown
    answer_md = "### Agent answer\n\n" + result.answer

    # Format theorems as HTML table
    if result.theorems:
        row_html = []
        for i, t in enumerate(result.theorems, start=1):
            score_str = f"{t.score:.3f}"
            title = t.title.replace("|", " ")
            snippet = t.snippet.replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            row_html.append(
                f"<tr>"
                f"<td>T{i}</td>"
                f"<td>{score_str}</td>"
                f"<td>{title}</td>"
                f"<td>{snippet}</td>"
                f"</tr>"
            )

        header = "### Retrieved theorem snippets\n\n"
        table_html = (
            "<table style='width:100%; table-layout:fixed; border-collapse:collapse;'>"
            "<colgroup>"
            "<col style='width:8%'>"
            "<col style='width:12%'>"
            "<col style='width:30%'>"
            "<col style='width:50%'>"
            "</colgroup>"
            "<thead>"
            "<tr>"
            "<th style='text-align:left; border-bottom:1px solid #ccc;'>ID</th>"
            "<th style='text-align:left; border-bottom:1px solid #ccc;'>Score</th>"
            "<th style='text-align:left; border-bottom:1px solid #ccc;'>Title</th>"
            "<th style='text-align:left; border-bottom:1px solid #ccc;'>Snippet</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            + "".join(row_html)
            + "</tbody>"
            "</table>"
        )
        theorems_md = header + table_html
    else:
        theorems_md = "### Retrieved theorem snippets\n\n_None._"

    # Format SymPy context as bullet list
    if result.sympy_context:
        lines = "\n".join(f"- `{entry}`" for entry in result.sympy_context)
        sympy_md = "### SymPy tool outputs\n\n" + lines
    else:
        sympy_md = "### SymPy tool outputs\n\n_None._"

    return answer_md, theorems_md, sympy_md


def create_demo(config: Optional[LLMathConfig] = None) -> gr.Blocks:
    """Create the Gradio demo interface.

    Args:
        config: Optional configuration to use.

    Returns:
        Configured Gradio Blocks interface.
    """
    global _config
    if config is not None:
        _config = config

    with gr.Blocks(title="Theorem-Aware Math Assistant") as demo:
        gr.Markdown(
            """
            # Theorem-Aware Math Assistant

            Retrieval-augmented math agent using the NaturalProofs corpus,
            SymPy tools, and a fine-tuned DeepSeek-Math model.
            """
        )

        with gr.Column():
            question_input = gr.Textbox(
                lines=4,
                label="Math question",
                placeholder="Example: Prove that the derivative of x**2*sin(x) is 2*x*sin(x) + x**2*cos(x).",
            )
            sympy_hint_input = gr.Textbox(
                lines=2,
                label="Optional SymPy expression hint",
                placeholder="Example: diff: x**2*sin(x)  or  solve: x**2 - 1 = 0",
            )
            run_button = gr.Button("Run agent")

            gr.Examples(
                examples=[
                    [
                        "Prove that the derivative of x**2*sin(x) is 2*x*sin(x) + x**2*cos(x).",
                        "diff: x**2*sin(x)",
                    ],
                    [
                        "Compute the derivative of sin(x)*cos(x) and simplify your answer.",
                        "diff: sin(x)*cos(x)",
                    ],
                    [
                        "Solve the equation x**2 - 1 = 0 and justify your steps.",
                        "solve: x**2 - 1 = 0",
                    ],
                    [
                        "Show that the sum of two continuous functions on R is continuous.",
                        "",
                    ],
                ],
                inputs=[question_input, sympy_hint_input],
            )

            answer_output = gr.Markdown()
            theorems_output = gr.Markdown()
            sympy_output = gr.Markdown()

        run_button.click(
            fn=math_agent_ui,
            inputs=[question_input, sympy_hint_input],
            outputs=[answer_output, theorems_output, sympy_output],
        )

    return demo


def launch_demo(
    config: Optional[LLMathConfig] = None,
    share: bool = False,
    port: int = 7860,
) -> None:
    """Convenience function to create and launch the demo.

    Args:
        config: Optional configuration.
        share: Whether to create a public link.
        port: Port to run on.
    """
    demo = create_demo(config)
    demo.launch(share=share, server_port=port)
