"""Gradio web interface for the math agent.

Provides an interactive demo for the theorem-aware math assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import gradio as gr

from ..agent.math_agent import MathAgent
from ..agent.react_agent import ReActAgent
from ..config import LLMathConfig
from ..inference.deepseek import DeepSeekMathModel
from ..prompts.orchestrator import create_orchestrator
from ..prompts.react_templates import build_react_system_prompt
from ..retrieval import NaturalProofsRetriever
from ..tools.registry import create_react_registry


@dataclass
class AppResources:
    retriever: NaturalProofsRetriever
    model: DeepSeekMathModel
    manual_agent: MathAgent
    react_agent: ReActAgent


_resources: AppResources | None = None
_config: LLMathConfig | None = None

_THEOREM_LINE_RE = re.compile(
    r"\[T(?P<num>\d+)\]\s+\(idx=(?P<idx>\d+)\)\s+"
    r"\(score=(?P<score>[0-9.]+)\)\s+(?P<title>.*?):\s+(?P<snippet>.*)"
)


def _get_resources() -> AppResources:
    global _resources, _config

    if _resources is None:
        if _config is None:
            _config = LLMathConfig()

        retriever = NaturalProofsRetriever(rebuild_index=False)
        model = DeepSeekMathModel.from_config(
            _config.model,
            _config.generation,
        )
        orchestrator = create_orchestrator(retriever, _config.agent)
        manual_agent = MathAgent(orchestrator, model, _config.agent)

        registry = create_react_registry(
            retriever=retriever,
            agent_config=_config.agent,
            react_config=_config.react,
        )
        system_prompt = build_react_system_prompt(registry.list_tools())
        react_agent = ReActAgent(
            model=model,
            tool_registry=registry,
            config=_config.react,
            system_prompt=system_prompt,
        )

        _resources = AppResources(
            retriever=retriever,
            model=model,
            manual_agent=manual_agent,
            react_agent=react_agent,
        )

    return _resources


def _parse_retrieval_output(observation: str) -> list[dict]:
    theorems: list[dict] = []
    for line in observation.splitlines():
        match = _THEOREM_LINE_RE.match(line.strip())
        if match:
            theorems.append(
                {
                    "idx": int(match.group("idx")),
                    "score": float(match.group("score")),
                    "title": match.group("title"),
                    "snippet": match.group("snippet"),
                }
            )
    return theorems


def _format_theorem_table(theorems: list[dict]) -> str:
    if not theorems:
        return "### Retrieved theorems\n\n_None._"

    rows = []
    for i, t in enumerate(theorems, start=1):
        score_str = f"{t['score']:.3f}"
        title = str(t["title"]).replace("|", " ")
        snippet = str(t["snippet"]).replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        rows.append(f"<tr><td>T{i}</td><td>{score_str}</td><td>{title}</td><td>{snippet}</td></tr>")

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
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )

    return "### Retrieved theorems\n\n" + table_html


def _build_tool_calls_from_sympy(sympy_context: list[str]) -> list[dict]:
    calls: list[dict] = []
    for entry in sympy_context:
        if "->" in entry:
            tool, obs = entry.split("->", 1)
            calls.append({"tool": tool.strip(), "observation": obs.strip()})
        else:
            calls.append({"tool": entry.strip(), "observation": ""})
    return calls


def _format_tool_log(tool_calls: list[dict]) -> tuple[str, str]:
    if not tool_calls:
        return "### Tool execution log\n\n_None._", "### Verification\n\n_None._"

    lines = []
    verified = []
    for call in tool_calls:
        tool = call.get("tool", "")
        observation = call.get("observation", "")
        status = "verified"
        if observation.startswith("[error]") or observation.startswith("[sympy_error]"):
            status = "error"
        if tool.startswith("retrieve"):
            status = "retrieved"

        lines.append(f"- `{tool}` -> `{observation}` [{status}]")
        if status == "verified" and not tool.startswith("retrieve"):
            verified.append(tool)

    verification = "### Verification\n\n" + (
        "Verified by SymPy: " + ", ".join(verified) if verified else "None."
    )

    return "### Tool execution log\n\n" + "\n".join(lines), verification


def _format_trace(steps) -> str:
    if not steps:
        return "### Reasoning trace\n\n_None._"

    lines = []
    for step in steps:
        if step.thought:
            lines.append(f"<think>{step.thought}</think>")
        if step.tool:
            lines.append(f"<tool>{step.tool}</tool>")
        if step.observation:
            lines.append(f"<observe>{step.observation}</observe>")
        if step.answer:
            lines.append(f"<answer>{step.answer}</answer>")

    trace_block = "\n".join(lines)
    return "### Reasoning trace\n\n```\n" + trace_block + "\n```"


def _format_exports(answer: str, trace: str) -> tuple[str, str]:
    export_md = "## Answer\n\n" + answer.strip()
    if trace and "```" in trace:
        export_md += "\n\n## Trace\n\n" + trace
    export_latex = "$$\n" + answer.strip() + "\n$$"
    return export_md, export_latex


def math_agent_ui(
    mode: str,
    question: str,
    sympy_expression_hint: str,
) -> tuple[str, str, str, str, str, str, str]:
    question = (question or "").strip()
    hint = (sympy_expression_hint or "").strip()

    if not question:
        empty = "Please enter a question."
        return (
            "### Answer\n\n" + empty,
            "### Retrieved theorems\n\n_None._",
            "### Tool execution log\n\n_None._",
            "### Reasoning trace\n\n_None._",
            "### Verification\n\n_None._",
            "",
            "",
        )

    resources = _get_resources()

    if mode == "Autonomous":
        react_result = resources.react_agent.run(question)
        answer = react_result.answer or ""
        tool_calls = []
        retrieved = []
        for step in react_result.steps:
            if step.tool:
                tool_calls.append({"tool": step.tool, "observation": step.observation or ""})
            if step.tool and step.observation and step.tool.startswith("retrieve"):
                retrieved.extend(_parse_retrieval_output(step.observation))

        theorems_md = _format_theorem_table(retrieved)
        tool_log_md, verification_md = _format_tool_log(tool_calls)
        trace_md = _format_trace(react_result.steps)
    else:
        sympy_expressions = [hint] if hint else []
        manual_result = resources.manual_agent.run(
            question=question,
            sympy_expressions=sympy_expressions,
            k=5,
            max_new_tokens=256,
        )
        answer = manual_result.answer
        theorems = [
            {
                "idx": t.idx,
                "score": t.score,
                "title": t.title,
                "snippet": t.snippet,
            }
            for t in manual_result.theorems
        ]
        theorems_md = _format_theorem_table(theorems)
        tool_calls = _build_tool_calls_from_sympy(manual_result.sympy_context)
        tool_log_md, verification_md = _format_tool_log(tool_calls)
        trace_md = "### Reasoning trace\n\n_Manual mode does not emit a trace._"

    answer_md = "### Answer\n\n" + answer
    export_md, export_latex = _format_exports(answer, trace_md)

    return (
        answer_md,
        theorems_md,
        tool_log_md,
        trace_md,
        verification_md,
        export_md,
        export_latex,
    )


def create_demo(config: LLMathConfig | None = None) -> gr.Blocks:
    global _config
    if config is not None:
        _config = config

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap');

    :root {
        --bg: #f6f4ef;
        --panel: #ffffff;
        --ink: #1f2933;
        --accent: #2f5d62;
        --muted: #6b7280;
    }

    body, .gradio-container {
        font-family: 'Space Grotesk', sans-serif;
        background: radial-gradient(circle at top left, #fef3c7, #f6f4ef 45%, #e5e7eb 100%);
        color: var(--ink);
    }

    .llm-panel {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        background: var(--panel);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }
    """

    with gr.Blocks(title="LLMath ReAct Demo", css=css) as demo:
        gr.Markdown(
            """
            # LLMath ReAct Demo

            Autonomous mathematical reasoning with retrieval and SymPy tools.
            Select a mode, ask a question, and inspect the reasoning trace.
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                mode_selector = gr.Radio(
                    choices=["Manual", "Autonomous"],
                    value="Manual",
                    label="Mode",
                )
                question_input = gr.Textbox(
                    lines=4,
                    label="Math question",
                    placeholder=(
                        "Example: Prove that the derivative of x**2*sin(x) is "
                        "2*x*sin(x) + x**2*cos(x)."
                    ),
                )
                sympy_hint_input = gr.Textbox(
                    lines=2,
                    label="Optional SymPy expression hint (manual mode)",
                    placeholder="Example: diff: x**2*sin(x)",
                )
                run_button = gr.Button("Run")

                gr.Examples(
                    examples=[
                        [
                            "Manual",
                            "Prove that the derivative of x**2*sin(x) is 2*x*sin(x) + x**2*cos(x).",
                            "diff: x**2*sin(x)",
                        ],
                        [
                            "Manual",
                            "Compute the derivative of sin(x)*cos(x) and simplify your answer.",
                            "diff: sin(x)*cos(x)",
                        ],
                        [
                            "Manual",
                            "Solve the equation x**2 - 1 = 0 and justify your steps.",
                            "solve: x**2 - 1 = 0",
                        ],
                        [
                            "Manual",
                            "Find the indefinite integral of 2*x and explain your reasoning.",
                            "integrate: 2*x",
                        ],
                        [
                            "Manual",
                            "Simplify (x**2 - 1) / (x - 1) and describe the step.",
                            "simplify: (x**2 - 1) / (x - 1)",
                        ],
                        [
                            "Manual",
                            "Differentiate x**3 + 3*x with respect to x.",
                            "diff: x**3 + 3*x",
                        ],
                        [
                            "Autonomous",
                            "Show that the sum of two continuous functions on R is continuous.",
                            "",
                        ],
                        [
                            "Autonomous",
                            "Prove that the derivative of x**2 is 2*x.",
                            "",
                        ],
                        [
                            "Autonomous",
                            "Explain why the product rule applies to f(x)g(x).",
                            "",
                        ],
                        [
                            "Autonomous",
                            "Evaluate the derivative of sin(x)*cos(x) using known rules.",
                            "",
                        ],
                        [
                            "Autonomous",
                            "Show that if f is differentiable then f is continuous.",
                            "",
                        ],
                        [
                            "Autonomous",
                            "Provide a proof sketch for the chain rule.",
                            "",
                        ],
                    ],
                    inputs=[mode_selector, question_input, sympy_hint_input],
                )

            with gr.Column(scale=7):
                with gr.Group(elem_classes="llm-panel"):
                    answer_output = gr.Markdown()
                with gr.Accordion("Retrieved theorems", open=False):
                    theorems_output = gr.Markdown()
                with gr.Accordion("Tool execution log", open=False):
                    tool_log_output = gr.Markdown()
                    verification_output = gr.Markdown()
                with gr.Accordion("Reasoning trace", open=False):
                    trace_output = gr.Markdown()
                with gr.Accordion("Exports", open=False):
                    export_md_output = gr.Textbox(
                        label="Trace + answer (Markdown)",
                        lines=6,
                        show_copy_button=True,
                    )
                    export_latex_output = gr.Textbox(
                        label="Answer (LaTeX)",
                        lines=4,
                        show_copy_button=True,
                    )

        run_button.click(
            fn=math_agent_ui,
            inputs=[mode_selector, question_input, sympy_hint_input],
            outputs=[
                answer_output,
                theorems_output,
                tool_log_output,
                trace_output,
                verification_output,
                export_md_output,
                export_latex_output,
            ],
        )

    return demo


def launch_demo(
    config: LLMathConfig | None = None,
    share: bool = False,
    port: int = 7860,
) -> None:
    demo = create_demo(config)
    demo.launch(share=share, server_port=port)
