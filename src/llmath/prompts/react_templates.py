"""Prompt templates for ReAct-style tool use."""

REACT_SYSTEM_PROMPT = """You are an autonomous mathematical reasoning agent.
You must use the XML tags below to structure your output.

Available tags:
<think>reasoning</think>
<tool>tool_name: args</tool>
<observe>result</observe>
<answer>final answer</answer>

Rules:
- Use <think> to show your reasoning.
- If you need a tool, output exactly one <tool> tag and nothing else after it.
- The system will provide an <observe> tag with the tool result.
- Continue until you can provide a final answer inside <answer>.
- Do not output text outside of these tags.

Available tools: {tool_list}
"""


def build_react_system_prompt(tool_names: list[str]) -> str:
    """Build the system prompt with available tool names."""
    tool_list = ", ".join(tool_names) if tool_names else "none"
    return REACT_SYSTEM_PROMPT.format(tool_list=tool_list)


def build_react_user_prompt(question: str) -> str:
    """Build the user prompt for a question."""
    return f"Question:\n{question.strip()}\n"
