"""Prompt templates for the math agent.

Contains the system header and instruction templates used to construct
prompts for the LLM.
"""

SYSTEM_HEADER = """You are a mathematical assistant. \
You receive a math question, some retrieved theorems/definitions, \
and symbolic tool outputs. \
Use them to write a rigorous, step-by-step solution.

"""

ANSWER_INSTRUCTIONS = """Write a self-contained, proof-style solution in your own words.
- You may refer to the retrieved items as [T1], [T2], etc., when you use them.
- Use the symbolic results [S1], [S2], etc., where they are relevant.
- Show the main logical steps that justify the result; do not just state the final formula.
- NEVER answer with only a heading or a stub like 'Theorems Used:' or 'From the NaturalProofs corpus:'.
- Do not include meta-instructions or template text from this prompt in your answer.
- Finish with one or two sentences summarizing the final conclusion.

Now give the final answer.
"""

BASELINE_PROMPT_TEMPLATE = """You are a careful mathematical assistant.
Answer the following question rigorously and step by step, \
but DO NOT reference any external tools.

QUESTION:
{question}
"""
