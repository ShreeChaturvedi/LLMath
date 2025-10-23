## ReAct Protocol

LLMath uses a strict XML-style token protocol for autonomous tool use.
The model must emit exactly one tag per step and follow this loop:

```
<think>reasoning</think>
<tool>tool_name: args</tool>
<observe>result</observe>
<answer>final answer</answer>
```

### Rules

- `think` is free-form reasoning.
- `tool` must be a single call using the format `tool_name: args`.
- `observe` is injected by the system with tool output.
- `answer` ends the loop; no additional tags are allowed afterward.
- Do not emit text outside of these tags.

### Tools

Built-in tools:

- `retrieve`: theorem retrieval from NaturalProofs
- `simplify`: SymPy simplify
- `solve`: SymPy solve
- `diff`: SymPy differentiate
- `integrate`: SymPy integrate

### Example

```
<think>Retrieve the relevant theorem.</think>
<tool>retrieve: derivative of product</tool>
<observe>[T1] (idx=123) (score=0.912) Product Rule: ...</observe>
<think>Apply the product rule and compute the derivative.</think>
<tool>diff: x**2*sin(x)</tool>
<observe>2*x*sin(x) + x**2*cos(x)</observe>
<answer>The derivative follows from the product rule and the computation above.</answer>
```
