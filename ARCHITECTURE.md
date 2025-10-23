## Architecture

LLMath is a theorem-aware mathematical assistant built around three pillars:

1. Retrieval-augmented context (NaturalProofs + FAISS)
2. Symbolic tool use (SymPy)
3. A fine-tuned DeepSeek-Math model

The system supports both manual tool routing and an autonomous ReAct agent.

### High-level flow

```
User Question
   │
   ├─ Manual mode: retrieve → optional tools → single model generation
   │
   └─ Autonomous mode (ReAct):
         think → tool → observe → ... → answer
```

### Core modules

- `src/llmath/retrieval`: FAISS-backed NaturalProofs retrieval
- `src/llmath/tools`: SymPy tool implementations and registry
- `src/llmath/prompts`: prompt builders and ReAct templates
- `src/llmath/agent`: manual `MathAgent` and autonomous `ReActAgent`
- `src/llmath/inference`: model loading and generation wrappers
- `src/llmath/training`: SFT and ReAct data generation + training
- `src/llmath/evaluation`: baseline comparison and TheoremQA utilities

### Manual agent pipeline

1. Retrieve top-k theorems with `TheoremKB`.
2. Run optional SymPy tool commands.
3. Build a structured prompt with retrieval + tool outputs.
4. Generate a proof-style answer with DeepSeek-Math.

### ReAct agent loop

The ReAct agent uses the token protocol defined in `docs/react-protocol.md`:

1. `generate_step()` emits `<think>` and either `<tool>` or `<answer>`.
2. If a tool is emitted, the tool is executed and injected as `<observe>`.
3. The loop continues until `<answer>` or `max_iterations`.

### Training

Training supports two formats:

- SFT: prompt/response pairs from NaturalProofs
- ReAct: multi-turn traces with retrieval + tool calls

Tokenization uses the DeepSeek chat template for both formats.

### Evaluation

TheoremQA benchmarking compares:

- Baseline: model only
- RAG only: retrieval, no tools
- Manual: retrieval + user-specified tools
- Autonomous: full ReAct loop

See `docs/benchmarks.md` for details.
