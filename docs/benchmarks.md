## Benchmarks

LLMath evaluates autonomous tool use on TheoremQA (800 questions).

### Dataset

- Default dataset name: `theoremqa`
- Default split: `test`
- You may provide a local JSON/JSONL path via `--dataset-path`

Each example must include `question` and `answer`.
Optional fields:

- `sympy_expressions`: manual tool calls
- `theorem_ids` or `theorem_indices`: gold retrieval indices

### Metrics

- **Accuracy**: normalized exact/fuzzy match with gold answer
- **Tool Call Precision**: fraction of tool calls with non-error outputs
- **Retrieval Recall@k**: overlap with gold theorem indices (when provided)
- **Iterations to Solution**: average ReAct steps (autonomous only)
- **Token Efficiency**: tokens per correct answer

### Running the benchmark

```
python scripts/run_benchmark.py \
  --dataset theoremqa \
  --modes baseline,rag,manual,autonomous \
  --output benchmarks/results.json
```

### Plotting charts

```
python scripts/plot_benchmarks.py \
  --input benchmarks/results.json \
  --output-dir benchmarks/charts
```

Plotting requires matplotlib (included in `.[dev]`).

### Output format

Results are stored as JSON under `benchmarks/results.json`:

```
{
  "dataset": { "name": "...", "path": "...", "split": "..." },
  "modes": {
    "baseline": {
      "metrics": { "accuracy": 0.0, ... },
      "examples": [ ... ]
    }
  }
}
```
