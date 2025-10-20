# LLMath

Theorem Retrieval-Augmented Generation with Agentic Tool Use for Mathematical Proofs

> **Note**: This repository was created in January 2026. The codebase represents work
> completed during the Fall 2025 semester (August-October 2025) for CSE 434 at Miami University,
> subsequently refactored from a Jupyter notebook into a production-quality Python package.

## Overview

LLMath is a mathematical proof assistant that combines:

1. **Retrieval-Augmented Generation (RAG)** - Retrieves relevant theorems and definitions from the NaturalProofs corpus
2. **Symbolic Tool Use** - Leverages SymPy for symbolic computation (differentiation, integration, solving, simplification)
3. **Fine-tuned LLM** - Uses DeepSeek-Math 7B RL with LoRA fine-tuning for proof-style answers

## Problem & Motivation

LLMs answering mathematical questions frequently hallucinate, which is unacceptable in mathematics where claims must be justified via cited theorems and symbolic steps. LLMath addresses this by:

- Retrieving theorems and definitions from a curated math corpus
- Routing queries to theorem application and/or computation
- Composing concise proof-style answers with explicit citations and tool-checked steps

## Features

- **NaturalProofs Retrieval**: Semantic search over 12,000+ theorem statements using FAISS and sentence-transformers
- **SymPy Integration**: Symbolic differentiation, integration, equation solving, and simplification
- **Proof-style Answers**: Structured responses with theorem citations ([T1], [T2]) and tool results ([S1], [S2])
- **LoRA Fine-tuning**: Efficient training on mathematical proofs with 4-bit quantization
- **Interactive Demo**: Gradio-based UI for exploring the system

## Requirements

- Python 3.10+
- PyTorch with CUDA support (for GPU inference)
- ~16GB VRAM for 4-bit quantized inference
- ~8GB disk space for FAISS index

## Installation

```bash
pip install llmath
```

Or install from source:

```bash
git clone https://github.com/ShreeChaturvedi/LLMath.git
cd LLMath
pip install -e ".[dev]"
```

## Quick Start

### Using the Python API

```python
from llmath.retrieval import NaturalProofsRetriever
from llmath.retrieval.theorem_kb import TheoremKB
from llmath.tools import simplify_expr, solve_equation, differentiate_expr

# Initialize retriever
retriever = NaturalProofsRetriever()
kb = TheoremKB(retriever)

# Retrieve relevant theorems
theorems = kb.get_theorems("derivative of product of functions", k=3)
for t in theorems:
    print(f"[{t.title}] {t.snippet[:100]}...")

# Use symbolic tools
print(differentiate_expr("x**2 * sin(x)"))  # -> 2*x*sin(x) + x**2*cos(x)
print(solve_equation("x**2 - 1 = 0"))       # -> [-1, 1]
```

### Using the Full Agent (requires GPU)

```python
from llmath.agent import create_math_agent
from llmath.retrieval import NaturalProofsRetriever

retriever = NaturalProofsRetriever()
agent = create_math_agent(retriever)

result = agent.run(
    question="Prove that the derivative of x**2*sin(x) is 2*x*sin(x) + x**2*cos(x).",
    sympy_expressions=["diff: x**2*sin(x)"]
)

print(result.answer)
```

### CLI Scripts

```bash
# Build the FAISS index (first-time setup)
python scripts/build_index.py --rebuild

# Run LoRA fine-tuning
python scripts/train_lora.py --epochs 4 --output-dir outputs/lora

# Evaluate baseline vs agent
python scripts/evaluate.py --adapter-path outputs/lora

# Launch interactive demo
python scripts/run_demo.py --share
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Tool Registry  │───▶│   SymPy Tools   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                             │
         ▼                                             ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  NaturalProofs  │───▶│ Prompt Builder  │◀───│  Symbolic       │
│  Retriever      │    │                 │    │  Results        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │  DeepSeek-Math  │
                       │  (LoRA tuned)   │
                       └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │  Proof-style    │
                       │  Answer         │
                       └─────────────────┘
```

## Project Structure

```
llmath/
├── src/llmath/
│   ├── config.py          # Pydantic configuration
│   ├── retrieval/         # FAISS-based theorem retrieval
│   │   ├── faiss_retriever.py
│   │   └── theorem_kb.py
│   ├── tools/             # SymPy symbolic computation
│   │   ├── sympy_tools.py
│   │   └── registry.py
│   ├── prompts/           # Prompt templates and orchestration
│   │   ├── templates.py
│   │   ├── builder.py
│   │   └── orchestrator.py
│   ├── inference/         # Model loading and generation
│   │   ├── model_loader.py
│   │   ├── generation.py
│   │   └── deepseek.py
│   ├── training/          # LoRA fine-tuning
│   │   ├── data.py
│   │   ├── formatting.py
│   │   └── trainer.py
│   ├── evaluation/        # Baseline comparison
│   │   ├── baseline.py
│   │   └── comparison.py
│   ├── agent/             # Main agent
│   │   └── math_agent.py
│   └── api/               # Gradio interface
│       └── gradio_app.py
├── scripts/               # CLI entry points
├── configs/               # YAML configurations
├── tests/                 # Unit and integration tests
└── notebooks/             # Demo notebooks
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=llmath

# Run specific test file
pytest tests/unit/test_sympy_tools.py
```

### Code Style

The project uses:
- Black for formatting
- isort for import sorting
- mypy for type checking

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

model:
  model_name: "deepseek-ai/deepseek-math-7b-rl"
  load_in_4bit: true
  lora_r: 16
  lora_alpha: 32

generation:
  max_new_tokens: 512
  temperature: 0.4

agent:
  default_k: 5
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- Shree Chaturvedi (chaturs@miamioh.edu)
- Jiahao Han (hanj24@miamioh.edu)

## Acknowledgments

- [NaturalProofs](https://github.com/wellecks/naturalproofs) dataset by Sean Welleck et al.
- [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) model
- Miami University CSE 434: Introduction to Machine Learning
