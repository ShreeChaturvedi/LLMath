# LLMath

Theorem Retrieval-Augmented Generation with Agentic Tool Use for Mathematical Proofs

> **Note**: This repository was created in January 2026. The codebase represents work
> completed during the Fall 2025 semester (August-October 2025) for CSE 434 at Miami University,
> subsequently refactored from a Jupyter notebook into a production-quality Python package.

## Overview

LLMath is a mathematical proof assistant that combines:

1. **Retrieval-Augmented Generation (RAG)** - Retrieves relevant theorems and definitions from the NaturalProofs corpus
2. **Symbolic Tool Use** - Leverages SymPy for symbolic computation (differentiation, integration, solving, simplification)
3. **Fine-tuned LLM** - Uses DeepSeek-Math 7B with LoRA fine-tuning for proof-style answers

## Problem & Motivation

LLMs answering mathematical questions frequently hallucinate, which is unacceptable in mathematics where claims must be justified via cited theorems and symbolic steps. LLMath addresses this by:

- Retrieving theorem and definitions from a curated math corpus
- Routing queries to theorem application and/or computation
- Composing concise proof-style answers with explicit citations and tool-checked steps

## Features

- **NaturalProofs Retrieval**: Semantic search over 12,000+ theorem statements using FAISS
- **SymPy Integration**: Symbolic differentiation, integration, equation solving, and simplification
- **Proof-style Answers**: Structured responses with theorem citations and step-by-step reasoning
- **Interactive Demo**: Gradio-based UI for exploring the system

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

```python
from llmath import MathAgent, load_config

# Load configuration
config = load_config()

# Initialize agent
agent = MathAgent.from_config(config)

# Ask a question
result = agent.run(
    question="Prove that the derivative of x**2*sin(x) is 2*x*sin(x) + x**2*cos(x).",
    sympy_expressions=["diff: x**2*sin(x)"]
)

print(result.answer)
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
│   ├── retrieval/      # FAISS-based theorem retrieval
│   ├── tools/          # SymPy symbolic computation tools
│   ├── prompts/        # Prompt templates and orchestration
│   ├── inference/      # Model loading and generation
│   ├── training/       # LoRA fine-tuning code
│   ├── evaluation/     # Baseline vs agent comparison
│   ├── agent/          # Main agent orchestration
│   └── api/            # Gradio demo interface
├── scripts/            # CLI scripts for training/evaluation
├── configs/            # YAML configuration files
├── tests/              # Unit and integration tests
└── notebooks/          # Demo notebooks
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- Shree Chaturvedi (chaturs@miamioh.edu)
- Jiahao Han (hanj24@miamioh.edu)

## Acknowledgments

- [NaturalProofs](https://github.com/wellecks/naturalproofs) dataset
- [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) model
- CSE 434 at Miami University
