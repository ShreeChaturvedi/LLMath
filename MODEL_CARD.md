## LLMath ReAct LoRA Model Card

### Overview

- **Base model**: DeepSeek-Math 7B RL
- **Task**: Proof-style mathematical reasoning with retrieval and tool use
- **Training data**: NaturalProofs + synthetic ReAct traces
- **Method**: LoRA fine-tuning with 4-bit quantization support

### Intended use

- Mathematical explanation and proof-style responses
- Retrieval-augmented theorem referencing
- Tool-assisted symbolic computation

### Limitations

- Output correctness depends on retrieved context and tool accuracy
- Long proofs may require more iterations than the default limit
- TheoremQA evaluation is a proxy, not a complete measure of rigor

### Usage

```
python scripts/run_demo.py
```

### License

This model card follows the repository license (MIT).
