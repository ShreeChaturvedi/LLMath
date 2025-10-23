## Contributing

Thanks for your interest in improving LLMath.

### Development setup

```
git clone https://github.com/ShreeChaturvedi/LLMath.git
cd LLMath
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running checks

```
ruff check .
ruff format .
mypy src/llmath
pytest
```

### Testing a full workflow

```
python scripts/build_index.py --rebuild
python scripts/train_lora.py --mode react --epochs 4
python scripts/run_benchmark.py --modes baseline,autonomous
python scripts/run_demo.py
```

### Style guide

- Keep functions small and well documented.
- Prefer typed public APIs.
- Avoid heavyweight imports at module import time.

### Pull requests

- Describe the change and the rationale.
- Update docs when behavior changes.
- Add or adjust tests where relevant.
