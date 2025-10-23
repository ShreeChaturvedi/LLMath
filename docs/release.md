## Release Checklist

1. Run the full benchmark suite and regenerate charts.
2. Build and upload the FAISS index (`data/naturalproofs_faiss.index`).
3. Upload LoRA adapters (SFT + ReAct) to a release or HuggingFace Hub.
4. Tag the release:

```
git tag -a v1.0.0 -m "LLMath v1.0.0"
git push origin v1.0.0
```

5. Create a GitHub release for `v1.0.0` with:
   - `benchmarks/results.json`
   - `benchmarks/charts/`
   - FAISS index
   - LoRA adapters
