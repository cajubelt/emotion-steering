# Current Status

This repository currently contains a working implementation scaffold for reproducing an activation-steering workflow on Gemma 4. The code supports:

- self-generation of emotion-labeled stories,
- residual-stream activation capture from a dynamically discovered decoder layer,
- construction of emotion vectors with several contrast modes,
- validation through logit-lens projections, prompt probes, and PCA geometry,
- an interactive REPL that applies residual-stream steering during generation.

## Verified locally

The following checks passed on April 25, 2026 on an Apple Silicon laptop:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m compileall llm_emotions scripts
python scripts/smoke_test_model.py --skip-generation
```

The smoke test loaded `google/gemma-4-E2B-it`, found the decoder layers at `model.model.language_model.layers`, selected layer 23 of 35 as the default steering layer, and captured a residual-stream activation with shape `(1536,)`.

## Not yet completed

The repository does not currently include generated story datasets, saved vectors, or validation reports. Those outputs are intentionally ignored by git because they are large/generated artifacts. The next research milestone is to run a small end-to-end pilot, inspect the generated stories, and include a short result summary with representative validation artifacts.
