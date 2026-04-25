# Current Status

This repository contains a working implementation for reproducing an activation-steering workflow on Gemma 4. The code supports:

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

## End-to-end result

An end-to-end run has been completed through story generation, vector extraction, validation, and interactive steering. One qualitative chat result:

```text
Prompt:
Write a short text message someone sends after their coworker took credit for their work again. Keep it realistic and one paragraph.

Base:
"Seriously? That was my idea. Let's keep the credit where it belongs."

+angry:
"Seriously? That was my idea. Don't do it again."

-angry:
"Just wanted to loop back on the Q3 report—it looks great, and I'm excited to see it land. Hope we can chat about the actual data breakdown soon so everyone has the full picture!"
```

The result is directionally consistent with the intended intervention: increasing the `angry` direction made the response more forceful, while subtracting the direction made it more diplomatic and collaborative.

## Remaining work

Generated story datasets, saved vectors, and validation reports are not currently tracked in git. They are ignored by default because they can be large and are generated artifacts. The next useful cleanup step is to curate a small result bundle, such as a short validation summary and one representative PCA plot, so readers can inspect the empirical output without rerunning the full pipeline.
