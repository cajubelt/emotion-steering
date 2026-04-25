# Research Plan

This project is a small mechanistic-interpretability reproduction and extension: recover emotion-related directions from a Gemma 4 instruction model and test whether residual-stream interventions can steer generated text in predictable ways.

## Motivation

Anthropic's "Emotion Concepts and their Function in a Large Language Model" studies linearly accessible emotion representations and activation steering in a production model. This repository adapts the core workflow to an open-weight model that can run locally. The goal is not to exactly reproduce Anthropic's internal setup, but to demonstrate the empirical loop: generate controlled data, extract candidate representations, validate them, intervene on activations, and document where the method succeeds or fails.

## Pipeline

1. Generate short stories labeled by target emotion, using the model itself.
2. Capture residual-stream activations at a decoder layer about two-thirds through the network.
3. Average token activations and construct emotion vectors by contrasting each emotion against a baseline.
4. Validate the directions with token projections, held-out prompt probes, and vector geometry.
5. Apply vectors during generation with PyTorch forward hooks and inspect behavioral changes.

## Near-term experiments

- Run a smoke-scale pilot with four emotions and one to three stories per emotion.
- Compare grand-mean, one-vs-rest, and pairwise vector construction.
- Check whether prompt probes recover the intended emotion above chance.
- Test small positive and negative steering strengths in the chat REPL.
- Record failure modes, especially confounds between valence, arousal, and topic.

## Practical constraints

The default path uses `google/gemma-4-E2B-it` because this repository is meant to be runnable on a lower-resource laptop. Larger Gemma 4 variants can be selected with `--model-id` or `EMOTION_STEERING_MODEL_ID`, but they are not the default reproducibility target.
