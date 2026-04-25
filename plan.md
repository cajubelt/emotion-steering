---
name: Reproduce Emotion Steering
overview: Reproduce the Anthropic "Emotion Concepts" paper's findings on an open-weight model running locally on a 16GB Apple Silicon Mac, then build a steered chat interface that amplifies or diminishes emotions during generation.
todos:
  - id: setup-env
    content: Set up Python environment, install dependencies, download Gemma 4 E2B-it, verify loading in float16 on MPS
    status: in_progress
  - id: generate-stories
    content: Build story generation script using Gemma 4 E2B-it itself, then spot-check ~10% of stories in Composer for quality
    status: pending
  - id: extract-vectors
    content: "Build vector extraction pipeline: forward passes, residual stream capture, mean-difference computation, PCA denoising"
    status: pending
  - id: validate-vectors
    content: Validate emotion vectors with logit lens, test prompts, and PCA geometry visualization
    status: pending
  - id: steered-chat
    content: Build steered chat interface with forward hooks for real-time emotion steering during generation
    status: pending
  - id: eval-stretch
    content: "Future direction (out of scope): test steering effects on evals (sycophancy, reasoning, etc.)"
    status: cancelled
isProject: false
---

# Reproducing Emotion Steering Locally

## Why Not Ollama

Ollama is a pure inference server built on llama.cpp. You send it a prompt and get tokens back -- there is no mechanism to hook into intermediate layers, extract activations from the residual stream, or inject steering vectors during the forward pass. Activation steering requires intercepting the model's internal computation mid-forward-pass to either read or write activations at specific layers. This means we need **full PyTorch model access** via HuggingFace `transformers`, with hooks provided by a library like `nnsight`.

The chat experience will be a Python script (terminal-based or Gradio UI) that loads the model, applies steering hooks, and streams tokens. Not as slick as `ollama run`, but it's the only way to steer.

## Model: Gemma 4 E2B-it

- **Size**: 5.1B total params (2.3B effective), ~10GB in float16
- **Architecture**: Dense transformer, hybrid sliding-window + global attention, same family as the larger Gemma 4 models
- **Fits on 16GB**: Yes, with ~5-6GB headroom -- no quantization needed, which keeps activation quality clean for vector extraction
- **Why this model**: Brand new (released April 2, 2026). Nobody has studied its emotion representations yet, so any findings are genuinely novel. Dense architecture is straightforward to hook into for steering. Fits comfortably in float16 on 16GB Apple Silicon, avoiding the complexity and potential quality loss of quantization. Same architecture family as E4B, so findings may generalize upward if you later get access to more compute.
- **Caveat**: 2B effective parameters is on the smaller side. Emotion representations may be less nuanced than in larger models, but prior work shows even small models form linear representations of concepts. We will validate this in the extraction phase.

## How Steering Actually Works

The paper's approach (simplified):

1. **Extraction**: Run text through the model, capture the residual stream vector at a target layer (~2/3 of the way through), averaged across tokens. Do this for many stories labeled by emotion.
2. **Vector computation**: For each emotion, average the activation vectors across all stories of that emotion, then subtract the grand mean across all emotions. This gives a direction in activation space that represents "more of this emotion than average."
3. **Steering at inference time**: During generation, register a PyTorch forward hook on the target layer. On every forward pass, add `strength * emotion_vector` to the residual stream activation before it continues to the next layer. This nudges the model's computation as though that emotion were more (or less) active.

The steering strength in the paper is measured as a fraction of the residual stream norm. Typical values: 0.01 to 0.1 (small but impactful). Too much steering degrades coherence.

## Implementation Pipeline

### Phase 1: Environment and model setup

- Create Python environment with: `torch`, `transformers`, `nnsight`, `numpy`, `scipy`, `tqdm`
- Download `google/gemma-4-E2B-it` from HuggingFace (Apache 2.0 license, no approval needed)
- Verify model loads in float16 on MPS and generates text
- Verify nnsight can hook into the model's layers

### Phase 2: Generate emotion-labeled stories

- Define a list of emotion words (start with ~20-30 core emotions: happy, sad, angry, afraid, calm, desperate, excited, nervous, proud, guilty, loving, surprised, etc.)
- For each emotion, generate ~50-100 short stories (1 paragraph) where a character experiences that emotion, using Gemma 4 E2B-it itself. This mirrors the paper's approach (Sonnet 4.5 generated its own stories) and means the stories naturally reflect what the model "thinks" each emotion looks like.
- Use diverse topic prompts to get variety (e.g., "Write a short story about a character at a job interview who feels [emotion]")
- Save stories to JSON files on disk
- **Spot-check**: After generation, sample ~10% of stories and review them with Composer (Cursor's fast/cheap model) to verify they actually contain the intended emotional content. Flag any emotions where the model struggles to produce convincing stories -- these may need prompt tweaks or may indicate the model doesn't represent that emotion well.

### Phase 3: Extract emotion vectors

- For each story, run it through the model and capture residual stream activations at the target layer (~2/3 depth through the model's 42 layers, so around layer 28)
- Average activations across token positions (starting from token 50 to skip non-emotional preamble)
- Average across all stories per emotion to get one vector per emotion
- Subtract the grand mean across all emotions
- (Optional denoising) Run neutral text through the model, compute top PCs of activations on neutral text (enough to explain ~50% variance), project these out of the emotion vectors
- Save emotion vectors as a `.pt` or `.npz` file

### Phase 4: Validate emotion vectors

- **Logit lens**: Project each emotion vector through the model's unembedding matrix, check that top tokens match the emotion (e.g., "desperate" vector upweights "desperate", "urgent", "bankrupt")
- **Activation on test prompts**: Run emotionally-charged prompts through the model, measure projection onto emotion vectors, verify correct emotions light up
- **Geometry**: Run PCA on the emotion vectors, check that PC1 correlates with valence and PC2 with arousal

### Phase 5: Steered chat interface

- Build a generation loop that registers a forward hook on the target layer
- The hook adds `steering_strength * emotion_vector` to the residual stream on every forward pass during token generation
- Wrap this in a simple chat REPL or Gradio UI where the user can:
  - Set the emotion to steer with (e.g., `calm`, `desperate`)
  - Set the steering strength (e.g., `+0.05`, `-0.03`)
  - Chat normally and see the steered responses
- Support multiple simultaneous steering vectors (e.g., `+desperate -calm`)

### Future direction: Eval on benchmarks (not part of this plan)

Once the steered chat interface is working, the natural next step would be to test whether steering affects behavior in measurable ways -- e.g., does `+calm -desperate` improve reasoning benchmarks? Does `+happy +loving` increase sycophancy? The steered chat interface in Phase 5 will be built with this in mind (clean API for programmatic steering, not just interactive chat), but running evals is out of scope for now.

## Key Files We Will Create

```text
llm-emotions/
  requirements.txt          # Dependencies
  README.md                 # Setup and usage instructions
  scripts/
    generate_stories.py     # Phase 2: generate emotion-labeled stories
    extract_vectors.py      # Phase 3: extract emotion vectors
    validate_vectors.py     # Phase 4: validate and visualize
    steered_chat.py         # Phase 5: interactive steered chat
  data/
    emotions.json           # List of emotion words
    stories/                # Generated stories per emotion
  vectors/                  # Saved emotion vectors
```
