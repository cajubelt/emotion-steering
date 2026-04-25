from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

from llm_emotions.config import DEFAULT_MODEL_ID


@dataclass
class LoadedModel:
    model_id: str
    processor: object
    model: torch.nn.Module
    device: torch.device


def infer_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> LoadedModel:
    device = device or infer_device()
    torch_dtype = torch_dtype or default_dtype_for_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    model_class = AutoModelForCausalLM
    if getattr(config, "model_type", None) == "gemma4":
        model_class = Gemma4ForConditionalGeneration
    model = model_class.from_pretrained(model_id, dtype=torch_dtype, low_cpu_mem_usage=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return LoadedModel(
        model_id=model_id,
        processor=tokenizer,
        model=model,
        device=device,
    )


def locate_decoder_layers_container(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    candidates = [
        ("model", model),
        ("model.model", getattr(model, "model", None)),
        ("model.model.model", getattr(getattr(model, "model", None), "model", None)),
        ("model.model.language_model", getattr(getattr(model, "model", None), "language_model", None)),
        (
            "model.model.language_model.model",
            getattr(getattr(getattr(model, "model", None), "language_model", None), "model", None),
        ),
        ("model.language_model", getattr(model, "language_model", None)),
        ("model.language_model.model", getattr(getattr(model, "language_model", None), "model", None)),
    ]
    for path, candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return path, candidate
    raise AttributeError("Could not locate decoder layers on the loaded model.")


def get_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    _, container = locate_decoder_layers_container(model)
    return list(container.layers)


def target_layer_index(model: torch.nn.Module, fraction: float = 2 / 3) -> int:
    layers = get_decoder_layers(model)
    return min(len(layers) - 1, max(0, int(round((len(layers) - 1) * fraction))))


def build_chat_text(processor, messages: list[dict], enable_thinking: bool = False) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def processor_to_device(processor_inputs, device: torch.device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in processor_inputs.items()}


def tokenize_text(processor, text: str, device: torch.device):
    inputs = processor(text=text, return_tensors="pt")
    return processor_to_device(inputs, device)


def tokenize_chat(processor, messages: list[dict], device: torch.device, enable_thinking: bool = False):
    text = build_chat_text(processor, messages, enable_thinking=enable_thinking)
    inputs = processor(text=text, return_tensors="pt")
    inputs = processor_to_device(inputs, device)
    input_len = int(inputs["input_ids"].shape[-1])
    return inputs, input_len


def decode_generated_tokens(processor, outputs: torch.Tensor, input_len: int) -> str:
    generated_ids = outputs[0][input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True)


def generate_text(
    loaded: LoadedModel,
    messages: list[dict],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    enable_thinking: bool = False,
) -> str:
    inputs, input_len = tokenize_chat(
        loaded.processor,
        messages,
        loaded.device,
        enable_thinking=enable_thinking,
    )
    do_sample = temperature > 0
    with torch.no_grad():
        outputs = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=loaded.processor.pad_token_id,
        )
    return decode_generated_tokens(loaded.processor, outputs, input_len)


def _unwrap_hook_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def pool_hidden_states(
    hidden: torch.Tensor,
    *,
    start_token: int = 50,
    pooling_strategy: str = "mean",
    pool_size: int = 32,
) -> torch.Tensor:
    token_start = min(start_token, max(0, hidden.shape[0] - 1))
    candidate_hidden = hidden[token_start:]

    if pooling_strategy == "mean":
        pooled_hidden = candidate_hidden
    elif pooling_strategy == "suffix":
        window_size = min(max(1, pool_size), candidate_hidden.shape[0])
        pooled_hidden = candidate_hidden[-window_size:]
    elif pooling_strategy == "max_norm_window":
        window_size = min(max(1, pool_size), candidate_hidden.shape[0])
        token_norms = candidate_hidden.float().norm(dim=-1)
        window_scores = token_norms.unfold(0, window_size, 1).mean(dim=-1)
        window_start = int(torch.argmax(window_scores).item())
        pooled_hidden = candidate_hidden[window_start : window_start + window_size]
    else:
        raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

    return pooled_hidden.mean(dim=0)


def capture_hidden_mean(
    loaded: LoadedModel,
    text: str,
    *,
    layer_idx: int,
    start_token: int = 50,
    pooling_strategy: str = "mean",
    pool_size: int = 32,
) -> torch.Tensor:
    layers = get_decoder_layers(loaded.model)
    captured = {}

    def hook(_module, _inputs, output):
        hidden = _unwrap_hook_output(output)
        captured["hidden"] = hidden.detach()
        return output

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        inputs = tokenize_text(loaded.processor, text, loaded.device)
        with torch.no_grad():
            loaded.model(**inputs)
    finally:
        handle.remove()

    hidden = captured["hidden"][0]
    pooled = pool_hidden_states(
        hidden,
        start_token=start_token,
        pooling_strategy=pooling_strategy,
        pool_size=pool_size,
    )
    return pooled.float().cpu()


def capture_hidden_means(
    loaded: LoadedModel,
    text: str,
    *,
    layer_indices: list[int],
    start_token: int = 50,
    pooling_strategy: str = "mean",
    pool_size: int = 32,
) -> dict[int, torch.Tensor]:
    if not layer_indices:
        raise ValueError("layer_indices must not be empty")

    ordered_indices = list(dict.fromkeys(layer_indices))
    layers = get_decoder_layers(loaded.model)
    invalid_indices = [idx for idx in ordered_indices if idx < 0 or idx >= len(layers)]
    if invalid_indices:
        raise IndexError(f"Layer indices out of range: {invalid_indices}")

    captured: dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in ordered_indices:
        def hook(_module, _inputs, output, *, layer_idx: int = layer_idx):
            hidden = _unwrap_hook_output(output)[0]
            pooled = pool_hidden_states(
                hidden.detach(),
                start_token=start_token,
                pooling_strategy=pooling_strategy,
                pool_size=pool_size,
            )
            captured[layer_idx] = pooled.float().cpu()
            return output

        handles.append(layers[layer_idx].register_forward_hook(hook))

    try:
        inputs = tokenize_text(loaded.processor, text, loaded.device)
        with torch.no_grad():
            loaded.model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {layer_idx: captured[layer_idx] for layer_idx in ordered_indices}


def residual_norm_for_text(
    loaded: LoadedModel,
    text: str,
    *,
    layer_idx: int,
) -> float:
    layers = get_decoder_layers(loaded.model)
    captured = {}

    def hook(_module, _inputs, output):
        hidden = _unwrap_hook_output(output)
        captured["norm"] = hidden.detach().norm(dim=-1).mean().item()
        return output

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        inputs = tokenize_text(loaded.processor, text, loaded.device)
        with torch.no_grad():
            loaded.model(**inputs)
    finally:
        handle.remove()
    return float(captured["norm"])


@contextmanager
def apply_residual_steering(
    loaded: LoadedModel,
    *,
    layer_idx: int,
    steering_vectors: Iterable[tuple[torch.Tensor, float]],
):
    layers = get_decoder_layers(loaded.model)
    device = loaded.device
    vectors = []
    for vector, strength in steering_vectors:
        if strength == 0:
            continue
        vector = vector.to(device=device, dtype=default_dtype_for_device(device))
        vector = vector / vector.norm().clamp_min(1e-8)
        vectors.append((vector, strength))

    def hook(_module, _inputs, output):
        if not vectors:
            return output
        hidden = _unwrap_hook_output(output)
        delta = torch.zeros_like(hidden)
        token_norm = hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for vector, strength in vectors:
            delta = delta + (strength * token_norm * vector.view(1, 1, -1))

        steered = hidden + delta
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
