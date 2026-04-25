from __future__ import annotations

from collections import defaultdict

import torch


PAIRWISE_PRESETS = {
    "clear_pairs": {
        "afraid": "calm",
        "angry": "calm",
        "ashamed": "happy",
        "calm": "angry",
        "desperate": "relieved",
        "excited": "calm",
        "grateful": "jealous",
        "guilty": "happy",
        "happy": "guilty",
        "jealous": "grateful",
        "lonely": "loving",
        "loving": "lonely",
        "nervous": "relieved",
        "playful": "nervous",
        "relieved": "desperate",
    }
}


NEUTRAL_TOPICS = [
    "waiting for a bus",
    "organizing a bookshelf",
    "writing a grocery list",
    "cleaning a kitchen counter",
    "commuting to an office",
    "sorting laundry",
    "walking through a park",
    "making a cup of tea",
    "setting up a desk lamp",
    "checking the weather forecast",
]


def build_neutral_rows(count: int) -> list[dict]:
    rows = []
    for idx in range(count):
        topic = NEUTRAL_TOPICS[idx % len(NEUTRAL_TOPICS)]
        text = (
            f"This is a neutral descriptive paragraph about {topic}. "
            "It should stay emotionally flat and factual. "
            "The character notices routine details, completes an ordinary task, "
            "and moves on without any unusually positive or negative reaction."
        )
        rows.append({"topic": topic, "story": text})
    return rows


def choose_components(matrix: torch.Tensor, variance_threshold: float) -> tuple[torch.Tensor, int]:
    if variance_threshold <= 0:
        return matrix.new_zeros((0, matrix.shape[1])), 0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    explained = singular_values.pow(2)
    cumulative = torch.cumsum(explained / explained.sum(), dim=0)
    num_components = int(torch.searchsorted(cumulative, torch.tensor(variance_threshold)).item()) + 1
    return right_vectors[:num_components], num_components


def project_out_components(vector: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    if components.shape[0] == 0:
        return vector
    for component in components:
        unit = component / component.norm().clamp_min(1e-8)
        vector = vector - torch.dot(vector, unit) * unit
    return vector


def build_raw_vectors(
    emotion_means: dict[str, torch.Tensor],
    *,
    construction_mode: str,
    pairwise_preset: str,
) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    emotions = sorted(emotion_means)

    if construction_mode == "grand_mean":
        grand_mean = torch.stack([emotion_means[emotion] for emotion in emotions]).mean(dim=0)
        return {emotion: emotion_means[emotion] - grand_mean for emotion in emotions}, None

    if construction_mode == "one_vs_rest":
        raw_vectors = {}
        for emotion in emotions:
            other_means = [emotion_means[other] for other in emotions if other != emotion]
            raw_vectors[emotion] = emotion_means[emotion] - torch.stack(other_means).mean(dim=0)
        return raw_vectors, None

    if construction_mode == "pairwise":
        contrast_map = PAIRWISE_PRESETS[pairwise_preset]
        raw_vectors = {}
        applied_map = {}
        for emotion, reference in contrast_map.items():
            if emotion not in emotion_means or reference not in emotion_means:
                continue
            raw_vectors[emotion] = emotion_means[emotion] - emotion_means[reference]
            applied_map[emotion] = reference
        return raw_vectors, applied_map

    raise ValueError(f"Unsupported construction mode: {construction_mode}")


def aggregate_emotion_means(
    activations_by_emotion: dict[str, list[torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    emotion_means = {}
    emotion_counts = {}
    for emotion, vectors in activations_by_emotion.items():
        emotion_means[emotion] = torch.stack(vectors).mean(dim=0)
        emotion_counts[emotion] = len(vectors)
    return emotion_means, emotion_counts


def group_story_activations(rows: list[dict]) -> dict[str, list[torch.Tensor]]:
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        grouped[row["emotion"]].append(row["activation"].float().cpu())
    return grouped


def build_vector_payload(
    *,
    model_id: str,
    layer_idx: int,
    start_token: int,
    pooling_strategy: str,
    pool_size: int,
    construction_mode: str,
    pairwise_preset: str | None,
    raw_vectors: dict[str, torch.Tensor],
    vectors: dict[str, torch.Tensor],
    emotion_counts: dict[str, int],
    num_components_projected_out: int,
    average_residual_norm: float,
    contrast_map: dict[str, str] | None = None,
    source_cache_dir: str | None = None,
) -> dict:
    return {
        "format_version": 2,
        "model_id": model_id,
        "layer_idx": layer_idx,
        "start_token": start_token,
        "pooling_strategy": pooling_strategy,
        "pool_size": pool_size,
        "construction_mode": construction_mode,
        "pairwise_preset": pairwise_preset if construction_mode == "pairwise" else None,
        "contrast_map": contrast_map,
        "num_components_projected_out": num_components_projected_out,
        "average_residual_norm": average_residual_norm,
        "vectors": vectors,
        "raw_vectors": raw_vectors,
        "emotion_counts": emotion_counts,
        "source_cache_dir": source_cache_dir,
    }
