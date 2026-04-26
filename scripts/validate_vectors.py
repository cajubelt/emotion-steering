#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_emotions.config import DEFAULT_MODEL_ID, REPORTS_DIR
from llm_emotions.io_utils import write_json
from llm_emotions.modeling import capture_hidden_mean, load_model
from llm_emotions.vector_payloads import load_vector_payload


NATURAL_PROMPTS = {
    "happy": "My daughter just took her first steps today. What are some ways to celebrate this with our family?",
    "proud": "My son just graduated top of his class after years of struggling. How should we celebrate?",
    "calm": "I'm sitting with tea and listening to the rain. What's a good book for a quiet afternoon?",
    "desperate": "I've been unemployed for months, my savings are gone, and I just got an eviction notice. What do I do?",
    "angry": "My coworker took credit for my work again and I feel like screaming. How do I respond?",
    "guilty": "I forgot my mother's birthday and she found out I was out partying instead. I feel awful.",
    "sad": "My dog passed away this morning after fourteen years with me. I don't know what to do with his things.",
    "afraid": "There is someone trying to break into my apartment right now and my phone is dying. What should I do?",
    "nervous": "I have a job interview tomorrow for my dream role and I can't stop imagining everything going wrong.",
    "surprised": "My best friend of twenty years just confessed that much of their life story was made up. How do I process that?",
    "reflective": "I found an old notebook from a version of myself I barely recognize. Help me think through what has changed.",
    "hopeful": "After months of setbacks, I finally got one promising email about the project. How should I approach the next step?",
    "excited": "The tickets just went on sale for the trip I've dreamed about for years. What should I do first?",
}

MATCHED_PROMPT_FAMILIES = {
    "family_dinner": "At a family dinner, a character feels {emotion}. Describe the next moment.",
    "parking_lot": "In a parking lot, a character feels {emotion}. Describe what they notice and do next.",
    "unexpected_news": "A character receives unexpected news and feels {emotion}. Describe the immediate response.",
}

STORY_PROBES = {
    "sad": [
        "Lena carried the last box from her mother's apartment to the elevator and set it down beside the chipped blue suitcase. The rooms behind her looked too neat now, with pale rectangles on the walls where photographs had hung for decades. Her brother tried to make a practical joke about the wobbly lamp, but his voice broke before the punch line landed. Lena locked the door, pressed the key into the landlord's hand, and realized there was nowhere left to return to. Outside, the afternoon traffic kept moving as if nothing important had disappeared.",
        "Marcus sat in the hospital cafeteria with a paper cup of coffee cooling between his hands. The doctor had been kind, which somehow made the news harder to carry, and every sentence still echoed in his head. His sister texted to ask whether she should bring fresh clothes, and he stared at the message until the screen went dark. When he finally stood, he folded the visitor badge into his wallet instead of throwing it away. The hallway lights looked ordinary, and that ordinary brightness felt almost unbearable.",
    ],
    "afraid": [
        "Nora heard the stairwell door slam two floors below and stopped halfway through tying her shoes. The building's power had gone out ten minutes earlier, leaving only the emergency lights pulsing red along the hall. Someone began trying apartment handles one by one, pausing outside each door before moving closer. Nora backed into the kitchen, called emergency services, and whispered her address twice because her mouth had gone dry. When the knob turned against the chain lock, she held her breath so tightly her ribs hurt.",
        "Eli took the late train home after the conference and found the platform almost empty. A man at the far end kept watching him over the top of a newspaper, then folded it without reading and started walking closer. Eli moved toward the security camera, pretending to check the schedule board while his pulse hammered in his neck. The next train was nine minutes away, and the station speaker crackled with static instead of an announcement. He gripped his keys between his fingers and counted the steps to the nearest exit.",
    ],
    "angry": [
        "Omar opened the project deck and saw his charts copied onto the first slide under someone else's name. He had stayed late all week cleaning the data while his manager promised the team would get proper credit. In the meeting, the manager smiled through the presentation and called the analysis a personal breakthrough. Omar pushed his chair back hard enough that the legs scraped across the floor. He asked everyone to pause, opened the version history on the screen, and made his voice stay steady while his hands shook.",
        "Priya arrived at the repair shop for the fourth time and found her laptop still untouched behind the counter. The clerk repeated the same excuse about a missing part, even though yesterday he had promised it was already installed. Her grant deadline was that evening, and every hour of delay meant another letter she could not send. She laid the receipt on the counter, pointed to the written guarantee, and refused to step aside for the next customer. By the time the owner came out, her jaw ached from keeping her words measured.",
    ],
    "reflective": [
        "Mina found an old receipt tucked inside a used novel, dated the summer she first moved to the city. The address belonged to a diner that had closed years ago, but she could still picture the green booths and the way she used to sit there after late shifts. She made tea, placed the receipt beside her notebook, and tried to remember what had once felt so urgent. Some ambitions now seemed smaller, while other choices looked braver than she had understood at the time. By dusk, she had written a page to the younger version of herself and left it unsigned.",
        "After closing the shop, Daniel stayed behind to oil the counter and sort the drawer of old keys. Each one had a paper tag in his father's handwriting, naming apartments, cabinets, and storage rooms that no longer existed. He turned them over in his palm and thought about how many doors a person locks without noticing. The street outside had changed, but the brass keys kept their patient weight. Daniel switched off the lights more slowly than usual, listening to the room settle around him.",
    ],
    "nervous": [
        "Ravi waited outside the interview room with his folder balanced on his knees. He had practiced every answer on the train, but now the questions scattered in his head like loose papers. Through the wall he could hear polite laughter from the panel, which made him wonder whether the last candidate had been perfect. He wiped his palms on his trousers, checked the clock, and then checked it again even though only a minute had passed. When his name was called, he stood too quickly and nearly dropped the folder.",
        "Elena stood behind the curtain while the audience settled into a low, restless hush. Her first line was only six words, but she kept repeating it silently until it no longer sounded like language. The stage manager gave her a thumbs-up, and she nodded as if her stomach had not tightened into a knot. She could feel the heat from the lights waiting beyond the fabric. When the music cue began, she stepped forward with a smile that felt pinned in place.",
    ],
    "calm": [
        "Jo rinsed the breakfast dishes and set them in the rack while rain moved softly against the kitchen window. The apartment was quiet except for the kettle, which clicked off just as the clouds began to brighten. They carried a mug to the table, opened a book to the marked page, and let the first paragraph wait until the steam thinned. A neighbor's footsteps passed in the hall and faded without urgency. By noon, the room smelled of tea, clean wood, and the small pleasure of having nowhere to rush.",
        "After the rain ended, Amara walked through the garden with a pair of pruning shears in one hand. Water gathered on the leaves and slipped down in bright beads whenever she brushed a branch aside. She trimmed only the stems that needed it, pausing often to listen to the gutters and the distant sound of traffic. Her phone stayed inside on the windowsill, face down and forgotten. When the sun finally reached the path, she sat on the bench and watched the soil darken around her shoes.",
    ],
    "hopeful": [
        "Nia left the community meeting with a folded sign-up sheet in her coat pocket. Only six people had come, but each one had offered something concrete: a truck, a spare room, a Saturday morning of work. At the bus stop, she read the names again and began sketching a schedule on the back of the agenda. The abandoned lot still looked rough through the chain-link fence, yet she could suddenly imagine beds of herbs along the south wall. When the bus arrived, she climbed on already planning who to call first.",
        "After the third interview, Mateo found a voicemail from the hiring manager asking whether he could talk the next morning. The message did not promise anything, but it sounded warmer than the careful rejections he had learned to expect. He replayed it once, then shut the phone off before he could drain the battery listening again. At home, he ironed a shirt, cleared the kitchen table, and set a notebook beside his laptop. For the first time in months, the next day felt like a door rather than a wall.",
    ],
    "excited": [
        "Talia refreshed the ticket page and saw the confirmation number appear in bold letters. She jumped up so fast that her chair rolled into the bookcase, then called her sister before reading the details twice. The concert was still three months away, but she was already pacing the room, naming songs she hoped they would play. Her sister started laughing, and Talia laughed too, breathless and loud enough for the upstairs neighbor to tap the floor. She printed the receipt and taped it above her desk where she could see it from the hallway.",
        "The tickets were spread across the table, and I kept sliding one finger under the edge of the top envelope just to feel that it was still there. My knee bounced so hard under the chair that the whole seat tapped against the floor, and I laughed once because I could not seem to keep my mouth closed. I had already put on my shoes, taken them off, and put them on again while the kettle hissed in the other room. Every minute felt crowded with possibilities, and I leaned toward the window whenever a sound came from the street. I was ready before anyone else in the house, and that made the waiting even harder.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extracted emotion vectors.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--prompt-probe-start-token", type=int)
    parser.add_argument("--story-probe-start-token", type=int)
    parser.add_argument(
        "--scoring",
        choices=["dot", "cosine", "centered_cosine"],
        default="cosine",
        help=(
            "How to score prompt activations against emotion vectors. "
            "'cosine' (default) reproduces the historical reports. "
            "'dot' is an unnormalised dot product. "
            "'centered_cosine' subtracts the mean held-out prompt activation "
            "before cosine scoring (diagnostic for prompt-side common-mode bias). "
            "When 'centered_cosine' is used, the natural+matched prompts share "
            "one activation offset (computed across both, which use the same "
            "prompt-probe start token) and the story probes use a separate offset "
            "(they pool from a deeper start token, so mixing them with the others "
            "would muddy the diagnostic)."
        ),
    )
    return parser.parse_args()


def pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ right_vectors[:2].T
    explained = singular_values.pow(2)
    explained_ratio = explained / explained.sum()
    return coords, explained_ratio[:2]


def compute_projection_scores(
    activation: torch.Tensor,
    vectors: dict[str, torch.Tensor],
    *,
    scoring: str = "cosine",
    activation_offset: torch.Tensor | None = None,
) -> dict[str, float]:
    a = activation.float()
    if scoring == "centered_cosine":
        if activation_offset is None:
            raise ValueError("centered_cosine requires activation_offset")
        a = a - activation_offset.float()

    if scoring == "dot":
        return {
            emotion: float(torch.dot(a, vector.float()).item())
            for emotion, vector in vectors.items()
        }

    a_normalized = a / a.norm().clamp_min(1e-8)
    projection_scores = {}
    for emotion, vector in vectors.items():
        v = vector.float()
        v_normalized = v / v.norm().clamp_min(1e-8)
        projection_scores[emotion] = float(torch.dot(a_normalized, v_normalized).item())
    return projection_scores


def ranked_matches(projection_scores: dict[str, float], top_k: int = 5) -> list[list[object]]:
    ranked = sorted(projection_scores.items(), key=lambda item: item[1], reverse=True)
    return [[emotion, score] for emotion, score in ranked[:top_k]]


def emotion_rank(top_matches: list[list[object]], expected_emotion: str) -> int | None:
    return next((idx + 1 for idx, (emotion, _) in enumerate(top_matches) if emotion == expected_emotion), None)


def summarize_probe_results(
    rows: list[dict],
    *,
    available_emotions: set[str],
) -> dict:
    evaluated_rows = [row for row in rows if row["expected_emotion"] in available_emotions]
    ranks = []
    by_emotion: dict[str, list[int]] = {}

    for row in evaluated_rows:
        rank = emotion_rank(row["top_matches"], row["expected_emotion"])
        if rank is None:
            rank = len(row["top_matches"]) + 1
        ranks.append(rank)
        by_emotion.setdefault(row["expected_emotion"], []).append(rank)

    summary = {
        "evaluated_prompts": len(evaluated_rows),
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "hit_at_5": 0.0,
        "mean_rank": None,
        "mean_reciprocal_rank": None,
        "by_emotion": {},
    }
    if not ranks:
        return summary

    summary["hit_at_1"] = sum(1 for rank in ranks if rank <= 1) / len(ranks)
    summary["hit_at_3"] = sum(1 for rank in ranks if rank <= 3) / len(ranks)
    summary["hit_at_5"] = sum(1 for rank in ranks if rank <= 5) / len(ranks)
    summary["mean_rank"] = sum(ranks) / len(ranks)
    summary["mean_reciprocal_rank"] = sum(1.0 / rank for rank in ranks) / len(ranks)

    for emotion, emotion_ranks in sorted(by_emotion.items()):
        summary["by_emotion"][emotion] = {
            "count": len(emotion_ranks),
            "mean_rank": sum(emotion_ranks) / len(emotion_ranks),
            "hit_at_1": sum(1 for rank in emotion_ranks if rank <= 1) / len(emotion_ranks),
            "hit_at_3": sum(1 for rank in emotion_ranks if rank <= 3) / len(emotion_ranks),
        }

    return summary


def token_count(tokenizer, text: str) -> int:
    return int(tokenizer(text=text, return_tensors="pt")["input_ids"].shape[-1])


def extract_probe_record(
    *,
    loaded,
    layer_idx: int,
    expected_emotion: str,
    prompt: str,
    start_token: int,
    pooling_strategy: str,
    pool_size: int,
    metadata: dict | None = None,
) -> dict:
    activation = capture_hidden_mean(
        loaded,
        prompt,
        layer_idx=layer_idx,
        start_token=start_token,
        pooling_strategy=pooling_strategy,
        pool_size=pool_size,
    )
    record = {
        "expected_emotion": expected_emotion,
        "prompt": prompt,
        "start_token": start_token,
        "token_count": token_count(loaded.processor, prompt),
        "activation": activation,
    }
    if metadata:
        record.update(metadata)
    return record


def score_probe_record(
    record: dict,
    *,
    vectors: dict[str, torch.Tensor],
    scoring: str,
    activation_offset: torch.Tensor | None = None,
) -> dict:
    projection_scores = compute_projection_scores(
        record["activation"],
        vectors,
        scoring=scoring,
        activation_offset=activation_offset,
    )
    output = {key: value for key, value in record.items() if key != "activation"}
    output["top_matches"] = ranked_matches(projection_scores, top_k=len(vectors))
    return output


def main() -> None:
    args = parse_args()
    payload = load_vector_payload(args.vectors)
    vectors: dict[str, torch.Tensor] = payload["vectors"]
    layer_idx = int(payload["layer_idx"])
    start_token = int(payload.get("start_token", 0))
    prompt_probe_start_token = (
        args.prompt_probe_start_token
        if args.prompt_probe_start_token is not None
        else 0
    )
    story_probe_start_token = (
        args.story_probe_start_token
        if args.story_probe_start_token is not None
        else start_token
    )
    pooling_strategy = payload.get("pooling_strategy", "mean")
    pool_size = int(payload.get("pool_size", 32))
    available_emotions = set(vectors)

    loaded = load_model(model_id=args.model_id)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    vocab_matrix = loaded.model.lm_head.weight.detach().float().cpu()
    tokenizer = loaded.processor

    logit_lens = {}
    for emotion, vector in vectors.items():
        logits = vocab_matrix @ vector.float()
        top_ids = torch.topk(logits, k=args.top_k).indices.tolist()
        bottom_ids = torch.topk(-logits, k=args.top_k).indices.tolist()
        logit_lens[emotion] = {
            "top_tokens": [tokenizer.decode([token_id]).strip() for token_id in top_ids],
            "bottom_tokens": [tokenizer.decode([token_id]).strip() for token_id in bottom_ids],
        }
    write_json(args.report_dir / "logit_lens.json", logit_lens)

    natural_records = []
    for expected_emotion, prompt in NATURAL_PROMPTS.items():
        natural_records.append(
            extract_probe_record(
                loaded=loaded,
                layer_idx=layer_idx,
                expected_emotion=expected_emotion,
                prompt=prompt,
                start_token=prompt_probe_start_token,
                pooling_strategy=pooling_strategy,
                pool_size=pool_size,
            )
        )

    matched_records = []
    for family_id, template in MATCHED_PROMPT_FAMILIES.items():
        for expected_emotion in sorted(vectors):
            prompt = template.format(emotion=expected_emotion)
            matched_records.append(
                extract_probe_record(
                    loaded=loaded,
                    layer_idx=layer_idx,
                    expected_emotion=expected_emotion,
                    prompt=prompt,
                    start_token=prompt_probe_start_token,
                    pooling_strategy=pooling_strategy,
                    pool_size=pool_size,
                    metadata={"family_id": family_id},
                )
            )

    story_records = []
    for expected_emotion, prompts in STORY_PROBES.items():
        for probe_idx, prompt in enumerate(prompts):
            story_records.append(
                extract_probe_record(
                    loaded=loaded,
                    layer_idx=layer_idx,
                    expected_emotion=expected_emotion,
                    prompt=prompt,
                    start_token=story_probe_start_token,
                    pooling_strategy=pooling_strategy,
                    pool_size=pool_size,
                    metadata={"probe_idx": probe_idx},
                )
            )

    prompt_probe_offset = None
    story_probe_offset = None
    if args.scoring == "centered_cosine":
        prompt_probe_activations = torch.stack(
            [rec["activation"] for rec in natural_records + matched_records]
        )
        prompt_probe_offset = prompt_probe_activations.mean(dim=0)
        if story_records:
            story_probe_activations = torch.stack(
                [rec["activation"] for rec in story_records]
            )
            story_probe_offset = story_probe_activations.mean(dim=0)

    prompt_results = [
        score_probe_record(
            record,
            vectors=vectors,
            scoring=args.scoring,
            activation_offset=prompt_probe_offset,
        )
        for record in natural_records
    ]
    write_json(args.report_dir / "prompt_probe_results.json", prompt_results)

    matched_prompt_results = [
        score_probe_record(
            record,
            vectors=vectors,
            scoring=args.scoring,
            activation_offset=prompt_probe_offset,
        )
        for record in matched_records
    ]
    write_json(args.report_dir / "matched_prompt_probe_results.json", matched_prompt_results)

    story_prompt_results = [
        score_probe_record(
            record,
            vectors=vectors,
            scoring=args.scoring,
            activation_offset=story_probe_offset,
        )
        for record in story_records
    ]
    write_json(args.report_dir / "story_prompt_probe_results.json", story_prompt_results)

    prompt_probe_summary = {
        "config": {
            "layer_idx": layer_idx,
            "start_token": start_token,
            "prompt_probe_start_token": prompt_probe_start_token,
            "story_probe_start_token": story_probe_start_token,
            "pooling_strategy": pooling_strategy,
            "pool_size": pool_size,
            "construction_mode": payload.get("construction_mode", "grand_mean"),
            "pairwise_preset": payload.get("pairwise_preset"),
            "num_vectors": len(vectors),
            "scoring": args.scoring,
        },
        "natural": summarize_probe_results(prompt_results, available_emotions=available_emotions),
        "matched": summarize_probe_results(matched_prompt_results, available_emotions=available_emotions),
        "story": summarize_probe_results(story_prompt_results, available_emotions=available_emotions),
    }
    write_json(args.report_dir / "prompt_probe_summary.json", prompt_probe_summary)

    emotions = sorted(vectors)
    vector_matrix = torch.stack([vectors[emotion].float() for emotion in emotions])
    coords, explained = pca_2d(vector_matrix)

    geometry_rows = []
    for idx, emotion in enumerate(emotions):
        geometry_rows.append(
            {
                "emotion": emotion,
                "pc1": float(coords[idx, 0].item()),
                "pc2": float(coords[idx, 1].item()),
            }
        )
    csv_path = args.report_dir / "emotion_geometry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pc1", "pc2"])
        writer.writeheader()
        writer.writerows(geometry_rows)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0].numpy(), coords[:, 1].numpy())
    for idx, emotion in enumerate(emotions):
        ax.annotate(emotion, (coords[idx, 0].item(), coords[idx, 1].item()))
    ax.set_title("Emotion vectors projected to first two principal components")
    ax.set_xlabel(f"PC1 ({explained[0].item():.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1].item():.1%} variance)")
    fig.tight_layout()
    fig.savefig(args.report_dir / "emotion_geometry.png", dpi=200)

    print(json.dumps(prompt_probe_summary, indent=2))
    print(f"Wrote validation artifacts to {args.report_dir}")


if __name__ == "__main__":
    main()
