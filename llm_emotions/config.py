import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STORIES_DIR = DATA_DIR / "stories"
VECTORS_DIR = PROJECT_ROOT / "vectors"
REPORTS_DIR = PROJECT_ROOT / "reports"

LIGHTWEIGHT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MODEL_ID = os.environ.get("EMOTION_STEERING_MODEL_ID", LIGHTWEIGHT_MODEL_ID)
DEFAULT_STORIES_PER_EMOTION = 60
DEFAULT_REVIEW_SAMPLE_FRACTION = 0.10

DEFAULT_EMOTION_PATH = DATA_DIR / "emotions.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful writing assistant helping generate emotionally distinct short stories "
    "for interpretability research."
)
