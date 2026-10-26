"""Append-only log of agent corrections ("flag incorrect classification")."""
import json
import time
from pathlib import Path

FEEDBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "flagged.jsonl"
LABELS = ["billing", "technical", "account", "other"]


def flag(message: str, predicted: str, confidence: float, correct_label: str, path: Path = FEEDBACK_PATH):
    if correct_label not in LABELS:
        raise ValueError("unknown label")
    rec = {"ts": int(time.time()), "text": message[:2000], "predicted": predicted,
           "confidence": round(confidence, 4), "correct_label": correct_label}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
