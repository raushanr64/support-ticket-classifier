"""Inference wrapper: input validation, prediction, uncertainty handling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "classifier.joblib"
MAX_CHARS = 2000
MIN_CHARS = 3
# Below this top-class probability the prediction is shown as "uncertain".
# Chosen from the challenge set: catches 3/5 errors at the cost of flagging
# 9/35 correct ones (see README).
UNCERTAIN_THRESHOLD = 0.50


class InputError(ValueError):
    """Raised for blank / invalid user input. Message is safe to show."""


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict
    uncertain: bool


def validate_message(text) -> str:
    if text is None or not isinstance(text, str):
        raise InputError("Please enter a support message.")
    text = " ".join(text.split())  # collapse whitespace / control chars
    if len(text) < MIN_CHARS:
        raise InputError("The message is empty or too short to classify.")
    if len(text) > MAX_CHARS:
        raise InputError(f"The message is too long ({len(text)} chars). Limit is {MAX_CHARS}.")
    if not any(c.isalpha() for c in text):
        raise InputError("The message has no readable text.")
    return text


_model = None


def load_model(path: Path = MODEL_PATH):
    global _model
    if _model is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                "Trained model not found. Run `python -m src.make_dataset && python -m src.train`.")
        _model = joblib.load(path)
    return _model


def predict(text: str, model=None, threshold: float = UNCERTAIN_THRESHOLD) -> Prediction:
    text = validate_message(text)
    model = model or load_model()
    proba = model.predict_proba([text])[0]
    probs = {c: float(p) for c, p in zip(model.classes_, proba)}
    label = max(probs, key=probs.get)
    conf = probs[label]
    return Prediction(label, conf, probs, uncertain=conf < threshold)
