"""Train + evaluate the ticket classifier.

TF-IDF (word 1-2 grams) + Logistic Regression. The held-out test set
(data/test.csv) is used ONLY for the final evaluation, never for fitting
or tuning. Also reports a majority-class baseline and a hand-written
challenge set.

Usage: python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["billing", "technical", "account", "other"]


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=10, max_iter=1000, class_weight="balanced")),
    ])


def evaluate(model, df):
    pred = model.predict(df["text"])
    return {
        "n": int(len(df)),
        "accuracy": round(accuracy_score(df["label"], pred), 4),
        "macro_f1": round(f1_score(df["label"], pred, average="macro"), 4),
        "confusion_matrix": confusion_matrix(df["label"], pred, labels=LABELS).tolist(),
        "report": classification_report(df["label"], pred, labels=LABELS, output_dict=True, zero_division=0),
    }


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), LABELS, rotation=30)
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    challenge = pd.read_csv(ROOT / "data/challenge.csv")
    assert not set(train["text"]) & set(test["text"]), "train/test leakage"

    model = make_pipeline().fit(train["text"], train["label"])
    baseline = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])

    metrics = {
        "labels": LABELS,
        "majority_class": str(baseline.predict(test["text"])[0]),
        "test": evaluate(model, test),
        "test_baseline": {"accuracy": round(accuracy_score(test["label"], baseline.predict(test["text"])), 4)},
        "challenge": evaluate(model, challenge),
        "challenge_baseline": {"accuracy": round(accuracy_score(challenge["label"], baseline.predict(challenge["text"])), 4)},
    }
    # Calibration view: confidence on correct vs wrong challenge predictions
    proba = model.predict_proba(challenge["text"])
    conf = proba.max(axis=1)
    ok = model.predict(challenge["text"]) == challenge["label"].values
    metrics["challenge_confidence"] = {
        "mean_when_correct": round(float(conf[ok].mean()), 3),
        "mean_when_wrong": round(float(conf[~ok].mean()), 3) if (~ok).any() else None,
    }

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/classifier.joblib")
    (ROOT / "reports/metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_cm(metrics["test"]["confusion_matrix"], "Held-out test set", ROOT / "reports/confusion_matrix_test.png")
    plot_cm(metrics["challenge"]["confusion_matrix"], "Hand-written challenge set", ROOT / "reports/confusion_matrix_challenge.png")

    print(f"Majority baseline ({metrics['majority_class']}): test acc {metrics['test_baseline']['accuracy']}")
    print(f"Model test acc {metrics['test']['accuracy']}  macro-F1 {metrics['test']['macro_f1']}  (n={metrics['test']['n']})")
    print(f"Challenge acc {metrics['challenge']['accuracy']} vs baseline {metrics['challenge_baseline']['accuracy']} (n={metrics['challenge']['n']})")
    print("Test confusion matrix (rows=true, cols=pred, order=%s):" % LABELS)
    for row in metrics["test"]["confusion_matrix"]:
        print(row)
    print("Challenge confusion matrix:")
    for row in metrics["challenge"]["confusion_matrix"]:
        print(row)
    print("Challenge confidence:", metrics["challenge_confidence"])


if __name__ == "__main__":
    main()
