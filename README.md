# AI Support Ticket Classifier (StarvoxLabs technical evaluation)

A small, working MVP: paste a customer-support message, get a **category + confidence**, an **editable AI-drafted reply**, and a button to **flag a wrong classification**. Built by Raushan Kumar Raj.

> **Drafts are never sent.** The UI labels every reply "DRAFT, NOT SENT". The app has no email or messaging code at all.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) enable real LLM drafting. Without a key the app uses clearly-labelled templates.
export ANTHROPIC_API_KEY="sk-..."        # never commit this
export LLM_MODEL="claude-haiku-5-5"      # optional, this is the default

streamlit run app.py
```

The trained model (`models/classifier.joblib`) is included. To rebuild everything from scratch:

```bash
python -m src.make_dataset   # synthetic data -> data/train.csv, data/test.csv (80/20 stratified)
python -m src.train          # trains, evaluates, writes models/ and reports/
python -m src.evidence       # writes reports/example_outputs.md
python -m unittest discover -s tests -t . -v
```

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a public GitHub repo (`models/classifier.joblib` must be committed).
2. share.streamlit.io -> New app -> pick the repo, main file `app.py`.
3. Settings -> Secrets -> add `ANTHROPIC_API_KEY = "..."` (optional; see `.streamlit/secrets.toml.example`).

## Architecture

```
message -> validate (blank/length/no-letters) -> TF-IDF(1-2gram) + LogisticRegression
        -> category + probability -> low-confidence flag (< 50%)
        -> LLM prompt (category-specific guidance, message treated as untrusted data)
              -> on any failure: labelled template fallback
        -> UI: category, confidence, editable draft, "Flag as incorrect" -> data/flagged.jsonl
```

| File | Role |
|---|---|
| `src/make_dataset.py` | Synthetic data generator + stratified split |
| `src/train.py` | Training, evaluation, baseline, confusion matrices |
| `src/classifier.py` | Validation + inference + uncertainty |
| `src/llm.py` | LLM drafting (stdlib HTTP, no SDK) + fallbacks |
| `src/feedback.py` | Append-only flag log |
| `app.py` | Streamlit UI |

**Choices:** TF-IDF + Logistic Regression: tiny, fast on CPU, gives calibrated-ish probabilities, trains in under a second. Anthropic Claude Haiku-class model for drafting: cheap and quick; swappable via `LLM_MODEL`. Streamlit: fastest path to an editable UI with free hosting.

## Results

| | Majority baseline | Model |
|---|---|---|
| Held-out synthetic test set (n=120) | 31.7% (always "technical") | **100%** (macro-F1 1.00) |
| Hand-written challenge set (n=40) | 25.0% | **87.5%** (macro-F1 ~0.88) |

Confusion matrices: `reports/confusion_matrix_test.png`, `reports/confusion_matrix_challenge.png`. Raw numbers: `reports/metrics.json`. The test set was split before training and is used only for evaluation (a unit test checks there is no text overlap).

**Read the 100% with suspicion.** The synthetic data is generated from a limited set of phrases, so the random test split shares phrases with training. That number mostly proves the pipeline works, not that it will generalise. The 40-message challenge set was written by hand, separately from the generator, and **87.5% is the realistic figure**. Even that is small (n=40), so expect roughly +/-10 points of uncertainty.

**Uncertainty:** predictions below 50% confidence show a warning. On the challenge set this catches 3 of 5 errors but also flags 9 of 35 correct predictions. Mean confidence is 0.69 when correct vs 0.51 when wrong, so confidence is informative but imperfect.

## Testing evidence

`python -m unittest discover -s tests -t . -v` runs 15 tests (all passing when written): blank/whitespace/too-short/too-long/non-text input rejected; both demo scenarios; LLM no-key, HTTP 429, network down, timeout, connection reset, malformed JSON, empty reply all fall back to a labelled template; the prompt wraps the message as untrusted data; flag log round-trip; train/test overlap check.

Example outputs, including failures: `reports/example_outputs.md`.

| Demo input | Result |
|---|---|
| "I was charged twice for my subscription" | billing, 98% |
| "I cannot log in after resetting my password" | account, 94% |
| "asdf qwerty zxcv" | technical, 31%, **uncertain warning** |
| "   " | rejected, "empty or too short" |

## Known limitations (honest list)

- **Synthetic data only**, 600 messages, English only. Real tickets are messier and often span several categories (e.g. a billing problem that locks an account). Single-label classification cannot express that.
- **No out-of-domain class.** Off-topic text such as "what is the weather" is still forced into a category (billing at 56%, which is above the warning threshold). A real system needs an "unknown" class or an outlier detector.
- The 50% threshold was picked by looking at 40 examples; it is a heuristic, not a tuned operating point.
- Prompt-injection is mitigated (message wrapped as data, output is only a draft a human reads) but not eliminated.
- `data/flagged.jsonl` is local to the app instance. On Streamlit Cloud the disk is ephemeral, so flags can be lost on restart. A database would be the next step. Flagged text is stored unredacted, so don't enter personal data.
- No authentication or rate limiting on the public demo.
- The LLM path was unit-tested with mocked responses. Live output quality depends on the API key you configure.

## Approximate running cost

- Hosting: $0 (Streamlit Community Cloud). Classifier: $0 (CPU, milliseconds).
- LLM: about 300 input + 200 output tokens per draft. With a Haiku-class model that is a fraction of a cent per ticket, so roughly $1 per few thousand drafts at typical list prices (check current pricing). With no key set, cost is $0 and templates are used.

## Data & security notes

All data is synthetic or hand-written by the author, with no personal or confidential data. No secrets are committed (`.gitignore` excludes `.env`, `secrets.toml`, and flag logs). Inputs are length-limited and validated; the LLM call has a 20s timeout and never raises into the UI.

## AI assistance disclosure

This project was built with the help of an AI coding assistant (permitted by the brief). The author is expected to be able to explain and modify all of it: the generator, the TF-IDF + LR pipeline, the threshold logic, and the fallback design.
