"""Streamlit UI: AI Support Ticket Classifier (StarvoxLabs technical evaluation)."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src import classifier, feedback, llm

ROOT = Path(__file__).parent
SAMPLES = [
    "I was charged twice for my subscription",
    "I cannot log in after resetting my password",
    "The app crashes every time I open the dashboard",
    "Do you offer discounts for non-profits?",
]

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="wide")

# Optional: pull the API key from Streamlit secrets into the environment (never committed).
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass


@st.cache_resource
def get_model():
    return classifier.load_model()


def set_sample(text):
    st.session_state["msg"] = text


st.title("🎫 AI Support Ticket Classifier")
st.caption("Classifies a support message, then drafts a reply for a human agent to review. "
           "Nothing is ever sent automatically. Use synthetic or non-personal text only.")

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

main, side = st.columns([3, 2], gap="large")

with main:
    st.subheader("1. Customer message")
    st.write("Try a sample:")
    cols = st.columns(len(SAMPLES))
    for c, s in zip(cols, SAMPLES):
        c.button(s if len(s) < 34 else s[:31] + "...", key=f"s_{s}", on_click=set_sample, args=(s,),
                 use_container_width=True)
    st.text_area("Message", key="msg", height=120, max_chars=classifier.MAX_CHARS,
                 placeholder="Paste a support message here...")

    if st.button("Classify and draft reply", type="primary"):
        st.session_state.pop("flagged", None)
        try:
            pred = classifier.predict(st.session_state.get("msg", ""), model=model)
        except classifier.InputError as e:
            st.session_state.pop("result", None)
            st.warning(str(e))
        else:
            with st.spinner("Drafting a suggested reply..."):
                draft = llm.draft_response(" ".join(st.session_state["msg"].split()), pred.label)
            st.session_state["result"] = {"pred": pred, "draft": draft,
                                          "message": " ".join(st.session_state["msg"].split())}
            st.session_state["draft_text"] = draft.text

    res = st.session_state.get("result")
    if res:
        pred, draft = res["pred"], res["draft"]
        st.subheader("2. Prediction")
        m1, m2 = st.columns(2)
        m1.metric("Predicted category", pred.label.capitalize())
        m2.metric("Confidence", f"{pred.confidence:.0%}")
        st.progress(min(max(pred.confidence, 0.0), 1.0))
        if pred.uncertain:
            st.warning(f"Low confidence (below {classifier.UNCERTAIN_THRESHOLD:.0%}). "
                       "The category may be wrong, so please check it before using the draft.")
        st.bar_chart(pd.Series(pred.probabilities, name="probability"))

        st.subheader("3. Suggested reply (editable draft)")
        st.info("DRAFT, NOT SENT. Review and edit before sending it yourself.")
        if draft.warning:
            st.warning(draft.warning)
        st.caption("Source: " + ("LLM-generated" if draft.source == "llm" else "generic template (fallback)"))
        st.text_area("Draft", key="draft_text", height=240, label_visibility="collapsed")

        st.subheader("4. Wrong category?")
        labels = feedback.LABELS
        fc1, fc2 = st.columns([2, 1])
        correct = fc1.selectbox("Correct category", labels, index=labels.index(pred.label))
        if fc2.button("Flag as incorrect", use_container_width=True):
            if correct == pred.label:
                st.info("Pick a different category to flag a correction.")
            else:
                feedback.flag(res["message"], pred.label, pred.confidence, correct)
                st.session_state["flagged"] = correct
        if st.session_state.get("flagged"):
            st.success(f"Thanks, flagged as '{st.session_state['flagged']}'. Saved for review.")

with side:
    st.subheader("Model evidence")
    mp = ROOT / "reports" / "metrics.json"
    if mp.exists():
        met = json.loads(mp.read_text())
        a, b = st.columns(2)
        a.metric("Held-out test accuracy", f"{met['test']['accuracy']:.1%}",
                 f"baseline {met['test_baseline']['accuracy']:.1%}", delta_color="off")
        b.metric("Hand-written challenge", f"{met['challenge']['accuracy']:.1%}",
                 f"baseline {met['challenge_baseline']['accuracy']:.1%}", delta_color="off")
        st.caption("Test set is synthetic and template-based, so near-perfect accuracy is expected "
                   "and optimistic. The hand-written challenge set is the more honest number.")
        for name, f in [("Test", "confusion_matrix_test.png"), ("Challenge", "confusion_matrix_challenge.png")]:
            p = ROOT / "reports" / f
            if p.exists():
                st.image(str(p), caption=f"Confusion matrix: {name}")
    else:
        st.info("Run `python -m src.train` to generate metrics.")
