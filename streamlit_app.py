import os
import sys
import datetime
import pandas as pd
import streamlit as st


_this_dir = os.path.dirname(os.path.abspath(__file__))
for _sub in ('lstm', 'svm', 'naive_bayes'):
    _p = os.path.abspath(os.path.join(_this_dir, '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️", layout="centered")


REVIEW_LOG_PATH = os.path.join(_this_dir, 'ui_human_reviewed_log.csv')


def log_human_review(result, human_decision):
    entry = pd.DataFrame([{
        "model": result["model"],
        "text": result["text"],
        "model_label": result.get("display_label", result["label"]),
        "confidence": result["confidence"],
        "human_decision": human_decision,
        "timestamp": datetime.datetime.now().isoformat(),
    }])
    if os.path.exists(REVIEW_LOG_PATH):
        entry.to_csv(REVIEW_LOG_PATH, mode='a', header=False, index=False)
    else:
        entry.to_csv(REVIEW_LOG_PATH, index=False)


MODELS = {}
LOAD_ERRORS = {}

try:
    import lstm_infer
    MODELS["LSTM"] = lstm_infer
except Exception as e:
    LOAD_ERRORS["LSTM"] = str(e)

try:
    import svm_infer
    MODELS["SVM"] = svm_infer
except Exception as e:
    LOAD_ERRORS["SVM"] = str(e)

try:
    import nb_infer
    MODELS["Naive Bayes"] = nb_infer
except Exception as e:
    LOAD_ERRORS["Naive Bayes"] = str(e)

# ============================================
# Header
# ============================================
st.title("🛡️ Cyberbullying Detection")
st.caption("Type a message below and check whether it's flagged as cyberbullying.")

if LOAD_ERRORS:
    with st.expander(f"⚠️ {len(LOAD_ERRORS)} model(s) unavailable — click for details", expanded=False):
        for name, err in LOAD_ERRORS.items():
            st.write(f"**{name}**: not loaded.")
            st.code(err, language=None)
        st.caption(
            "Run that model's training script first (with the retrain flag "
            "on) so its saved model file exists, then restart this app."
        )

if not MODELS:
    st.error("No trained models are available. Run lstm_model.py / svm_model.py / "
              "naive_bayes_model.py at least once each, then restart this app.")
    st.stop()

# ============================================
# Controls
# ============================================
available_model_names = list(MODELS.keys())
choice_options = available_model_names + (["All Three (Compare)"] if len(MODELS) > 1 else [])

message = st.text_area(
    "Message to check",
    placeholder="e.g. You are so stupid and worthless",
    height=100,
)

model_choice = st.radio("Model", choice_options, horizontal=True)

analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

# ============================================
# Session state: results persist across reruns so the confirm/deny buttons 
# ============================================
if "results" not in st.session_state:
    st.session_state.results = []  # list of result dicts, each with a "review_key" and "human_decision"

if analyze_clicked:
    if not message.strip():
        st.warning("Type a message first.")
        st.session_state.results = []
    else:
        with st.spinner("Analyzing..."):
            names_to_run = available_model_names if model_choice == "All Three (Compare)" else [model_choice]
            new_results = []
            for name in names_to_run:
                r = MODELS[name].predict(message)
                r["review_key"] = f"{name}_{hash(message)}"
                r["human_decision"] = None
                new_results.append(r)
            st.session_state.results = new_results

# ============================================
# Display helpers
# ============================================
LABEL_STYLE = {
    "Cyberbullying": ("🔴", "#d9534f"),
    "Needs Human Review": ("🟠", "#f0ad4e"),
    "Normal": ("🟢", "#5cb85c"),
}


def render_result(result):
    label = result.get("display_label", result["label"])
    icon, color = LABEL_STYLE.get(label, ("⚪", "#999999"))

    st.markdown(
        f"<div style='padding:0.75rem 1rem; border-radius:0.5rem; "
        f"background-color:{color}22; border:1px solid {color}; margin-bottom:0.5rem;'>"
        f"<span style='font-size:1.1rem; font-weight:600; color:{color};'>{icon} {label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    with col2:
        st.metric("Intensity", f"{result['intensity_score']:.1%}")

    st.write(f"**Severity:** {result['severity']}")
    st.write(f"**Has personal target:** {'Yes' if result['has_personal_target'] else 'No'}")

    triggers = result.get("trigger_words") or []
    if triggers:
        trigger_text = ", ".join(f"`{w}` ({lvl.split(' - ')[0]})" for w, lvl in triggers)
        st.write(f"**Trigger words:** {trigger_text}")
    else:
        st.write("**Trigger words:** none")

    if result.get("phrase_override"):
        st.caption("⚠️ Flagged via indirect-bullying phrase match (bypassed the usual threshold check).")

    if label == "Needs Human Review":
        if result["human_decision"] is None:
            st.write("**Is this actually cyberbullying?**")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("Yes, cyberbullying", key=f"yes_{result['review_key']}", use_container_width=True):
                    result["human_decision"] = "Cyberbullying"
                    log_human_review(result, "Cyberbullying")
                    st.rerun()
            with bcol2:
                if st.button("No, it's normal", key=f"no_{result['review_key']}", use_container_width=True):
                    result["human_decision"] = "Normal"
                    log_human_review(result, "Normal")
                    st.rerun()
        else:
            st.success(f"Confirmed by human reviewer: **{result['human_decision']}**")


def render_comparison(results):
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            st.markdown(f"**{result['model']}**")
            render_result(result)

if st.session_state.results:
    if len(st.session_state.results) > 1:
        render_comparison(st.session_state.results)
    else:
        render_result(st.session_state.results[0])

st.divider()
st.caption(
    "This tool is for demonstration and moderation-support purposes only — "
    "results are a model's best estimate, not a definitive judgment."
)