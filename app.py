import os
import io
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# PATHS
SVM_MODEL_PATH  = "outputs/linear/linear_model.pkl"
VECTORIZER_PATH = "outputs/vectorizer.pkl"

# THE KEYS
RESET_KEY = "reset_counter"
PRED_KEY  = "last_pred"

# UTILITIES
def inject_css(path="styles.css"):
    """Inject custom CSS."""
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model_and_vectorizer():
    if not os.path.exists(SVM_MODEL_PATH):
        st.error(f"Missing model: {SVM_MODEL_PATH}")
        st.stop()
    if not os.path.exists(VECTORIZER_PATH):
        st.error(f"Missing vectorizer: {VECTORIZER_PATH}")
        st.stop()
    return _load_pickle(SVM_MODEL_PATH), _load_pickle(VECTORIZER_PATH)

# SIGMOID | # SPAM = 1 AND  HAM = 0
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def predict_label(text: str, model, vectorizer):
    X = vectorizer.transform([text])
    if getattr(X, "nnz", 0) == 0 or len(text.strip().split()) < 2:
        return None, None, None, "Message too short for a reliable prediction."

    score = float(model.decision_function(X)[0])
    p = sigmoid(score)
    y = int(model.predict(X)[0])
    conf = p if y == 1 else (1 - p)
    return y, conf, score, None

# CLEAR FUNCTION
def _clear_all():
    st.session_state[RESET_KEY] = st.session_state.get(RESET_KEY, 0) + 1
    st.session_state[PRED_KEY] = None
    st.session_state["do_clear"] = True

def render_sigmoid_card(label: str, conf: float, score: float):
    import io
    x = np.linspace(-6, 6, 200)
    y = 1.0 / (1.0 + np.exp(-x))
    accent = "#ff0000" if label == "SPAM" else "#3de5ad"

    # SIGMOID CURVE
    fig, ax = plt.subplots(figsize=(3.25, 2.1), dpi=160)
    ax.plot(x, y, linewidth=2, color="#0F172A")
    ax.axvline(score, linestyle="--", linewidth=1, color=accent, alpha=0.85)
    ax.scatter([score], [1.0 / (1.0 + np.exp(-score))], s=28, color=accent, zorder=3)
    ax.set_xlim(-6, 6); ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title("SIGMOID MARGIN", fontsize=10, pad=6)
    ax.grid(alpha=0.15)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig); buf.seek(0)

    # CARD CONTENT
    st.markdown('<div style="margin:0;padding:0;">', unsafe_allow_html=True)
    st.image(buf, use_container_width=True)

    # COLORED CONFIDENCE LABEL
    pct_num = int(round(min(1.0, conf or 0.0) * 100))
    pct_text = f"{pct_num}%"
    st.markdown(
        f"""
        <div style="text-align:center;font-size:16px;font-weight:800;
                    color:{accent};margin-top:6px;margin-bottom:10px;">
            CONFIDENCE&nbsp;&nbsp;{pct_text}
        </div>
        """,
        unsafe_allow_html=True,)

    # PROGRESS BAR
    st.markdown(
        f"""
        <div style="
            height:12px; width:100%;
            background:rgba(0,0,0,0.06);
            border-radius:999px; overflow:hidden;">
         <div style="
              height:12px; width:{pct_num}%;
              background:{accent};
              border-radius:999px;">
          </div>
        </div>
        """,
        unsafe_allow_html=True,)

    st.markdown("</div>", unsafe_allow_html=True)



# MAIN
def main():
    st.set_page_config(page_title="SPAM vs HAM", page_icon="📬", layout="centered")
    inject_css()

    
    st.session_state.setdefault(RESET_KEY, 0)
    st.session_state.setdefault(PRED_KEY, None)
    st.session_state.setdefault("do_clear", False)

    # SIDEBAR SIGMOID
    with st.sidebar:
        st.markdown(
            '<div class="browser-dots">'
            '<span class="dot red"></span>'
            '<span class="dot yellow"></span>'
            '<span class="dot green"></span>'
            '</div>',
            unsafe_allow_html=True,)

        res = st.session_state.get(PRED_KEY)
        if res:
            label, conf, score = res
            if label != "warn" and conf is not None and score is not None:
                render_sigmoid_card(label, conf, score)

    # THE TITLE
    st.markdown('<div class="title-center">SPAM vs HAM</div>', unsafe_allow_html=True)

    # THE TEXT AREA
    ta_key = f"input_text_{st.session_state[RESET_KEY]}"
    user_text = st.text_area(
        label="",
        key=ta_key,
        height=180,
        label_visibility="collapsed",
        placeholder="",)

    # THE BUTTONS
    c1, c2 = st.columns([1, 1])
    with c1:
        go = st.button("Predict", type="primary", use_container_width=True, key="predict_btn")
    with c2:
        st.button("Clear", use_container_width=True, key="clear_btn", on_click=_clear_all)

    # PREDICT
    if go:
        if not user_text.strip():
            st.warning("Please paste some text first.")
        else:
            model, vectorizer = load_model_and_vectorizer()
            y, conf, score, warn = predict_label(user_text, model, vectorizer)
            if warn:
                st.session_state[PRED_KEY] = ("warn", None, None)
            else:
                st.session_state[PRED_KEY] = ("SPAM" if y == 1 else "HAM", conf, score)
            st.rerun()

    # TO CENTER THE RESULT
    result = st.session_state[PRED_KEY]
    if result:
        label, conf, score = result
        if label == "warn":
            st.warning("MESSAGE IS TOO SHORT FOR RELIABLE PREDICTION.")
        else:
            (st.error if label == "SPAM" else st.success)(f"**{label}**")

    if st.session_state["do_clear"]:
        st.session_state["do_clear"] = False
        st.rerun()

# RUN
if __name__ == "__main__":
    main()
