from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
DATA_PATH = ROOT / "data" / "sample_reviews.csv"
METRICS_PATH = ROOT / "reports" / "training_metrics.json"

st.set_page_config(
    page_title="INSIGHT | Customer Voice",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0b1020"
PANEL = "#121d32"
MUTED = "#9aabc5"
TEAL = "#63d6c2"
COLORS = {"Positive": "#58c6a9", "Neutral": "#e6b85c", "Negative": "#ed777e"}
ORDER = ["Positive", "Neutral", "Negative"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
.stApp {background: radial-gradient(ellipse at 12% 0%, #172b49 0%, #0b1020 42%); color:#eef2ff}
[data-testid="stSidebar"] {background:#0d1729;border-right:1px solid #263653}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1500px}
.hero {padding:1.8rem 2rem;border:1px solid #2b4263;border-radius:22px;
background:linear-gradient(115deg,rgba(25,48,78,.98),rgba(15,24,42,.98));
box-shadow:0 18px 50px rgba(0,0,0,.18);margin-bottom:1.5rem}
.eyebrow {color:#70d6c4;letter-spacing:.18em;font-size:.72rem;font-weight:800}
.hero h1 {font-size:2.55rem;line-height:1.1;margin:.45rem 0;color:#f5f7ff;font-weight:800}
.hero p {color:#b2c0d6;margin:.4rem 0 0;font-size:1rem}
.hero-tag {display:inline-block;border:1px solid #41627d;color:#b9d9df;border-radius:30px;padding:.28rem .7rem;font-size:.72rem;margin-top:1rem}
.kpi {background:linear-gradient(145deg,#17263e,#111b2e);border:1px solid #2a3e5c;border-radius:17px;padding:1rem 1.1rem;min-height:112px}
.kpi-label {color:#9aabc5;font-size:.73rem;font-weight:700;letter-spacing:.08em}
.kpi-value {font-size:1.85rem;font-weight:800;color:#f5f7ff;margin-top:.35rem}
.kpi-foot {font-size:.72rem;color:#8297b5;margin-top:.1rem}
.section-title {font-size:1.08rem;font-weight:700;color:#eef2ff;margin:.4rem 0}
.panel {background:#111c30;border:1px solid #273b59;border-radius:17px;padding:1rem 1.2rem}
.note {border:1px solid #695735;background:rgba(83,63,28,.25);padding:.85rem 1rem;border-radius:12px;color:#eadab7}
div.stButton > button[kind="primary"] {background:linear-gradient(90deg,#37bca9,#5ed9c4);color:#081a25;border:0;font-weight:800;border-radius:10px}
.stDownloadButton button {border-radius:10px}
[data-testid="stMetric"] {background:#142139;border:1px solid #2b3e5b;padding:1rem;border-radius:14px}
[data-testid="stDataFrame"] {border:1px solid #273b59;border-radius:12px}
</style>
<div class="hero">
  <div class="eyebrow">CUSTOMER INTELLIGENCE  /  NLP ANALYTICS</div>
  <h1>INSIGHT <span style="color:#70d6c4">—</span> Customer Voice</h1>
  <p>Turn written feedback into a clearer view of customer sentiment.</p>
  <span class="hero-tag">AI / MACHINE LEARNING PROTOTYPE</span>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_demo_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["review_text", "sentiment"])
    return pd.read_csv(DATA_PATH)


with st.sidebar:
    st.markdown("## ◈ INSIGHT")
    st.caption("CUSTOMER VOICE ANALYTICS")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Executive Overview", "Live Review Lab", "Batch Analysis", "Model Insights"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.caption("Educational prototype")
    st.caption("Dataset: synthetic demo reviews")
    st.markdown("---")
    st.caption("Built with Python · Scikit-learn · Streamlit")

model = load_model()
data = load_demo_data()

if model is None:
    st.warning("The dashboard interface is ready, but the trained model is missing.")
    st.code("python train_model.py\npython -m streamlit run app.py", language="powershell")
    st.info("Run these commands from the project folder after installing requirements.txt.")
    st.stop()

if data.empty or not {"review_text", "sentiment"}.issubset(data.columns):
    st.error("The demo dataset is missing or does not contain review_text and sentiment columns.")
    st.stop()

counts = data["sentiment"].value_counts()
total = len(data)
positive_share = counts.get("Positive", 0) / total * 100 if total else 0
negative_share = counts.get("Negative", 0) / total * 100 if total else 0
neutral_share = counts.get("Neutral", 0) / total * 100 if total else 0

if page == "Executive Overview":
    st.markdown("### Executive overview")
    st.caption("A quick snapshot of the bundled sample dataset — not live business data.")
    k1, k2, k3, k4 = st.columns(4)
    metrics = [
        (k1, "REVIEWS IN SAMPLE", f"{total:,}", "Bundled demonstration records"),
        (k2, "POSITIVE SHARE", f"{positive_share:.1f}%", "Of the sample reviews"),
        (k3, "NEGATIVE SHARE", f"{negative_share:.1f}%", "Of the sample reviews"),
        (k4, "MODEL PIPELINE", "TF-IDF + LR", "Text features + classifier"),
    ]
    for col, label, value, foot in metrics:
        with col:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([0.9, 1.35], gap="large")
    with left:
        st.markdown('<div class="section-title">Sentiment mix</div>', unsafe_allow_html=True)
        pie = px.pie(
            data, names="sentiment", hole=.68,
            category_orders={"sentiment": ORDER},
            color="sentiment", color_discrete_map=COLORS,
        )
        pie.update_traces(textposition="inside", textinfo="percent")
        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf0ff", legend_title_text="", margin=dict(t=15,b=10,l=5,r=5),
        )
        st.plotly_chart(pie, use_container_width=True)
        st.caption(f"Neutral share: {neutral_share:.1f}%")
    with right:
        st.markdown('<div class="section-title">Explore sample feedback</div>', unsafe_allow_html=True)
        selected = st.selectbox("Filter by sentiment", ["All"] + ORDER)
        subset = data if selected == "All" else data[data["sentiment"] == selected]
        st.dataframe(
            subset[["review_text", "sentiment"]].head(15),
            use_container_width=True, hide_index=True,
        )
    st.markdown(
        '<div class="note"><b>Data transparency:</b> This repository includes synthetic example reviews '
        'for demonstration. The charts and percentages describe only this small sample and must not '
        'be presented as real customer research.</div>',
        unsafe_allow_html=True,
    )

elif page == "Live Review Lab":
    st.markdown("### Live review lab")
    st.caption("Test the trained model on one customer review at a time.")
    with st.container(border=True):
        review = st.text_area(
            "Paste a customer review",
            placeholder="Example: The delivery was quick and the product quality exceeded my expectations.",
            height=145,
        )
        run = st.button("Analyze review  →", type="primary", disabled=not review.strip())
    if run:
        try:
            prediction = model.predict([review])[0]
            probabilities = model.predict_proba([review])[0]
            classes = list(model.classes_)
            confidence = float(max(probabilities)) * 100
            st.markdown("#### Analysis result")
            a, b = st.columns(2)
            a.metric("Predicted sentiment", str(prediction))
            b.metric("Top-class model probability", f"{confidence:.1f}%")
            st.caption("Probability is the model's output, not a guarantee of correctness.")
            probability_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities,
            })
            bar = px.bar(
                probability_df, x="Sentiment", y="Probability",
                color="Sentiment", color_discrete_map=COLORS,
                category_orders={"Sentiment": ORDER},
            )
            bar.update_layout(
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eaf0ff", showlegend=False,
                margin=dict(t=20,b=10,l=5,r=5),
            )
            st.plotly_chart(bar, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

elif page == "Batch Analysis":
    st.markdown("### Batch review analysis")
    st.caption("Upload a CSV with a column named review_text to classify multiple reviews.")
    st.download_button(
        "Download sample CSV template",
        data=pd.DataFrame({"review_text": [
            "The product quality is excellent and delivery was fast.",
            "The order arrived late and customer support was unhelpful.",
        ]}).to_csv(index=False).encode("utf-8"),
        file_name="review_upload_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload review data (CSV)", type=["csv"])
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            if "review_text" not in batch.columns:
                st.error("Required column not found: review_text")
            else:
                batch = batch.dropna(subset=["review_text"]).copy()
                if batch.empty:
                    st.warning("No non-empty reviews were found in this file.")
                else:
                    batch["review_text"] = batch["review_text"].astype(str)
                    batch["predicted_sentiment"] = model.predict(batch["review_text"])
                    st.success(f"Successfully analyzed {len(batch):,} reviews.")
                    summary = batch["predicted_sentiment"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Positive", int(summary.get("Positive", 0)))
                    c2.metric("Neutral", int(summary.get("Neutral", 0)))
                    c3.metric("Negative", int(summary.get("Negative", 0)))
                    st.dataframe(batch, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download analyzed CSV",
                        batch.to_csv(index=False).encode("utf-8"),
                        file_name="insight_review_predictions.csv",
                        mime="text/csv",
                        type="primary",
                    )
                    hist = px.histogram(
                        batch, x="predicted_sentiment", color="predicted_sentiment",
                        color_discrete_map=COLORS, category_orders={"predicted_sentiment": ORDER},
                    )
                    hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#eaf0ff", showlegend=False,
                    )
                    st.plotly_chart(hist, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

else:
    st.markdown("### Model insights")
    st.caption("How the prototype works, what it measures, and where caution is needed.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### The workflow")
        st.markdown("""
1. **Text input** — customer review text enters the pipeline.
2. **TF-IDF** — words and phrases are converted into numerical features.
3. **Logistic Regression** — the classifier predicts a sentiment label.
4. **Output** — Positive, Neutral, or Negative prediction with model probabilities.
""")
        st.markdown("#### Training setup")
        st.markdown("""
- Stratified train/test split (75/25)
- Fixed random seed: 42
- Evaluation report: accuracy, precision, recall, F1-score and confusion matrix
""")
    with b:
        st.markdown("#### Important limitations")
        st.markdown("""
- The bundled examples are synthetic and English-language only.
- A small demo dataset cannot establish real-world model performance.
- Sarcasm, context, domain-specific language and class imbalance may affect predictions.
- Human review is recommended before making business decisions.
""")
        if METRICS_PATH.exists():
            st.markdown("#### Latest local training report")
            try:
                st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                st.info("A training report exists but could not be read.")
        else:
            st.info("No training metrics file yet. Run python train_model.py to generate it.")

st.markdown("---")
st.caption("INSIGHT Customer Voice Analytics · Educational portfolio project · Synthetic demonstration data")
