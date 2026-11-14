from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
DATA_PATH = ROOT / "data" / "sample_reviews.csv"

st.set_page_config(page_title="INSIGHT | Customer Voice", page_icon="◈", layout="wide")
st.markdown("""
<style>
.stApp {background:#0b1020;color:#eef2ff}
[data-testid="stSidebar"] {background:#111a2e}
.block-container {padding-top:2rem;max-width:1450px}
.hero {padding:1.5rem 1.8rem;border:1px solid #263653;border-radius:18px;
background:linear-gradient(120deg,#14233d,#101729);margin-bottom:1.2rem}
.eyebrow {color:#70d6c4;letter-spacing:.16em;font-size:.75rem;font-weight:700}
.hero h1 {font-size:2.35rem;margin:.25rem 0;color:#f5f7ff}
.hero p {color:#aab8d0;margin:0}
.metric {background:#131e33;border:1px solid #263653;border-radius:14px;padding:1rem 1.2rem}
.metric-label {color:#9aabc5;font-size:.85rem}
.metric-value {font-size:1.8rem;font-weight:700;color:#f5f7ff}
.note {border-left:3px solid #e5b65c;background:#211f1a;padding:.8rem 1rem;border-radius:8px;color:#e9d9b7}
</style>
<div class="hero">
<div class="eyebrow">CUSTOMER INTELLIGENCE / NLP ANALYTICS</div>
<h1>INSIGHT <span style="color:#70d6c4">—</span> Customer Voice</h1>
<p>Explore customer feedback with an educational machine-learning sentiment prototype.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.warning("Model not found. In the project folder, run: python train_model.py")
    st.stop()

@st.cache_data
def load_demo_data():
    return pd.read_csv(DATA_PATH)

data = load_demo_data()
labels = ["Positive", "Neutral", "Negative"]
counts = data["sentiment"].value_counts()
total = len(data)
positive_share = counts.get("Positive", 0) / total * 100 if total else 0

with st.sidebar:
    st.markdown("## ◈ INSIGHT")
    page = st.radio("WORKSPACE", ["Overview", "Review Lab", "Batch Analysis", "Model & Method"])
    st.caption("Customer Voice Analytics")
    st.divider()
    st.caption("Prototype • Synthetic sample data")

if page == "Overview":
    st.markdown("### Executive overview")
    st.caption("A snapshot of the bundled demonstration dataset.")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "SAMPLE REVIEWS", f"{total:,}"),
        (c2, "SENTIMENT CLASSES", str(data["sentiment"].nunique())),
        (c3, "POSITIVE SHARE", f"{positive_share:.1f}%"),
        (c4, "MODEL", "TF-IDF + LR"),
    ]:
        with col:
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("#### Sentiment distribution")
        fig = px.pie(data, names="sentiment", hole=.62, color="sentiment",
                     color_discrete_map={"Positive":"#58c6a9","Neutral":"#e6b85c","Negative":"#ed777e"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#eaf0ff", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Review explorer")
        choice = st.selectbox("Filter sentiment", ["All"] + labels)
        view = data if choice == "All" else data[data["sentiment"] == choice]
        st.dataframe(view[["review_text", "sentiment"]].head(12),
                     use_container_width=True, hide_index=True)
    st.markdown('<div class="note"><b>Data note:</b> These are synthetic demonstration reviews. Charts describe this sample only and do not represent actual customer opinion.</div>', unsafe_allow_html=True)

elif page == "Review Lab":
    st.markdown("### Review lab")
    st.caption("Enter one review to generate a model prediction.")
    review = st.text_area("Customer review", placeholder="Type or paste a customer review...", height=150)
    if st.button("Analyze sentiment", type="primary", disabled=not review.strip()):
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        classes = model.classes_
        confidence = float(max(probabilities)) * 100
        st.markdown("#### Prediction")
        a, b = st.columns(2)
        a.metric("Predicted sentiment", str(prediction))
        b.metric("Model probability (top class)", f"{confidence:.1f}%")
        st.caption("Model probabilities are not calibrated confidence guarantees.")
        chart = pd.DataFrame({"Sentiment": classes, "Probability": probabilities})
        st.plotly_chart(px.bar(chart, x="Sentiment", y="Probability", color="Sentiment"),
                        use_container_width=True)

elif page == "Batch Analysis":
    st.markdown("### Batch analysis")
    st.caption("Upload a CSV file containing a column named review_text.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        if "review_text" not in batch.columns:
            st.error("The uploaded CSV must contain a 'review_text' column.")
        else:
            batch = batch.dropna(subset=["review_text"]).copy()
            batch["predicted_sentiment"] = model.predict(batch["review_text"].astype(str))
            st.success(f"Analyzed {len(batch):,} reviews.")
            st.dataframe(batch, use_container_width=True, hide_index=True)
            st.download_button("Download predictions CSV",
                               batch.to_csv(index=False).encode("utf-8"),
                               file_name="sentiment_predictions.csv", mime="text/csv")
            st.plotly_chart(px.histogram(batch, x="predicted_sentiment",
                                         color="predicted_sentiment"),
                            use_container_width=True)

else:
    st.markdown("### Model & method")
    st.markdown("""
**Workflow**
1. Review text is converted into numerical features using TF-IDF.
2. Logistic Regression learns patterns from labelled examples.
3. The trained pipeline predicts Positive, Neutral, or Negative sentiment.

**Training / evaluation**
- Stratified 75/25 train-test split
- Fixed random seed: 42
- Accuracy, class-wise precision/recall/F1, and confusion matrix saved in reports/training_metrics.json

**Limitations**
- Bundled examples are synthetic and English-language only.
- Results are a learning demonstration, not validated business intelligence.
- Sarcasm, context, domain language, and class imbalance can reduce performance.
- Human review is recommended before acting on predictions.

**Next steps**
- Evaluate on a representative labelled customer dataset.
- Compare models and calibrate probabilities.
- Add multilingual analysis, explainability, privacy controls, and drift monitoring.
""")
    metrics_path = ROOT / "reports" / "training_metrics.json"
    if metrics_path.exists():
        st.markdown("#### Latest training report")
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
