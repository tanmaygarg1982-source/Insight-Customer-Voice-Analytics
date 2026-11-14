# INSIGHT — Customer Voice Analytics

An educational customer-feedback analytics dashboard that classifies review text into Positive, Neutral, or Negative sentiment.

## Features
- Interactive Streamlit dashboard with sentiment distribution and review samples
- Single-review sentiment prediction
- Batch CSV analysis with downloadable predictions
- TF-IDF text features and Logistic Regression classifier
- Reproducible training script and evaluation report

## Tech stack
Python · Pandas · Scikit-learn · TF-IDF · Logistic Regression · Streamlit · Plotly

## Quick start
Python 3.10+ is recommended.

Create and activate a virtual environment:

Windows PowerShell:
    python -m venv .venv
    .venv\Scripts\activate

macOS/Linux:
    python -m venv .venv
    source .venv/bin/activate

Install and run:

    pip install -r requirements.txt
    python train_model.py
    streamlit run app.py

## Batch upload format
Upload a CSV containing a text column named review_text.

    review_text
    "The delivery was quick and the product works perfectly."
    "The item arrived late and was damaged."

## Project structure
- app.py — Streamlit dashboard
- train_model.py — model training and evaluation
- data/sample_reviews.csv — synthetic demo examples
- models/ — generated model artifact
- reports/ — generated training metrics

## Data and responsible-use note
The included sample reviews are synthetic demonstration examples, not real customer feedback. Model evaluation results are only a learning demonstration and must not be presented as evidence of real-world performance. Synthetic examples may be repetitive and may not represent actual customer language, ambiguity, or class balance.

Before business use, evaluate on a representative, consented, appropriately labelled real-world dataset; check class-wise precision/recall, bias, drift, and human-review requirements.

## Limitations
- English-language demonstration data only
- Three fixed sentiment classes
- No production monitoring, authentication, or privacy controls
- Predictions are model estimates, not guaranteed interpretations

## Future improvements
- Evaluate on a real, representative labelled dataset
- Add multilingual support and stronger text preprocessing
- Add model comparison, explainability, and confidence calibration
- Add privacy, monitoring, and human-in-the-loop review


## Deploy a public dashboard (Streamlit Community Cloud)

The repository is prepared for Streamlit Community Cloud. The dashboard trains its small synthetic demonstration model automatically on first launch if the model artifact is not present.

1. Open https://share.streamlit.io/ and sign in.
2. Connect/authorize your GitHub account if prompted.
3. Choose **Create app** → **Yup, I have an app**.
4. Select repository: `tanmaygarg1982-source/Insight-Customer-Voice-Analytics`
5. Select branch: `main`
6. Set the main file path to: `app.py`
7. Optionally set an app URL such as `insight-customer-voice`, if available.
8. Click **Deploy** and wait for the build to finish.

Streamlit Community Cloud creates a shareable `.streamlit.app` URL after a successful deployment. The first launch may take a little longer because the demo model is trained from the bundled synthetic dataset.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
