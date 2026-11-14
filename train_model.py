"""Train a TF-IDF + Logistic Regression sentiment classifier.
The bundled dataset is synthetic and intended only for demonstration.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_reviews.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    required = {"review_text", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    df = df.dropna(subset=["review_text", "sentiment"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].ne("")]
    if df["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"], df["sentiment"], test_size=0.25,
        random_state=42, stratify=df["sentiment"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                  max_features=12000, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    labels = sorted(df["sentiment"].unique())
    report = {
        "dataset_type": "synthetic demonstration data",
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": labels,
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }
    (REPORT_DIR / "training_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Demonstration holdout accuracy: {report['accuracy']:.3f}")
    print("Reminder: synthetic-data metrics are not real-world performance evidence.")


if __name__ == "__main__":
    main()
