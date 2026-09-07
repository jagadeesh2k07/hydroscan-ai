"""
train_model.py
================
Trains a lightweight RandomForestClassifier on the water-quality dataset and
persists the model, feature order, class labels, and global feature
importances to disk via joblib. Run automatically on first app startup if
no trained model exists yet; can also be re-run manually (e.g. `python -m
ml.train_model`) to retrain on an updated dataset.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.domain_knowledge import FEATURES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(BASE_DIR, "data", "water_quality_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "water_quality_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "data", "model_metrics.json")


def train(dataset_path: str = DEFAULT_DATASET, model_path: str = MODEL_PATH,
          n_estimators: int = 200, max_depth: int = 8, random_state: int = 42):
    df = pd.read_csv(dataset_path)

    missing = [f for f in FEATURES + ["label"] if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=FEATURES + ["label"])

    X = df[FEATURES].astype(float)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_).tolist()

    global_importance = dict(zip(FEATURES, clf.feature_importances_.tolist()))

    bundle = {
        "model": clf,
        "features": FEATURES,
        "classes": clf.classes_.tolist(),
        "global_importance": global_importance,
        "trained_on": dataset_path,
        "n_records": len(df),
        "accuracy": acc,
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)

    metrics = {
        "accuracy": round(acc, 4),
        "n_records": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": clf.classes_.tolist(),
        "confusion_matrix": cm,
        "classification_report": report,
        "global_importance": global_importance,
        "dataset_path": dataset_path,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = train()
    print(f"Trained on {m['n_records']} records | Test accuracy: {m['accuracy']:.3f}")
    print("Global feature importance:", m["global_importance"])
