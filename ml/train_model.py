"""
train_model.py
================
Trains the RandomForest water-quality classifier AND validates/calibrates its
confidence (see ml/calibration.py for the maths).

Protocol
--------
1. Stratified 80/20 train/test split (test set is never used for fitting
   anything, including the calibration temperature).
2. 5-fold stratified out-of-fold (OOF) probabilities on the TRAIN split ->
   fit the temperature tau* by minimising NLL.
3. Fit the final forest on the whole train split.
4. Report on the untouched TEST split: accuracy, macro-F1, NLL, Brier, ECE
   (raw vs calibrated) and a reliability table.
5. Report 5-fold stratified cross-validated accuracy on ALL data
   (mean +/- std) as a less noisy estimate of generalisation.

Run automatically on first start if no (current) model exists; re-run
manually with `python -m ml.train_model`.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.domain_knowledge import FEATURES
from ml import calibration as cal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(BASE_DIR, "data", "water_quality_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "water_quality_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "data", "model_metrics.json")

SCHEMA_VERSION = 2   # bump when the bundle layout changes (app retrains stale bundles)


def train(dataset_path: str = DEFAULT_DATASET, model_path: str = MODEL_PATH,
          n_estimators: int = 200, max_depth: int = 8, random_state: int = 42):
    df = pd.read_csv(dataset_path)
    missing = [f for f in FEATURES + ["label"] if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=FEATURES + ["label"])

    X = df[FEATURES].astype(float)
    y = df["label"].astype(str)

    base = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=3,
        random_state=random_state, class_weight="balanced", n_jobs=1,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    classes = sorted(y.unique())
    cls_idx = {c: i for i, c in enumerate(classes)}
    ytr_idx = y_tr.map(cls_idx).to_numpy()
    yte_idx = y_te.map(cls_idx).to_numpy()

    # --- 2. temperature from out-of-fold probabilities on TRAIN only -------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof = cross_val_predict(clone(base), X_tr, y_tr, cv=skf, method="predict_proba")
    tau = cal.fit_temperature(oof, ytr_idx)

    # --- 3. final forest -----------------------------------------------------
    clf = clone(base).fit(X_tr, y_tr)
    assert list(clf.classes_) == classes

    # --- 4. held-out evaluation ---------------------------------------------
    p_raw = clf.predict_proba(X_te)
    p_cal = cal.temperature_scale(p_raw, tau)
    y_pred = np.array(classes)[p_cal.argmax(axis=1)]
    acc = float(accuracy_score(y_te, y_pred))
    macro_f1 = float(f1_score(y_te, y_pred, average="macro"))
    ece_raw, _ = cal.reliability(p_raw, yte_idx)
    ece_cal, table = cal.reliability(p_cal, yte_idx)

    # --- 5. 5-fold CV accuracy on all data -----------------------------------
    cv_scores = cross_val_score(clone(base), X, y, cv=skf, scoring="accuracy")

    validation = {
        "n_records": int(len(df)),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "cv5_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv5_accuracy_std": round(float(cv_scores.std()), 4),
        "temperature_tau": round(float(tau), 4),
        "nll_raw": round(cal.nll(p_raw, yte_idx), 4),
        "nll_calibrated": round(cal.nll(p_cal, yte_idx), 4),
        "brier_raw": round(cal.brier_multiclass(p_raw, yte_idx), 4),
        "brier_calibrated": round(cal.brier_multiclass(p_cal, yte_idx), 4),
        "ece_raw": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "mean_confidence_test": round(float(p_cal.max(axis=1).mean()), 4),
        "reliability_table": table,
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=classes).tolist(),
    }

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model": clf,
        "features": FEATURES,
        "classes": classes,
        "global_importance": dict(zip(FEATURES, clf.feature_importances_.tolist())),
        "calibration_temperature": float(tau),
        "validation": validation,
        "trained_on": dataset_path,
        "n_records": int(len(df)),
        "accuracy": acc,
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)

    with open(METRICS_PATH, "w") as f:
        json.dump(validation, f, indent=2)
    return validation


if __name__ == "__main__":
    v = train()
    print(json.dumps({k: v[k] for k in v if k not in ("reliability_table", "confusion_matrix")}, indent=2))
