"""
dataset_generator.py
=====================
Generates a realistic synthetic water-quality dataset for training the
HydroScan AI lightweight classifier.

Each record has: ph, chlorine, hardness, nitrate, label
Labels are derived from domain thresholds (see domain_knowledge.py) combined
with a severity-scoring rule and a small amount of label noise, so the
resulting dataset is not trivially linearly separable — this gives the
RandomForest model genuine, non-trivial patterns to learn and gives SHAP
values real signal to explain, rather than acting as a plain lookup table.
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.domain_knowledge import STATUS_RANK, parameter_statuses

RNG_SEED = 42


def _sample_parameters(n, rng):
    """Sample parameter values from mixed distributions covering the full
    spectrum of safe, caution, and unsafe water so classes are balanced."""
    records = []

    # Roughly balance three underlying "water profiles" so we get a healthy
    # spread across Safe / Caution / Unsafe after label derivation.
    profile_choices = rng.choice(["clean", "moderate", "contaminated"], size=n,
                                  p=[0.40, 0.32, 0.28])

    for profile in profile_choices:
        if profile == "clean":
            ph = rng.normal(7.2, 0.35)
            chlorine = rng.normal(0.9, 0.35)
            hardness = rng.normal(90, 35)
            nitrate = rng.normal(5, 3)
        elif profile == "moderate":
            ph = rng.normal(6.9, 0.6)
            chlorine = rng.normal(1.6, 0.9)
            hardness = rng.normal(220, 60)
            nitrate = rng.normal(22, 10)
        else:  # contaminated
            ph = rng.choice([rng.normal(5.6, 0.5), rng.normal(9.3, 0.5)])
            chlorine = rng.choice([rng.normal(0.05, 0.05), rng.normal(4.5, 1.2)])
            hardness = rng.normal(400, 100)
            nitrate = rng.normal(60, 20)

        records.append((ph, chlorine, hardness, nitrate))

    df = pd.DataFrame(records, columns=["ph", "chlorine", "hardness", "nitrate"])

    # Clip to physically valid ranges
    df["ph"] = df["ph"].clip(0, 14)
    df["chlorine"] = df["chlorine"].clip(0, 10)
    df["hardness"] = df["hardness"].clip(0, 1000)
    df["nitrate"] = df["nitrate"].clip(0, 200)
    return df


def _derive_label(row, rng):
    statuses = parameter_statuses(row.to_dict())
    ranks = [STATUS_RANK[s] for s in statuses.values()]
    n_unsafe = sum(1 for r in ranks if r == 2)
    n_caution = sum(1 for r in ranks if r == 1)

    if n_unsafe >= 2 or (n_unsafe >= 1 and n_caution >= 2):
        label = "Unsafe"
    elif n_unsafe == 1 or n_caution >= 2:
        label = "Caution"
    elif n_caution == 1:
        label = "Caution" if rng.random() > 0.3 else "Safe"
    else:
        label = "Safe"

    # Small label noise (~4%) to simulate real-world measurement/labeling
    # imperfection and prevent the model from being a pure threshold lookup.
    if rng.random() < 0.04:
        label = rng.choice(["Safe", "Caution", "Unsafe"])
    return label


def generate_dataset(n_records: int = 1200, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _sample_parameters(n_records, rng)
    df["ph"] = df["ph"].round(2)
    df["chlorine"] = df["chlorine"].round(2)
    df["hardness"] = df["hardness"].round(1)
    df["nitrate"] = df["nitrate"].round(2)
    df["label"] = df.apply(lambda row: _derive_label(row, rng), axis=1)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "water_quality_dataset.csv")
    dataset = generate_dataset(1200)
    dataset.to_csv(out_path, index=False)
    print(f"Generated {len(dataset)} records -> {out_path}")
    print(dataset["label"].value_counts())
