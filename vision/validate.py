"""
validate.py - measure & validate the strip-confidence model.

    python -m vision.validate                  # fresh synthetic held-out set
    python -m vision.validate --real DIR       # DIR/valid/*.jpg + DIR/invalid/*.jpg (real photos)

Every image goes through the SAME gates + feature extraction + scoring code
the web app uses (vision.strip_analyzer), so validation cannot drift from
production behaviour.
"""
import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import strip_analyzer as sa
from vision.synthetic import generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def measure_rows(images):
    """images: iterable of (img, label, kind) -> list of row dicts."""
    rows = []
    for img, label, kind in images:
        row = {"label": int(label), "kind": kind, "gate": None, "features": None}
        try:
            row["features"] = sa.measure(sa.prepare_array(img))["features"]
        except sa.GateReject as e:
            row["gate"] = e.code
        rows.append(row)
    return rows


class Incomplete(RuntimeError):
    """Corpus build hit its time budget; progress is cached - just re-run."""


def build_rows(n_valid, n_invalid, seed, cache_prefix=None, chunk=150, budget=None):
    """Build the corpus in cached chunks so long runs are resumable."""
    import time, math
    t0, rows = time.time(), []
    for c in range(math.ceil(max(n_valid, n_invalid) / chunk)):
        nv = max(0, min(chunk, n_valid - c * chunk))
        ni = max(0, min(chunk, n_invalid - c * chunk))
        path = f"{cache_prefix}_{c}.pkl" if cache_prefix else None
        if path and os.path.exists(path):
            rows += pickle.load(open(path, "rb")); continue
        if budget and time.time() - t0 > budget:
            raise Incomplete(f"chunk {c} not built yet")
        part = measure_rows(generate(nv, ni, seed * 1000 + c, type_offset=c))
        if path:
            pickle.dump(part, open(path, "wb"))
        rows += part
    return rows


def real_rows(folder):
    def load(sub, label):
        out = []
        for f in sorted(glob.glob(os.path.join(folder, sub, "*"))):
            img = cv2.imread(f)
            if img is not None:
                out.append((img, label, "real"))
        return out
    return measure_rows(load("valid", 1) + load("invalid", 0))


def _p_of(row, model):
    return sa.p_from_features(model, row["features"])[0]


def evaluate(rows, model, threshold=sa.DECISION_THRESHOLD):
    y = np.array([r["label"] for r in rows])
    gate = [r["gate"] for r in rows]
    passed = np.array([g is None for g in gate])
    p = np.array([_p_of(r, model) if r["features"] else 0.0 for r in rows])
    accepted = passed & (p >= threshold)

    tp = int((accepted & (y == 1)).sum()); fp = int((accepted & (y == 0)).sum())
    fn = int((~accepted & (y == 1)).sum()); tn = int((~accepted & (y == 0)).sum())
    n_valid, n_invalid = int((y == 1).sum()), int((y == 0).sum())

    out = {
        "n_valid": n_valid, "n_invalid": n_invalid, "threshold": threshold,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "true_accept_rate": round(tp / max(n_valid, 1), 4),
        "false_accept_rate": round(fp / max(n_invalid, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "overall_accuracy": round((tp + tn) / len(rows), 4),
    }

    # Model-only quality on photos that cleared the deterministic gates
    if passed.sum() > 10 and len(set(y[passed])) == 2:
        yp, pp = y[passed], p[passed]
        out["model_only"] = {
            "n": int(passed.sum()),
            "roc_auc": round(float(roc_auc_score(yp, pp)), 4),
            "brier": round(float(np.mean((pp - yp) ** 2)), 4),
        }
        edges = np.linspace(0, 1, 11); ece = 0.0; table = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pp >= lo) & (pp < hi if hi < 1 else pp <= hi)
            if m.any():
                ece += m.mean() * abs(pp[m].mean() - yp[m].mean())
                table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                              "mean_predicted_P": round(float(pp[m].mean()), 3),
                              "observed_valid_fraction": round(float(yp[m].mean()), 3)})
        out["model_only"]["ece"] = round(float(ece), 4)
        out["model_only"]["reliability_table"] = table

    per_kind = {}
    for r, a in zip(rows, accepted):
        k = per_kind.setdefault(r["kind"], {"n": 0, "accepted": 0, "gate_rejected": 0})
        k["n"] += 1; k["accepted"] += int(a); k["gate_rejected"] += int(r["gate"] is not None)
    out["per_kind"] = {k: {**v, "accept_rate": round(v["accepted"] / v["n"], 4)} for k, v in sorted(per_kind.items())}
    gates = {}
    for g in gate:
        if g: gates[g] = gates.get(g, 0) + 1
    out["gate_rejections"] = gates
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="folder with valid/ and invalid/ subfolders of real photos")
    ap.add_argument("--n", type=int, default=500, help="synthetic images per class")
    ap.add_argument("--seed", type=int, default=3000, help="3000 = final test set")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "strip_validation_report.json"))
    a = ap.parse_args()
    model = sa.load_confidence_model()
    rows = real_rows(a.real) if a.real else build_rows(a.n, a.n, a.seed, cache_prefix=f'/tmp/strip_val_{a.seed}')
    rep = evaluate(rows, model)
    rep["source"] = f"real photos in {a.real}" if a.real else f"synthetic held-out corpus (seed={a.seed})"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_kind",)}, indent=2))
    print("\nper-kind accept rate (valid should be ~1, all others ~0):")
    for k, v in rep["per_kind"].items():
        print(f"  {k:<20} n={v['n']:<4} accepted={v['accept_rate']:.3f}")


if __name__ == "__main__":
    main()
