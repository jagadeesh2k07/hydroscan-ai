"""
train_confidence.py - fit the strip-validity (series-system) model.

    python -m vision.train_confidence

Maximises the Bernoulli likelihood of
        P(valid) = prod_j sigmoid(a_j + b_j * z_j)
over photos that clear the deterministic gates (seed 1000), with the SIGN of
each b_j fixed by physics (see FEATURE_DIRECTION). Writes
vision/confidence_model.json, then validates on a DIFFERENT, freshly
generated held-out corpus (seed 2000) and stores a summary of those metrics
inside the same JSON.
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_expit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import strip_analyzer as sa
from vision.validate import build_rows, evaluate, Incomplete

N_TRAIN, N_VAL = 900, 500
BUDGET = 200   # seconds per invocation; re-run to resume


def main():
    t0 = time.time()
    train = build_rows(N_TRAIN, N_TRAIN, seed=1000, cache_prefix="/tmp/strip_train", budget=BUDGET)
    ok = [r for r in train if r["gate"] is None]
    X = np.array([[r["features"][n] for n in sa.FEATURE_NAMES] for r in ok])
    y = np.array([r["label"] for r in ok])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    dirs = np.array([sa.FEATURE_DIRECTION[n] for n in sa.FEATURE_NAMES], float)
    k = Z.shape[1]

    def unpack(th):
        return th[:k], dirs * np.abs(th[k:])            # sign of b_j fixed by physics

    def nll_fn(Zs, ys, lam):
        def nll(th):
            a_, b_ = unpack(th)
            logp = log_expit(a_ + Zs * b_).sum(axis=1)       # ln P = sum_j ln f_j
            p = np.exp(logp)
            ll = ys * logp + (1 - ys) * np.log(np.clip(1 - p, 1e-12, 1.0))
            return -ll.mean() + lam * np.sum(np.abs(th[k:]))
        return nll

    th0 = np.concatenate([np.full(k, 2.0), np.full(k, 1.0)])
    bnds = [(-8, 12)] * k + [(0, 25)] * k

    def fit(Zs, ys, lam):
        """Multi-start (the series-sigmoid likelihood is non-convex): keep the best optimum."""
        best = None
        for slope in (1.0, 4.0, 8.0):
            start = np.concatenate([np.full(k, 3.0), np.full(k, slope)])
            r = minimize(nll_fn(Zs, ys, lam), start, method="L-BFGS-B", bounds=bnds, options={"maxiter": 3000})
            if best is None or r.fun < best.fun:
                best = r
        return best

    # choose the L1 strength by 5-fold CV log-loss on the TRAINING corpus only
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(y)) % 5
    cv = {}
    for lam in (1e-2, 1e-3, 1e-4, 1e-5):
        losses = []
        for f in range(5):
            tr, te = folds != f, folds == f
            th = fit(Z[tr], y[tr], lam).x
            losses.append(nll_fn(Z[te], y[te], 0.0)(th))
        cv[lam] = float(np.mean(losses))
    best_lam = min(cv, key=cv.get)
    res = fit(Z, y, best_lam)
    a_hat, b_hat = unpack(res.x)
    model = {
        "type": "series_sigmoid",
        "features": sa.FEATURE_NAMES, "mean": mu.tolist(), "std": sd.tolist(),
        "a": a_hat.tolist(), "b": b_hat.tolist(),
        "fit": {"n_images_generated": len(train), "n_passed_gates": len(ok),
                "n_valid": int(y.sum()), "n_invalid": int((1 - y).sum()),
                "seed": 1000, "l1_penalty": best_lam, "cv_nll_by_penalty": {str(k_): round(v, 4) for k_, v in cv.items()},
                "converged": bool(res.success)},
    }
    # seed 2000 = DEVELOPMENT set (looked at while designing features);
    # seed 3000 = FINAL TEST set (never used for any decision)
    dev = build_rows(N_VAL, N_VAL, seed=2000, cache_prefix="/tmp/strip_val_2000", budget=BUDGET)
    test = build_rows(N_VAL, N_VAL, seed=3000, cache_prefix="/tmp/strip_val_3000", budget=BUDGET)

    def summary(rep, name):
        return {"corpus": name,
                "true_accept_rate": rep["true_accept_rate"], "false_accept_rate": rep["false_accept_rate"],
                "precision": rep["precision"], "overall_accuracy": rep["overall_accuracy"],
                "roc_auc": rep.get("model_only", {}).get("roc_auc"), "ece": rep.get("model_only", {}).get("ece")}

    model["dev_summary"] = summary(evaluate(dev, model), f"{N_VAL}+{N_VAL} development photos (seed 2000)")
    model["validation_summary"] = summary(evaluate(test, model),
                                          f"{N_VAL} valid + {N_VAL} invalid FINAL-TEST synthetic photos (seed 3000, never used for any design decision)")
    json.dump(model, open(sa.MODEL_PATH, "w"), indent=2)
    print(f"trained in {time.time() - t0:.0f}s")
    print("a:", {n: round(v, 2) for n, v in zip(sa.FEATURE_NAMES, model["a"])})
    print("b:", {n: round(v, 2) for n, v in zip(sa.FEATURE_NAMES, model["b"])})
    print(json.dumps({"dev": model["dev_summary"], "final_test": model["validation_summary"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Incomplete as e:
        print("PROGRESS SAVED (" + str(e) + ") - run the same command again to continue.")
