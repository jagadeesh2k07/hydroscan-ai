"""
calibration.py
===============
The mathematics behind HydroScan's *model confidence* number.

DEFINITION
----------
For a water sample x, the RandomForest (T trees) produces class probabilities

        p_k(x) = (1/T) * sum_t  p_{t,k}(x)          (mean of per-tree leaf class frequencies)

and the reported confidence is the probability of the predicted class after
temperature scaling:

        q_k(x) = softmax_k( ln p_k(x) / tau )        tau > 0
        confidence(x) = max_k q_k(x)

tau is a single scalar fitted by minimising the negative log-likelihood (NLL)
of *out-of-fold* predictions on the training data (so the number is never
tuned on the data it is later evaluated on):

        tau* = argmin_tau  - (1/N) sum_i  ln q_{y_i}(x_i)

tau = 1 means the forest's probabilities were already calibrated; tau > 1
softens over-confident probabilities, tau < 1 sharpens under-confident ones.
Temperature scaling never changes the predicted class (argmax is preserved);
it only rescales how sure the model claims to be.

VALIDATION METRICS (all computed on a held-out test split)
----------------------------------------------------------
* Accuracy, macro-F1
* NLL / log-loss          : -(1/N) sum ln q_{y_i}
* Brier score (multiclass): (1/N) sum_i sum_k (q_k(x_i) - 1[y_i = k])^2
* ECE (10 equal-width bins): sum_b (n_b/N) * | acc(b) - conf(b) |
  i.e. "when the model says 80 %, is it right ~80 % of the time?"
"""
import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-9


def temperature_scale(probs: np.ndarray, tau: float) -> np.ndarray:
    """q = softmax(ln(p)/tau), row-wise. probs: (n, K)."""
    logp = np.log(np.clip(probs, EPS, 1.0)) / tau
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def fit_temperature(oof_probs: np.ndarray, y_idx: np.ndarray) -> float:
    """tau* = argmin NLL on out-of-fold probabilities."""
    res = minimize_scalar(lambda t: nll(temperature_scale(oof_probs, t), y_idx),
                          bounds=(0.25, 4.0), method="bounded")
    return float(res.x)


def brier_multiclass(probs: np.ndarray, y_idx: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 10):
    """Return (ece, table) where table rows are
    {bin, n, mean_confidence, accuracy} for non-empty bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_idx)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        mc, acc = float(conf[m].mean()), float(correct[m].mean())
        ece += float(m.mean()) * abs(acc - mc)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                      "mean_confidence": round(mc, 4), "accuracy": round(acc, 4)})
    return float(ece), table
