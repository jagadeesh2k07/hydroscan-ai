"""
model_manager.py
==================
Holds the single, currently-active trained model bundle + Explainer in
memory, so Flask request handlers always use the latest model without
re-loading from disk on every request. The Admin Panel's "Retrain Model"
action calls reload() after training completes, hot-swapping the active
model for all subsequent predictions — no server restart required.
"""
import threading
import joblib

from ml.explainability import Explainer

_lock = threading.Lock()
_state = {"bundle": None, "explainer": None}


def load(model_path: str):
    with _lock:
        bundle = joblib.load(model_path)
        _state["bundle"] = bundle
        _state["explainer"] = Explainer(bundle)
    return _state["explainer"]


def get_explainer() -> Explainer:
    if _state["explainer"] is None:
        raise RuntimeError("Model not loaded yet. Call model_manager.load() at startup.")
    return _state["explainer"]


def get_bundle() -> dict:
    if _state["bundle"] is None:
        raise RuntimeError("Model not loaded yet. Call model_manager.load() at startup.")
    return _state["bundle"]
