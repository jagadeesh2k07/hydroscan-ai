"""
explainability.py
==================
The core Explainable AI (XAI) engine of HydroScan AI.

For every prediction, this module:
  1. Computes per-feature SHAP contribution values for the PREDICTED class
     using a TreeExplainer against the trained RandomForest.
  2. Converts those SHAP values into human-friendly contribution
     PERCENTAGES that sum to 100%.
  3. Identifies the single most influential parameter.
  4. Generates plain-English reason sentences per parameter (domain rules).
  5. Produces an overall plain-English summary for non-technical users.

Design note: SHAP gives us model-faithful, mathematically grounded
attribution (it explains what the RandomForest actually did), while the
domain_knowledge templates translate that into language a non-technical
person can understand. Combining both is what makes the system genuinely
explainable rather than a black box with a confidence score bolted on.
"""
import os
import sys
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.domain_knowledge import (
    FEATURES, FEATURE_LABELS, STATUS_SAFE, STATUS_CAUTION, STATUS_UNSAFE,
    parameter_statuses, build_reason, build_causes, build_recommendations,
)


class Explainer:
    """Wraps a trained RandomForest bundle with a SHAP TreeExplainer."""

    def __init__(self, model_bundle: dict):
        self.model = model_bundle["model"]
        self.features = model_bundle["features"]
        self.classes = model_bundle["classes"]
        self.global_importance = model_bundle.get("global_importance", {})
        self._shap_explainer = shap.TreeExplainer(self.model)

    def explain(self, params: dict) -> dict:
        """
        Run a full prediction + explanation for a single water sample.

        params: dict with keys ph, chlorine, hardness, nitrate (floats)
        returns: a fully-populated explanation dict ready for the API/UI.
        """
        x = pd.DataFrame([[float(params[f]) for f in self.features]], columns=self.features)

        probs = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        prediction = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # ---- SHAP contribution for the predicted class ----------------
        shap_values = self._shap_explainer.shap_values(x)
        # shap_values shape handling: sklearn RF multi-class -> list per class
        # or ndarray (n_samples, n_features, n_classes) depending on SHAP version.
        contrib = self._extract_class_contributions(shap_values, pred_idx)

        abs_contrib = {f: abs(v) for f, v in contrib.items()}
        total = sum(abs_contrib.values()) or 1e-9
        contribution_pct = {f: round(100 * abs_contrib[f] / total, 1) for f in self.features}

        # Fix rounding drift so percentages sum to exactly 100
        contribution_pct = self._normalize_to_100(contribution_pct)

        top_feature = max(contribution_pct, key=contribution_pct.get)

        # ---- Rule-based status + plain-English reasons -----------------
        statuses = parameter_statuses(params)
        reasons = []
        # Order reasons by contribution (most influential first), but only
        # surface non-safe parameters as "reasons"; safe ones are omitted
        # from the reasons list to keep the explanation focused.
        ordered_features = sorted(self.features, key=lambda f: -contribution_pct[f])
        for f in ordered_features:
            if statuses[f] != STATUS_SAFE:
                reasons.append(build_reason(f, statuses[f], float(params[f])))
        if not reasons:
            reasons.append("All tested parameters fall within safe drinking-water ranges.")

        causes = build_causes(statuses, params) if prediction != STATUS_SAFE else []
        recommendations = build_recommendations(prediction, statuses, params)

        plain_summary = self._plain_summary(prediction, confidence, top_feature, statuses)

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 1),
            "probabilities": {cls: round(float(p) * 100, 1) for cls, p in zip(self.classes, probs)},
            "parameter_statuses": statuses,
            "contribution_percent": contribution_pct,
            "top_influential_parameter": FEATURE_LABELS[top_feature],
            "top_influential_parameter_key": top_feature,
            "reasons": reasons,
            "causes": causes,
            "recommendations": recommendations,
            "plain_summary": plain_summary,
            "shap_raw": {f: round(float(contrib[f]), 4) for f in self.features},
        }

    # ------------------------------------------------------------------
    def _extract_class_contributions(self, shap_values, class_idx) -> dict:
        """Normalize SHAP's several possible output shapes into a simple
        {feature: signed_contribution} dict for one sample & one class."""
        if isinstance(shap_values, list):
            # List of (n_samples, n_features) arrays, one per class
            arr = shap_values[class_idx][0]
        else:
            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                # (n_samples, n_features, n_classes)
                arr = shap_values[0, :, class_idx]
            else:
                # (n_samples, n_features) - binary/regression fallback
                arr = shap_values[0]
        return {f: float(v) for f, v in zip(self.features, arr)}

    @staticmethod
    def _normalize_to_100(pct: dict) -> dict:
        diff = round(100 - sum(pct.values()), 1)
        if abs(diff) >= 0.1:
            # Nudge the largest contributor to absorb rounding drift
            biggest = max(pct, key=pct.get)
            pct[biggest] = round(pct[biggest] + diff, 1)
        return pct

    @staticmethod
    def _plain_summary(prediction, confidence, top_feature, statuses) -> str:
        label = FEATURE_LABELS[top_feature]
        conf_pct = round(confidence * 100)
        if prediction == STATUS_SAFE:
            return (f"This water sample is classified as SAFE with {conf_pct}% confidence. "
                     f"All key parameters, including {label.lower()}, fall within healthy limits.")
        elif prediction == STATUS_CAUTION:
            return (f"This water sample requires CAUTION ({conf_pct}% confidence). "
                     f"{label} is the biggest factor pulling this result away from 'Safe' — "
                     f"it's outside the ideal range but not yet at a dangerous level.")
        else:
            return (f"This water sample is classified as UNSAFE ({conf_pct}% confidence). "
                     f"{label} is the dominant factor driving this result and should be "
                     f"addressed before the water is used for drinking.")
