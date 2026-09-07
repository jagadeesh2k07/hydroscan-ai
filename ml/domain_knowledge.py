"""
domain_knowledge.py
====================
Single source of truth for all water-quality domain rules used across the
HydroScan AI system: safe/caution/unsafe thresholds, plain-English reason
templates, probable causes, and recommendations.

Thresholds are simplified, portfolio-grade approximations inspired by
WHO / EPA drinking-water guidance. They are NOT a substitute for certified
laboratory testing or medical/regulatory advice.

Units:
    pH        -> unitless (0-14 scale)
    chlorine  -> mg/L (free residual chlorine)
    hardness  -> ppm as CaCO3
    nitrate   -> mg/L as NO3-
"""

FEATURES = ["ph", "chlorine", "hardness", "nitrate"]

FEATURE_LABELS = {
    "ph": "pH",
    "chlorine": "Chlorine",
    "hardness": "Hardness",
    "nitrate": "Nitrate",
}

FEATURE_UNITS = {
    "ph": "",
    "chlorine": "mg/L",
    "hardness": "ppm",
    "nitrate": "mg/L",
}

# Valid input ranges for form / sensor validation
VALID_RANGES = {
    "ph": (0.0, 14.0),
    "chlorine": (0.0, 10.0),
    "hardness": (0.0, 1000.0),
    "nitrate": (0.0, 200.0),
}

# Ideal display range used for gauges / cards
IDEAL_RANGE = {
    "ph": (6.5, 8.5),
    "chlorine": (0.2, 2.0),
    "hardness": (0.0, 150.0),
    "nitrate": (0.0, 10.0),
}

STATUS_SAFE, STATUS_CAUTION, STATUS_UNSAFE = "Safe", "Caution", "Unsafe"
STATUS_RANK = {STATUS_SAFE: 0, STATUS_CAUTION: 1, STATUS_UNSAFE: 2}


def status_ph(v: float) -> str:
    if 6.5 <= v <= 8.5:
        return STATUS_SAFE
    if 6.0 <= v < 6.5 or 8.5 < v <= 9.0:
        return STATUS_CAUTION
    return STATUS_UNSAFE


def status_chlorine(v: float) -> str:
    if 0.2 <= v <= 2.0:
        return STATUS_SAFE
    if (0.0 <= v < 0.2) or (2.0 < v <= 4.0):
        return STATUS_CAUTION
    return STATUS_UNSAFE


def status_hardness(v: float) -> str:
    if v <= 150:
        return STATUS_SAFE
    if v <= 300:
        return STATUS_CAUTION
    return STATUS_UNSAFE


def status_nitrate(v: float) -> str:
    if v <= 10:
        return STATUS_SAFE
    if v <= 45:
        return STATUS_CAUTION
    return STATUS_UNSAFE


STATUS_FUNCS = {
    "ph": status_ph,
    "chlorine": status_chlorine,
    "hardness": status_hardness,
    "nitrate": status_nitrate,
}


def parameter_statuses(params: dict) -> dict:
    """Return {feature: 'Safe'|'Caution'|'Unsafe'} for a params dict."""
    return {f: STATUS_FUNCS[f](float(params[f])) for f in FEATURES}


# ---------------------------------------------------------------------------
# Plain-English reason templates, keyed by (feature, status)
# ---------------------------------------------------------------------------
REASON_TEMPLATES = {
    ("ph", "Safe"):    "pH is within the ideal neutral range for safe drinking water.",
    ("ph", "Caution"): "pH is slightly outside the ideal range, trending {direction}.",
    ("ph", "Unsafe"):  "pH is significantly {direction}, which can cause pipe corrosion or unpleasant taste and irritation.",

    ("chlorine", "Safe"):    "Chlorine level is sufficient to disinfect the water without excess.",
    ("chlorine", "Caution"): "Chlorine level is {direction} the recommended disinfection range.",
    ("chlorine", "Unsafe"):  "Chlorine is {direction}: {detail}",

    ("hardness", "Safe"):    "Hardness is within an acceptable range for household use.",
    ("hardness", "Caution"): "Hardness is above the recommended range, which may cause mild scaling.",
    ("hardness", "Unsafe"):  "Hardness is well above the recommended range, indicating heavy mineral content.",

    ("nitrate", "Safe"):    "Nitrate level is within the safe limit.",
    ("nitrate", "Caution"): "Nitrate level is elevated and approaching the safety limit.",
    ("nitrate", "Unsafe"):  "Nitrate level exceeds the safe limit, posing a health risk, especially for infants.",
}


def build_reason(feature: str, status: str, value: float) -> str:
    """Build a plain-English reason sentence for one parameter."""
    if status == STATUS_SAFE:
        return REASON_TEMPLATES[(feature, "Safe")]

    if feature == "ph":
        direction = "acidic" if value < 6.5 else "alkaline"
        return REASON_TEMPLATES[(feature, status)].format(direction=direction)

    if feature == "chlorine":
        if status == "Caution":
            direction = "below" if value < 0.2 else "above"
            return REASON_TEMPLATES[(feature, status)].format(direction=direction)
        else:  # Unsafe
            if value <= 0.0:
                direction, detail = "absent", "no residual disinfection protection against pathogens."
            else:
                direction, detail = "far in excess", "this may cause irritation and an unpleasant taste/odor."
            return REASON_TEMPLATES[(feature, status)].format(direction=direction, detail=detail)

    if feature in ("hardness", "nitrate"):
        return REASON_TEMPLATES[(feature, status)]

    return f"{FEATURE_LABELS[feature]} reading is {status.lower()}."


# ---------------------------------------------------------------------------
# Probable causes, keyed by feature+status
# ---------------------------------------------------------------------------
CAUSES = {
    ("ph", "Unsafe"): [
        "Natural mineral deposits or bedrock geology altering water chemistry",
        "Industrial or chemical runoff into the water source",
    ],
    ("ph", "Caution"): [
        "Natural variation in source water mineral content",
    ],
    ("chlorine", "Unsafe_low"): [
        "Poor or inconsistent water treatment at the source",
        "Old or leaking pipelines allowing chlorine to dissipate",
    ],
    ("chlorine", "Unsafe_high"): [
        "Over-chlorination during treatment",
    ],
    ("chlorine", "Caution"): [
        "Inconsistent dosing during water treatment",
    ],
    ("hardness", "Unsafe"): [
        "Natural mineral deposits (limestone/chalk) in the water source",
        "Groundwater passing through calcium/magnesium-rich rock",
    ],
    ("hardness", "Caution"): [
        "Moderate natural mineral content in groundwater",
    ],
    ("nitrate", "Unsafe"): [
        "Agricultural fertilizer runoff into the water supply",
        "Leaching from septic systems or animal waste near the source",
        "Industrial pollution entering the groundwater",
    ],
    ("nitrate", "Caution"): [
        "Nearby agricultural activity contributing to elevated nitrate",
    ],
}


def build_causes(statuses: dict, params: dict) -> list:
    """Aggregate probable causes for all non-safe parameters."""
    causes = []
    for feature, status in statuses.items():
        if status == STATUS_SAFE:
            continue
        if feature == "chlorine" and status == "Unsafe":
            key = ("chlorine", "Unsafe_low" if params["chlorine"] <= 0.2 else "Unsafe_high")
        else:
            key = (feature, status)
        for c in CAUSES.get(key, []):
            if c not in causes:
                causes.append(c)
    if not causes:
        causes.append("No significant contamination indicators detected — water chemistry appears naturally balanced.")
    return causes


# ---------------------------------------------------------------------------
# Recommendations, keyed by overall prediction + specific flagged params
# ---------------------------------------------------------------------------
def build_recommendations(prediction: str, statuses: dict, params: dict) -> list:
    recs = []

    if prediction == STATUS_SAFE:
        recs.append("Water quality is within safe limits — continue routine periodic testing (every 3-6 months).")
        recs.append("Store water in a clean, covered container to prevent recontamination.")
        return recs

    if prediction == STATUS_CAUTION:
        recs.append("Use a home water filter (activated carbon or RO) as a precaution.")
        recs.append("Retest water after 1-2 weeks to monitor the trend.")

    if prediction == STATUS_UNSAFE:
        recs.append("Use a Reverse Osmosis (RO) purifier or certified filtration system before drinking.")
        recs.append("Boil water for at least 1 minute before drinking if no purifier is available.")

    if statuses.get("nitrate") in ("Caution", "Unsafe"):
        recs.append("Avoid using this water source for infant formula until nitrate levels are verified safe.")
    if statuses.get("chlorine") == "Unsafe" and params.get("chlorine", 0) <= 0.2:
        recs.append("Boil water thoroughly since disinfection appears insufficient to neutralize pathogens.")
    if statuses.get("hardness") in ("Caution", "Unsafe"):
        recs.append("Consider a water softener to reduce scaling on appliances and plumbing.")
    if statuses.get("ph") == "Unsafe":
        recs.append("Consider a pH-correction/neutralizing filter to bring pH into the safe range.")

    recs.append("Retest after treatment to confirm parameters have returned to safe levels.")
    recs.append("If contamination persists across repeated tests, contact your local water authority.")

    # De-duplicate while preserving order
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)
    return unique_recs
