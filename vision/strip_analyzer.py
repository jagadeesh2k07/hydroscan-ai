"""
strip_analyzer.py  (v3 - statistically grounded confidence)
============================================================
Reads a photo of a 4-pad test strip with OpenCV and returns estimated
pH / chlorine / hardness / nitrate values plus a *validated* confidence.

PIPELINE
--------
 1. Hard gates (deterministic, with a clear message): undecodable file,
    resolution < 150 px, no strip-shaped region found. (Focus, noise, exposure
    etc. are NOT hand-thresholded; the fitted model weighs them.)
 2. Locate the strip (Otsu x2 polarities, adaptive threshold, Canny; best
    candidate wins), straighten it, sample 4 equal pad windows.
 3. White-balance the pad colours using the strip's own backing as the
    neutral reference.
 4. Measure six image features x_i (below).
 5. CONFIDENCE = P(photo is a valid, usable strip | features) from a
    logistic model whose weights were FITTED ON DATA (not hand-picked):

        z_i = (x_i - mu_i) / s_i
        logit = b + sum_i w_i * z_i
        P(valid) = 1 / (1 + exp(-logit))

    mu, s, w, b live in vision/confidence_model.json, together with the
    held-out validation metrics. Accept if P >= c_FA / (c_FA + c_FR)
    (decision-theory threshold; a false accept is costed 1.5x a false
    reject, giving 0.6).
 6. Map each white-balanced pad colour to a value by inverse-distance
    interpolation on the Lab calibration chart, and report the CIE76 colour
    distance (delta-E) to the nearest chart colour so out-of-range pads are
    flagged instead of silently extrapolated.

FEATURES (all measured physical quantities, no hand-set weights)
----------------------------------------------------------------
 ln_aspect               ln(long side / short side) of the detected strip
 coverage                strip area / frame area
 ln_edge_width           ln(edge-spread width, px) = ln( grey-level step / peak
                         gradient ) across the strip boundary  (focus; a blur of
                         sigma px widens it ~2.5*sigma)
 ln_noise                ln(1 + sigma_n), sigma_n by Immerkaer's estimator
                         sigma = sqrt(pi/2) * sum|I * N| / (6 (W-2)(H-2)),
                         N = [[1,-2,1],[-2,4,-2],[1,-2,1]]
 exposure                mean brightness of the strip backing / 255
 pad_texture             sqrt(max(pad_sigma^2 - noise_sigma^2, 0)); pad_sigma = mean per-pad
                         colour std-dev on the RAW image. Subtracting the noise variance
                         leaves genuine texture (uniform pads -> ~0)
 ln_pad_distinctiveness  ln(1 + mean pairwise Lab distance of the 4 pads)
 ln_gap_contrast         ln(1 + Lab distance between pad colour and the backing
                         visible in the gaps between pads)
 edge_contrast           |mean gray inside strip - mean gray of surrounding ring| / 255

HONEST SCOPE: the confidence model is fitted and validated on a large
SYNTHETIC corpus (vision/synthetic.py) because no labelled real-photo set
exists for this project. `python -m vision.validate --real DIR` re-runs the
same metrics on real labelled photos (DIR/valid, DIR/invalid). The colour
chart is a generic approximation, not a certified calibration of any brand.
"""
import os
import json
import base64
import math
import cv2
import numpy as np

CALIBRATION = {
    "ph": [
        ((60, 45, 40), 5.0), ((65, 30, 30), 6.0), ((70, 5, 15), 6.5),
        ((75, -10, 5), 7.0), ((70, -20, -5), 7.5), ((60, -25, -20), 8.0),
        ((50, -15, -35), 8.5), ((45, 0, -40), 9.0),
    ],
    "chlorine": [
        ((95, -2, 5), 0.0), ((90, 5, 20), 0.5), ((85, 15, 35), 1.0),
        ((78, 30, 40), 2.0), ((65, 45, 35), 3.0), ((50, 55, 25), 4.0),
        ((40, 60, 15), 5.0),
    ],
    "hardness": [
        ((92, -5, 10), 0), ((85, 0, 25), 60), ((75, 10, 40), 120),
        ((65, 20, 45), 180), ((55, 35, 40), 250), ((45, 45, 30), 350),
        ((35, 50, 20), 450),
    ],
    "nitrate": [
        ((90, -10, 0), 0), ((80, 5, 15), 5), ((70, 25, 20), 10),
        ((60, 45, 25), 20), ((50, 65, 30), 45), ((40, 75, 35), 70),
        ((30, 80, 40), 100),
    ],
}
PAD_ORDER = ["ph", "chlorine", "hardness", "nitrate"]

FEATURE_NAMES = ["ln_aspect", "coverage", "ln_edge_width", "ln_noise", "exposure",
                 "pad_texture", "ln_pad_distinctiveness", "ln_gap_contrast", "edge_contrast"]
FEATURE_LABELS = {
    "ln_aspect": "strip shape (length : width)",
    "coverage": "strip size in frame",
    "ln_edge_width": "focus (edge sharpness)",
    "ln_noise": "image noise",
    "exposure": "exposure (brightness)",
    "pad_texture": "pad colour uniformity",
    "ln_pad_distinctiveness": "difference between the 4 pads",
    "ln_gap_contrast": "pads separated by backing gaps",
    "edge_contrast": "strip vs background contrast",
}
REJECT_HINT = {
    "ln_aspect": "the detected shape isn't a long, thin test strip",
    "coverage": "the strip is too small or too large in the frame",
    "ln_edge_width": "the photo is out of focus or too soft",
    "ln_noise": "the photo is too grainy/noisy",
    "exposure": "the photo is too dark or over-exposed",
    "pad_texture": "the pad areas are patchy rather than solid colour blocks",
    "ln_pad_distinctiveness": "the four pads are too similar in colour to be a real reagent strip",
    "ln_gap_contrast": "the pads aren't separated by strip backing the way a real test strip's are",
    "edge_contrast": "the strip barely stands out from the background",
}

MIN_DIMENSION_PX = 150
COST_FALSE_ACCEPT = 1.5       # relative cost of accepting a bad photo
COST_FALSE_REJECT = 1.0       # relative cost of rejecting a good photo
DECISION_THRESHOLD = COST_FALSE_ACCEPT / (COST_FALSE_ACCEPT + COST_FALSE_REJECT)  # 0.6
HIGH_CONFIDENCE = 0.85
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confidence_model.json")
_MODEL_CACHE = {}


class GateReject(ValueError):
    """A deterministic gate failed; .code identifies which one."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------
def _bgr_to_lab(bgr_pixel):
    patch = np.uint8([[np.clip(bgr_pixel, 0, 255)]])
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0][0]
    return (float(lab[0]) * 100.0 / 255.0, float(lab[1]) - 128.0, float(lab[2]) - 128.0)


def _delta_e(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))     # CIE76


def _match_calibration(lab_color, table):
    d = sorted(((_delta_e(lab_color, anchor), v) for anchor, v in table), key=lambda t: t[0])
    (d1, v1), (d2, v2) = d[0], d[1]
    if d1 + d2 == 0:
        return v1, 0.0
    w1, w2 = d2 / (d1 + d2), d1 / (d1 + d2)
    return v1 * w1 + v2 * w2, d1


# --------------------------------------------------------------------------
# strip location
# --------------------------------------------------------------------------
def _sharpness(img) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _candidates_from_mask(mask, h, w):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < h * w * 0.015 or area > h * w * 0.90:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect < 2.0:
            continue
        # ONLY used to rank competing candidates (which contour is "the strip").
        # It is NOT the reported confidence.
        rank = 0.6 * min((aspect - 2.0) / 4.0, 1.0) + 0.4 * (1 - min(abs(area / (h * w) - 0.25) / 0.25, 1.0))
        out.append((rank, rect, aspect, area / (h * w)))
    return out


def _locate_strip(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    cands = []
    for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, m = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
        cands += _candidates_from_mask(m, h, w)
    ad = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5)
    cands += _candidates_from_mask(ad, h, w)
    ed = cv2.dilate(cv2.Canny(blurred, 40, 120), np.ones((5, 5), np.uint8), iterations=2)
    cands += _candidates_from_mask(ed, h, w)
    if not cands:
        return None
    cands.sort(key=lambda t: -t[0])
    _, rect, aspect, area_frac = cands[0]
    return _refine_rect(gray, rect, aspect, area_frac)


def _refine_rect(gray, rect, aspect, area_frac):
    """Tighten a (possibly inflated) candidate box to the strip's real boundary:
    threshold at the midpoint between the strip interior and its surrounding
    ring, keep the largest blob inside the box, and re-fit a rotated rectangle."""
    h, w = gray.shape
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.intp(cv2.boxPoints(rect)), 255)
    ring = cv2.dilate(mask, np.ones((41, 41), np.uint8)) & ~mask
    if not mask.any() or not ring.any():
        return rect, aspect, area_frac
    inner, outer = float(gray[mask > 0].mean()), float(gray[ring > 0].mean())
    thr = (inner + outer) / 2
    binary = ((gray > thr) if inner > outer else (gray < thr)).astype(np.uint8) * 255
    binary &= cv2.dilate(mask, np.ones((15, 15), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return rect, aspect, area_frac
    c = max(cs, key=cv2.contourArea)
    r2 = cv2.minAreaRect(c)
    rw, rh = r2[1]
    if min(rw, rh) < 4:
        return rect, aspect, area_frac
    a2 = max(rw, rh) / min(rw, rh)
    f2 = cv2.contourArea(c) / (h * w)
    if a2 < 2.0 or f2 < 0.4 * area_frac:
        return rect, aspect, area_frac
    return r2, a2, f2


def _edge_metrics(gray_den, rect):
    """Return (edge_contrast, edge_spread_width_px) for the strip boundary."""
    mask = np.zeros(gray_den.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.intp(cv2.boxPoints(rect)), 255)
    ring = cv2.dilate(mask, np.ones((41, 41), np.uint8)) & ~mask
    if mask.sum() == 0 or ring.sum() == 0:
        return 0.0, 60.0
    step = abs(float(gray_den[mask > 0].mean()) - float(gray_den[ring > 0].mean()))
    g = gray_den.astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    band = cv2.dilate(mask, np.ones((7, 7), np.uint8)) & ~cv2.erode(mask, np.ones((7, 7), np.uint8))
    peak = float(np.percentile(grad[band > 0], 90)) if band.any() else 0.0
    width = float(np.clip(step / max(peak, 1e-3), 0.5, 60.0))
    return step / 255.0, width


def _noise_sigma(gray):
    """Immerkaer (1996) fast noise-sigma estimate."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray.shape
    r = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * np.abs(r).sum() / (6.0 * (w - 2) * (h - 2)))


# --------------------------------------------------------------------------
# pad sampling
# --------------------------------------------------------------------------
def _straighten(img, rect):
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
        rw, rh = rh, rw
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    x, y = max(int(cx - rw / 2), 0), max(int(cy - rh / 2), 0)
    strip = rotated[y:min(y + int(rh), img.shape[0]), x:min(x + int(rw), img.shape[1])]
    return None if (strip.size == 0 or strip.shape[0] < 6 or strip.shape[1] < 20) else strip


def _extract_pads(den, raw, rect):
    """Colours/crops from the denoised image (stable colour), uniformity from RAW."""
    sd, sr = _straighten(den, rect), _straighten(raw, rect)
    if sd is None or sr is None or sd.shape != sr.shape:
        return None
    sh, sw = sd.shape[:2]

    # white balance: the strip backing (thin top/bottom bands) is the neutral reference
    band = int(sh * 0.12)
    ref = (np.concatenate([sd[:band].reshape(-1, 3), sd[-band:].reshape(-1, 3)])
           if band >= 2 else sd.reshape(-1, 3)).astype(np.float32)
    ref_mean = ref.mean(axis=0)
    gain = ref_mean.mean() / np.clip(ref_mean, 20, 255)
    exposure = float(ref_mean.mean()) / 255.0

    pad_w, my = sw / 4, int(sh * 0.22)
    colors, crops, stds = [], [], []
    for i in range(4):
        x1, x2 = max(int(i * pad_w + pad_w * 0.30), 0), min(int(i * pad_w + pad_w * 0.70), sw)
        cd, cr = sd[my:sh - my, x1:x2], sr[my:sh - my, x1:x2]
        if cd.size == 0:
            return None
        colors.append(tuple(float(v) for v in np.clip(cd.reshape(-1, 3).astype(np.float32).mean(axis=0) * gain, 0, 255)))
        crops.append(cd)
        stds.append(float(cr.reshape(-1, 3).astype(np.float32).std(axis=0).mean()))

    # colour in the gaps between pads (at the 3 internal quarter boundaries)
    gaps = []
    for i in range(1, 4):
        g1, g2 = int(i * pad_w - pad_w * 0.05), int(i * pad_w + pad_w * 0.05)
        gc = sd[my:sh - my, max(g1, 0):max(g2, g1 + 1)]
        gaps.append(tuple(float(v) for v in np.clip(gc.reshape(-1, 3).astype(np.float32).mean(axis=0) * gain, 0, 255)))
    return colors, crops, stds, gaps, exposure


def _crop_b64(crop):
    ok, buf = cv2.imencode(".png", crop)
    return ("data:image/png;base64," + base64.b64encode(buf).decode()) if ok else None


# --------------------------------------------------------------------------
# measurement  (shared by the app, the training script and the validator)
# --------------------------------------------------------------------------
def prepare_image(image_bytes: bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise GateReject("decode", "Could not decode image. Please upload a valid JPG/PNG photo.")
    return prepare_array(img)


def prepare_array(img):
    h0, w0 = img.shape[:2]
    if min(h0, w0) < MIN_DIMENSION_PX:
        raise GateReject("resolution", "Image resolution is too low. Please upload a larger, clearer photo.")
    if max(h0, w0) > 900:
        s = 900 / max(h0, w0)
        img = cv2.resize(img, (int(w0 * s), int(h0 * s)))
    return img


def measure(img):
    """Run the deterministic gates + feature extraction. Returns a dict, or raises GateReject."""
    den = cv2.bilateralFilter(img, 7, 50, 50)
    found = _locate_strip(den)
    if found is None:
        raise GateReject("no_strip", "Could not detect a test strip in this photo. Please retake it with the "
                                     "full strip visible, laid flat against a plain, contrasting background, "
                                     "filling most of the frame.")
    rect, aspect, area_frac = found
    pads = _extract_pads(den, img, rect)
    if pads is None:
        raise GateReject("pads", "Could not isolate test pads from the strip. Try a clearer, well-lit photo.")
    colors, crops, stds, gaps, exposure = pads
    labs = [_bgr_to_lab(c) for c in colors]
    dists = [_delta_e(labs[i], labs[j]) for i in range(4) for j in range(i + 1, 4)]
    gap_d = [_delta_e(_bgr_to_lab(gaps[i - 1]),
                      tuple((a + b) / 2 for a, b in zip(labs[i - 1], labs[i]))) for i in range(1, 4)]
    contrast, width = _edge_metrics(cv2.cvtColor(den, cv2.COLOR_BGR2GRAY), rect)
    noise_sigma = _noise_sigma(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    pad_sigma = float(np.mean(stds))
    feats = {
        "ln_aspect": math.log(aspect),
        "coverage": float(area_frac),
        "ln_edge_width": math.log(width),
        "ln_noise": math.log1p(noise_sigma),
        "exposure": exposure,
        "pad_texture": math.sqrt(max(pad_sigma ** 2 - noise_sigma ** 2, 0.0)),
        "ln_pad_distinctiveness": math.log1p(sum(dists) / len(dists)),
        "ln_gap_contrast": math.log1p(sum(gap_d) / len(gap_d)),
        "edge_contrast": contrast,
    }
    return {"features": feats, "colors": colors, "crops": crops, "stds": stds, "labs": labs}


# --------------------------------------------------------------------------
# confidence model
# --------------------------------------------------------------------------
def load_confidence_model():
    if "m" not in _MODEL_CACHE:
        with open(MODEL_PATH) as f:
            _MODEL_CACHE["m"] = json.load(f)
    return _MODEL_CACHE["m"]


# +1: larger value makes a valid strip more likely, -1: less likely (physics, not fitted)
FEATURE_DIRECTION = {"ln_aspect": +1, "coverage": +1, "ln_edge_width": -1, "ln_noise": -1,
                     "exposure": +1, "pad_texture": -1, "ln_pad_distinctiveness": +1,
                     "ln_gap_contrast": +1, "edge_contrast": +1}


def p_from_features(model, features):
    """Series-system probability and per-criterion breakdown."""
    rows, log_p = [], 0.0
    for name, mu, sd, a, b in zip(model["features"], model["mean"], model["std"], model["a"], model["b"]):
        x = features[name]
        z = (x - mu) / sd
        f = 1.0 / (1.0 + math.exp(-(a + b * z)))
        log_p += math.log(max(f, 1e-12))
        rows.append({"feature": name, "label": FEATURE_LABELS[name], "value": round(x, 4),
                     "z": round(z, 3), "a": round(a, 3), "b": round(b, 3),
                     "pass_probability": round(f, 4)})
    return math.exp(log_p), rows


def score(features: dict):
    """P(valid strip | features) and the per-criterion breakdown."""
    return p_from_features(load_confidence_model(), features)


# --------------------------------------------------------------------------
def analyze_strip_image(image_bytes: bytes) -> dict:
    img = prepare_image(image_bytes)
    meas = measure(img)
    p, rows = score(meas["features"])

    if p < DECISION_THRESHOLD:
        worst = min(rows, key=lambda r: r["pass_probability"])
        raise ValueError(
            f"This photo doesn't look like a clear, usable test strip "
            f"(validity {p * 100:.0f}%, needs {DECISION_THRESHOLD * 100:.0f}%). "
            f"Main issue: {REJECT_HINT[worst['feature']]}. "
            f"Please retake it in good lighting with the strip flat and filling most of the frame."
        )

    values, pad_debug, far = {}, [], False
    for feat, bgr, lab, sd in zip(PAD_ORDER, meas["colors"], meas["labs"], meas["stds"]):
        v, de = _match_calibration(lab, CALIBRATION[feat])
        values[feat] = round(float(v), 2)
        far = far or de > 30
        pad_debug.append({"parameter": feat, "bgr": [round(c, 1) for c in bgr],
                          "pad_std": round(sd, 2), "delta_e_to_chart": round(de, 1)})

    if p >= HIGH_CONFIDENCE:
        note = "Strip detected with high confidence. Estimated values filled in below — please review before analyzing."
    else:
        note = (f"Strip detected, but validity is moderate ({p * 100:.0f}%). "
                "Please double-check the estimated values below, or retake the photo in better lighting.")
    if far:
        note += " One or more pad colours lie far outside the reference colour chart, so those values are extrapolated."

    return {
        "estimated_values": values,
        "strip_detected": True,
        "confidence": round(p * 100, 1),
        "confidence_detail": {
            "formula": "P(valid) = Π fⱼ,   fⱼ = 1 / (1 + exp(−(aⱼ + bⱼ·zⱼ))),   zⱼ = (xⱼ − μⱼ) / sⱼ",
            "threshold": round(DECISION_THRESHOLD * 100, 1),
            "breakdown": [r for r in rows if abs(r["b"]) > 1e-6],   # inert criteria (b = 0) are omitted
            "validation": load_confidence_model().get("validation_summary", {}),
        },
        "pad_debug": pad_debug,
        "pad_thumbnails": [_crop_b64(c) for c in meas["crops"]],
        "note": note,
    }
