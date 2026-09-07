"""
strip_analyzer.py
==================
Computer-vision pipeline that estimates water-quality parameters from a
photo of a 4-pad test strip using OpenCV.

Pipeline:
  1. Load & validate the image (decode, resolution check, blur check).
  2. Locate the strip: threshold + contour detection to find the largest
     elongated rectangular region (the physical strip against a background).
     If no confident strip-shaped region is found, the scan is REJECTED
     with a clear error — earlier versions silently fell back to guessing
     on the whole photo, which produced confident-looking numbers for any
     image at all (even ones with no strip in them). That fallback has
     been removed.
  3. Sample the four reagent pads along the strip's long axis.
  4. Validate each pad region: a real reagent pad is a small, fairly
     uniform block of solid color. If a sampled region has high internal
     color variance, it's more likely background/clutter than a real pad,
     so it lowers that pad's confidence.
  5. Convert each pad's average color to Lab color space and match it
     against a calibration chart (color -> parameter value) using
     inverse-distance interpolation, mimicking how commercial strip charts
     work.
  6. Compute an overall confidence score from strip-shape strength + pad
     uniformity. Scans below a minimum confidence are rejected outright
     rather than silently auto-filling numbers a user might trust.

NOTE ON ACCURACY: real strip-to-value calibration depends on the specific
test-strip brand, lighting conditions, and camera color response. This
module implements a genuine, general-purpose CV pipeline (strip
segmentation + pad localization + Lab-space color matching against a
calibration table) but the bundled calibration table is a reasonable
approximation, not a certified calibration for any specific commercial kit.
Users can and should edit the auto-filled values before analysis — the UI
always allows this, and now the pipeline also refuses to guess when the
input clearly isn't a usable strip photo.
"""
import cv2
import numpy as np
import base64


# ---------------------------------------------------------------------------
# Calibration chart: for each parameter, a list of (Lab_color, value) anchor
# points sampled from typical commercial 4-in-1 strip color charts. Colors
# are approximate averages in CIE Lab space (L, a, b).
# ---------------------------------------------------------------------------
CALIBRATION = {
    "ph": [
        ((60, 45, 40), 5.0),
        ((65, 30, 30), 6.0),
        ((70, 5, 15), 6.5),
        ((75, -10, 5), 7.0),
        ((70, -20, -5), 7.5),
        ((60, -25, -20), 8.0),
        ((50, -15, -35), 8.5),
        ((45, 0, -40), 9.0),
    ],
    "chlorine": [
        ((95, -2, 5), 0.0),
        ((90, 5, 20), 0.5),
        ((85, 15, 35), 1.0),
        ((78, 30, 40), 2.0),
        ((65, 45, 35), 3.0),
        ((50, 55, 25), 4.0),
        ((40, 60, 15), 5.0),
    ],
    "hardness": [
        ((92, -5, 10), 0),
        ((85, 0, 25), 60),
        ((75, 10, 40), 120),
        ((65, 20, 45), 180),
        ((55, 35, 40), 250),
        ((45, 45, 30), 350),
        ((35, 50, 20), 450),
    ],
    "nitrate": [
        ((90, -10, 0), 0),
        ((80, 5, 15), 5),
        ((70, 25, 20), 10),
        ((60, 45, 25), 20),
        ((50, 65, 30), 45),
        ((40, 75, 35), 70),
        ((30, 80, 40), 100),
    ],
}

PAD_ORDER = ["ph", "chlorine", "hardness", "nitrate"]

# ---------------------------------------------------------------------------
# Validation thresholds — tuned conservatively (lenient enough not to reject
# genuine, slightly-imperfect phone photos, strict enough to reject clearly
# invalid input like blank, blurry, or strip-less images).
# ---------------------------------------------------------------------------
MIN_DIMENSION_PX = 150          # reject images smaller than this on either side
MIN_SHARPNESS = 12.0            # Laplacian-variance blur threshold
MAX_PAD_COLOR_STD = 42.0        # per-channel std dev above this = "not a solid pad"
MIN_OVERALL_CONFIDENCE = 35.0   # scans scoring below this are rejected outright


def _bgr_to_lab(bgr_pixel):
    """Convert a single average BGR pixel (0-255) to Lab (OpenCV scaling)."""
    patch = np.uint8([[bgr_pixel]])
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0][0]
    # OpenCV Lab: L in [0,255] -> scale to [0,100]; a,b centered at 128 -> [-127,127]
    L = float(lab[0]) * 100.0 / 255.0
    a = float(lab[1]) - 128.0
    b = float(lab[2]) - 128.0
    return (L, a, b)


def _match_calibration(lab_color, table):
    """Nearest-neighbor + linear interpolation between the two closest
    calibration anchors, in Lab color space (perceptually uniform)."""
    dists = []
    for anchor_lab, value in table:
        d = sum((c1 - c2) ** 2 for c1, c2 in zip(lab_color, anchor_lab)) ** 0.5
        dists.append((d, value))
    dists.sort(key=lambda t: t[0])
    (d1, v1), (d2, v2) = dists[0], dists[1]
    if d1 + d2 == 0:
        return v1
    # Inverse-distance weighted interpolation between two nearest anchors
    w1, w2 = d2 / (d1 + d2), d1 / (d1 + d2)
    return v1 * w1 + v2 * w2


def _sharpness_score(img) -> float:
    """Laplacian variance — a standard, simple blur/focus metric. Low values
    mean the image is blurry/out of focus or nearly featureless (e.g. blank)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _locate_strip(img):
    """Find the largest elongated rectangular contour = the test strip.
    Returns (rect, shape_confidence) where rect is None if nothing
    confidently strip-shaped was found. shape_confidence (0-100) reflects
    how strongly the winning contour matches "a strip" (elongated, a
    plausible fraction of the frame) rather than being a marginal match."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    best = None
    best_score = 0
    best_aspect = 0

    # Try both polarities (strip lighter than background, or darker) since
    # we can't assume background color in a user-submitted photo.
    for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, thresh = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < (h * w) * 0.015 or area > (h * w) * 0.90:
                continue
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            if rw == 0 or rh == 0:
                continue
            long_side, short_side = max(rw, rh), min(rw, rh)
            aspect = long_side / short_side
            if aspect < 2.2:  # strips are long & thin — stricter than before
                continue
            score = area
            if score > best_score:
                best_score = score
                best = rect
                best_aspect = aspect

    if best is None:
        return None, 0.0

    # Confidence from shape alone: reward strong aspect ratio (a real strip
    # is usually 3:1 to 8:1) and reasonable frame coverage.
    aspect_conf = min((best_aspect - 2.2) / (6.0 - 2.2), 1.0) * 100
    area_frac = best_score / (h * w)
    coverage_conf = 100 - min(abs(area_frac - 0.25) / 0.25, 1.0) * 40  # sweet spot ~25% of frame
    shape_confidence = max(0.0, min(100.0, 0.6 * aspect_conf + 0.4 * coverage_conf))
    return best, shape_confidence


def _extract_pad_colors(img, rect):
    """Rotate/crop the strip to a straight horizontal band, then sample
    N evenly spaced pads along its length."""
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
        rw, rh = rh, rw

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    x = int(cx - rw / 2)
    y = int(cy - rh / 2)
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x + int(rw), img.shape[1]), min(y + int(rh), img.shape[0])
    strip = rotated[y:y2, x:x2]

    if strip.size == 0:
        return None, None, None

    n_pads = len(PAD_ORDER)
    pad_w = strip.shape[1] / n_pads
    pad_colors = []
    pad_crops = []
    pad_stds = []
    margin_y = int(strip.shape[0] * 0.2)

    for i in range(n_pads):
        px1 = int(i * pad_w + pad_w * 0.25)
        px2 = int(i * pad_w + pad_w * 0.75)
        px1, px2 = max(px1, 0), min(px2, strip.shape[1])
        py1, py2 = margin_y, strip.shape[0] - margin_y
        if px2 <= px1 or py2 <= py1:
            pad_colors.append((200, 200, 200))
            pad_crops.append(None)
            pad_stds.append(MAX_PAD_COLOR_STD)  # treat missing region as low confidence
            continue
        crop = strip[py1:py2, px1:px2]
        pixels = crop.reshape(-1, 3).astype(np.float32)
        avg_bgr = tuple(float(v) for v in pixels.mean(axis=0))
        # Per-channel std dev, averaged across channels — a real reagent pad
        # is a fairly solid color, so low variance is expected. High variance
        # means this region is probably background clutter, not a pad.
        std = float(pixels.std(axis=0).mean())
        pad_colors.append(avg_bgr)
        pad_crops.append(crop)
        pad_stds.append(std)

    return pad_colors, pad_crops, pad_stds


def _crop_to_base64(crop):
    if crop is None or crop.size == 0:
        return None
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def analyze_strip_image(image_bytes: bytes) -> dict:
    """
    Main entry point. Takes raw image bytes (from an uploaded file),
    returns a dict with estimated parameter values, per-pad detected
    colors, pad crop thumbnails, and an overall confidence score.

    Raises ValueError with a user-facing message if the image can't be
    decoded, is too small/blurry, no strip-shaped region can be found, or
    the overall confidence is too low to responsibly auto-fill values.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG photo.")

    h0, w0 = img.shape[:2]
    if min(h0, w0) < MIN_DIMENSION_PX:
        raise ValueError("Image resolution is too low. Please upload a larger, clearer photo.")

    # Resize for consistent processing speed/quality
    scale = 900 / max(h0, w0) if max(h0, w0) > 900 else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))

    sharpness = _sharpness_score(img)
    if sharpness < MIN_SHARPNESS:
        raise ValueError(
            "This photo looks blurry, out of focus, or nearly blank. "
            "Please retake it with good lighting and the camera held steady."
        )

    img = cv2.bilateralFilter(img, 7, 50, 50)  # denoise while keeping edges

    rect, shape_confidence = _locate_strip(img)
    if rect is None:
        raise ValueError(
            "Could not detect a test strip in this photo. Please retake it with "
            "the full strip visible, laid flat against a plain, contrasting "
            "background, filling most of the frame."
        )

    pad_colors, pad_crops, pad_stds = _extract_pad_colors(img, rect)
    if pad_colors is None:
        raise ValueError("Could not isolate test pads from the strip. Try a clearer, well-lit photo.")

    # Per-pad confidence from color uniformity, then blended with shape
    # confidence into one overall score for the whole scan.
    pad_confidences = [max(0.0, 100.0 - (std / MAX_PAD_COLOR_STD) * 100.0) for std in pad_stds]
    avg_pad_confidence = sum(pad_confidences) / len(pad_confidences)
    overall_confidence = round(0.5 * shape_confidence + 0.5 * avg_pad_confidence, 1)

    if overall_confidence < MIN_OVERALL_CONFIDENCE:
        raise ValueError(
            "This photo doesn't look like a clear, in-focus test strip with "
            "distinct color pads — the analysis would be unreliable. Please "
            "retake it in good lighting, strip flat, filling most of the frame."
        )

    results = {}
    pad_debug = []
    for feature, bgr, pad_conf in zip(PAD_ORDER, pad_colors, pad_confidences):
        lab = _bgr_to_lab(bgr)
        value = _match_calibration(lab, CALIBRATION[feature])
        results[feature] = round(float(value), 2)
        pad_debug.append({
            "parameter": feature,
            "bgr": [round(c, 1) for c in bgr],
            "confidence": round(pad_conf, 1),
        })

    pad_thumbnails = [_crop_to_base64(c) for c in pad_crops]

    if overall_confidence >= 70:
        note = "Strip detected with high confidence. Estimated values filled in below — please review before analyzing."
    else:
        note = (f"Strip detected, but confidence is moderate ({overall_confidence:.0f}%). "
                 "Please double-check the estimated values below, or retake the photo in better lighting.")

    return {
        "estimated_values": results,
        "strip_detected": True,
        "confidence": overall_confidence,
        "pad_debug": pad_debug,
        "pad_thumbnails": pad_thumbnails,
        "note": note,
    }

