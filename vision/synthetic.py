"""
synthetic.py
=============
Physically-parameterised generator of test-strip photos, used to FIT and
VALIDATE the strip-confidence model (see strip_analyzer.py).

Label definition (by construction, stated explicitly so the validation is
auditable):
  VALID   = a 4-pad strip, fully in frame, aspect 4-8:1, 3-50 % of the frame,
            blur sigma <= 1.6 px, sensor noise sigma <= 10, contrast with the
            background >= 28 grey levels, exposure gain 0.55-1.15, any tilt
            +/-35 deg, random warm/cool light cast, vignette, JPEG q 45-95.
  INVALID = no strip, OR a strip that violates the limits above by a clear
            margin (blur sigma >= 3.5, noise sigma >= 28, contrast <= 10,
            gain <= 0.28, strip < 1 % of frame), OR a non-strip look-alike
            (single-colour bar, two-colour bar, bar with textured pads).
Photos in the ambiguous bands between the two limits are never generated, so
no sample is labelled on a coin-flip.
"""
import math
import numpy as np
import cv2
from vision.strip_analyzer import CALIBRATION, PAD_ORDER

NEG_TYPES = ["noise", "flat", "gradient", "blobs", "uniform_bar", "two_tone_bar",
             "blurred_strip", "noisy_strip", "low_contrast_strip", "dark_strip",
             "textured_pads", "tiny_strip"]


def _lab_to_bgr(L, a, b):
    lab = np.uint8([[[np.clip(L * 255 / 100, 0, 255), np.clip(a + 128, 0, 255), np.clip(b + 128, 0, 255)]]])
    return tuple(int(v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0])


def _background(rng, W, H, gray, tint=True):
    t = rng.uniform(-14, 14, 3) if tint else np.zeros(3)
    base = np.clip(np.array([gray, gray, gray]) + t, 0, 255)
    img = np.empty((H, W, 3), np.float32)
    img[:] = base
    style = rng.integers(0, 4)
    if style == 1:                                   # grainy
        img += rng.normal(0, rng.uniform(2, 9), img.shape)
    elif style == 2:                                 # gradient
        g = np.linspace(-1, 1, W)[None, :, None] * rng.uniform(5, 22)
        img += g if rng.random() < 0.5 else g.transpose(1, 0, 2)[:H, :1] if False else g
    elif style == 3:                                 # wood-ish stripes
        yy = np.arange(H)[:, None, None]
        img += np.sin(yy / rng.uniform(3, 11)) * rng.uniform(3, 11) + rng.normal(0, 3, img.shape)
    return img


def _apply_camera(img, rng, gain, noise, blur, cast=True, vignette=True):
    img = img.astype(np.float32)
    if cast:
        c = rng.uniform(0.82, 1.2, 3)
        img *= (c / c.mean())
    img *= gain
    if vignette:
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        img *= (1 - rng.uniform(0, 0.25) * r / 1.414)[..., None]
    if blur > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(45, 96))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def render_strip(rng, W, H, *, pad_mode="real", blur=None, noise=None, gain=None,
                 low_contrast=False, long_frac=None, angle=None):
    blur = rng.uniform(0, 1.6) if blur is None else blur
    noise = rng.uniform(0, 10) if noise is None else noise
    gain = rng.uniform(0.55, 1.15) if gain is None else gain

    backing_gray = rng.uniform(200, 250)
    if low_contrast:
        bg_gray = backing_gray + rng.choice([-1, 1]) * rng.uniform(0, 7) / max(gain, 0.2)
    elif gain < 0.5:                       # deliberately under-exposed: no contrast guarantee
        bg_gray = rng.uniform(15, 175)
    else:
        while True:
            bg_gray = rng.uniform(15, 175) if rng.random() < 0.7 else rng.uniform(205, 250)
            if abs(backing_gray - bg_gray) * gain >= 28:
                break
    img = _background(rng, W, H, float(np.clip(bg_gray, 0, 255)))

    # --- geometry -----------------------------------------------------------
    aspect = rng.uniform(4, 8)
    ang = (rng.uniform(-35, 35) + (90 if rng.random() < 0.15 else 0)) if angle is None else angle
    th = math.radians(ang)
    L = (rng.uniform(0.45, 0.88) if long_frac is None else long_frac) * W
    S = L / aspect
    ext_w, ext_h = abs(L * math.cos(th)) + abs(S * math.sin(th)), abs(L * math.sin(th)) + abs(S * math.cos(th))
    k = min(1.0, 0.92 * W / ext_w, 0.92 * H / ext_h)
    L, S, ext_w, ext_h = L * k, S * k, ext_w * k, ext_h * k
    cx = rng.uniform(ext_w / 2 + 3, W - ext_w / 2 - 3)
    cy = rng.uniform(ext_h / 2 + 3, H - ext_h / 2 - 3)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])

    def poly(x0, x1, y0, y1):                       # local rect -> image polygon
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float) @ R.T + [cx, cy]
        return np.round(pts).astype(np.int32)

    backing = tuple(int(v) for v in np.clip(backing_gray + rng.uniform(-6, 6, 3), 0, 255))
    seg = L / 4
    if pad_mode == "uniform":
        col = tuple(int(v) for v in rng.integers(30, 200, 3))
        cv2.fillConvexPoly(img, poly(-L / 2, L / 2, -S / 2, S / 2), col)
    elif pad_mode == "two_tone":
        a, b = (tuple(int(v) for v in rng.integers(30, 220, 3)) for _ in range(2))
        cv2.fillConvexPoly(img, poly(-L / 2, 0, -S / 2, S / 2), a)
        cv2.fillConvexPoly(img, poly(0, L / 2, -S / 2, S / 2), b)
    else:
        cv2.fillConvexPoly(img, poly(-L / 2, L / 2, -S / 2, S / 2), backing)
        my = 0.18 * S
        for i, name in enumerate(PAD_ORDER):
            x0, x1 = -L / 2 + i * seg + 0.12 * seg, -L / 2 + (i + 1) * seg - 0.12 * seg
            if pad_mode == "textured":
                p = poly(x0, x1, -S / 2 + my, S / 2 - my)
                tex = np.zeros(img.shape[:2], np.uint8)
                cv2.fillConvexPoly(tex, p, 255)
                mean = rng.uniform(60, 190, 3)
                # spatially-correlated (low-frequency) texture: survives camera blur,
                # like a patterned/printed surface photographed instead of a flat pad
                coarse = rng.normal(0, 1, (max(2, H // 6), max(2, W // 6), 3)).astype(np.float32)
                field = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)
                noise_img = mean + field * rng.uniform(30, 55)
                img[tex > 0] = np.clip(noise_img[tex > 0], 0, 255)
            else:
                L_, a_, b_ = CALIBRATION[name][rng.integers(0, len(CALIBRATION[name]))][0]
                col = _lab_to_bgr(L_ + rng.normal(0, 2.5), a_ + rng.normal(0, 3), b_ + rng.normal(0, 3))
                cv2.fillConvexPoly(img, poly(x0, x1, -S / 2 + my, S / 2 - my), col)

    for _ in range(int(rng.integers(0, 4))):        # background clutter
        c = tuple(int(v) for v in np.clip(bg_gray + rng.uniform(-35, 35, 3), 0, 255))
        cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(4, 18)), c, -1)

    return _apply_camera(img, rng, gain, noise, blur)


def render_negative(rng, kind, W, H):
    if kind == "noise":
        return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    if kind == "flat":
        g = rng.uniform(20, 235)
        return _apply_camera(_background(rng, W, H, g, tint=True) * 0 + g, rng, 1.0, rng.uniform(0, 3), 0.0)
    if kind == "gradient":
        a, b = rng.uniform(10, 240, 2)
        img = np.repeat(np.linspace(a, b, W)[None, :, None], H, 0).repeat(3, 2)
        return _apply_camera(img, rng, 1.0, rng.uniform(0, 4), rng.uniform(0, 1.5))
    if kind == "blobs":
        img = _background(rng, W, H, rng.uniform(40, 220))
        for _ in range(int(rng.integers(3, 9))):
            col = tuple(int(v) for v in rng.integers(0, 256, 3))
            if rng.random() < 0.5:
                cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(20, 110)), col, -1)
            else:
                x, y = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 60))
                cv2.rectangle(img, (x, y), (x + int(rng.integers(40, 200)), y + int(rng.integers(40, 160))), col, -1)
        return _apply_camera(img, rng, rng.uniform(0.6, 1.1), rng.uniform(0, 8), rng.uniform(0, 1.5))
    if kind == "uniform_bar":
        return render_strip(rng, W, H, pad_mode="uniform")
    if kind == "two_tone_bar":
        return render_strip(rng, W, H, pad_mode="two_tone")
    if kind == "blurred_strip":
        return render_strip(rng, W, H, blur=rng.uniform(3.5, 9.0))
    if kind == "noisy_strip":
        return render_strip(rng, W, H, noise=rng.uniform(28, 55))
    if kind == "low_contrast_strip":
        return render_strip(rng, W, H, low_contrast=True)
    if kind == "dark_strip":
        return render_strip(rng, W, H, gain=rng.uniform(0.10, 0.28))
    if kind == "textured_pads":
        return render_strip(rng, W, H, pad_mode="textured")
    if kind == "tiny_strip":
        return render_strip(rng, W, H, long_frac=rng.uniform(0.07, 0.13))
    raise ValueError(kind)


def generate(n_valid, n_invalid, seed, type_offset=0):
    """Yield (image, label, kind). label 1 = valid, 0 = invalid."""
    rng = np.random.default_rng(seed)
    items = [("valid", 1)] * n_valid + [(NEG_TYPES[(i + type_offset) % len(NEG_TYPES)], 0) for i in range(n_invalid)]
    rng.shuffle(items)
    for kind, label in items:
        W, H = int(rng.integers(560, 900)), int(rng.integers(420, 700))
        img = render_strip(rng, W, H) if label else render_negative(rng, kind, W, H)
        yield img, label, kind
