"""
Hand-built regression scenarios for the strip scanner (independent of the
statistical corpus used to fit the confidence model).

    python -m tests.test_strip_scenarios
"""
import sys
import cv2
import numpy as np
from vision.strip_analyzer import analyze_strip_image


def strip(warm=False, tilt=0, blur=0, base=200):
    img = np.full((500, 700, 3), 45, np.uint8)
    x0, x1 = 100, 600
    cv2.rectangle(img, (x0, 200), (x1, 300), (base, base, base), -1)
    pads = [(90, 60, 100), (60, 120, 170), (70, 90, 140), (60, 70, 150)]
    seg = (x1 - x0) / 4
    for i, c in enumerate(pads):
        cv2.rectangle(img, (int(x0 + i * seg + 15), 215), (int(x0 + (i + 1) * seg - 15), 285), c, -1)
    if warm:
        f = img.astype(np.float32); f[:, :, 2] *= 1.2; f[:, :, 0] *= 0.85; img = np.clip(f, 0, 255).astype(np.uint8)
    if tilt:
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((350, 250), tilt, 1.0), (700, 500), borderValue=(45, 45, 45))
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def bar(color):
    img = np.full((500, 700, 3), 230, np.uint8)
    cv2.rectangle(img, (150, 220), (550, 280), color, -1)
    return img


rng = np.random.default_rng(0)
SCENARIOS = {                       # name: (image, should_be_accepted)
    "good light":              (strip(), True),
    "warm light cast":         (strip(warm=True), True),
    "tilted 15 deg":           (strip(tilt=15), True),
    "tilted -25 deg":          (strip(tilt=-25), True),
    "heavily blurred strip":   (strip(blur=9), False),
    "random noise":            (rng.integers(0, 256, (500, 700, 3), dtype=np.uint8), False),
    "blank wall":              (np.full((500, 700, 3), 200, np.uint8), False),
    "tiny image":              (cv2.resize(strip(), (100, 70)), False),
    "single-colour bar":       (bar((140, 90, 60)), False),
    "round object":            (cv2.circle(np.full((500, 700, 3), 220, np.uint8), (350, 250), 120, (100, 80, 150), -1), False),
}


def main():
    bad = 0
    for name, (img, want) in SCENARIOS.items():
        ok, buf = cv2.imencode(".png", img)
        try:
            r = analyze_strip_image(buf.tobytes())
            got, detail = True, f"validity={r['confidence']}%  {r['estimated_values']}"
        except ValueError as e:
            got, detail = False, str(e)[:80]
        flag = "PASS" if got == want else "FAIL"
        bad += flag == "FAIL"
        print(f"{flag}  {name:<24} {'accepted' if got else 'rejected':<9} {detail}")
    print("\nALL SCENARIOS PASSED" if not bad else f"\n{bad} SCENARIO(S) FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
