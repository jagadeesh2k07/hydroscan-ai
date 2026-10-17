"""Regenerate docs/CONFIDENCE_METHODOLOGY.md from the stored validation results.

    python -m scripts.build_methodology
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mm = json.load(open(os.path.join(ROOT, "data", "model_metrics.json")))
sm = json.load(open(os.path.join(ROOT, "vision", "confidence_model.json")))
sr = json.load(open(os.path.join(ROOT, "docs", "strip_validation_report.json")))
pc = lambda x, d=1: f"{x * 100:.{d}f}%"

rel_m = "\n".join(f"| {r['bin']} | {r['n']} | {pc(r['mean_confidence'])} | {pc(r['accuracy'])} |" for r in mm["reliability_table"])
rel_s = "\n".join(f"| {r['bin']} | {r['n']} | {pc(r['mean_predicted_P'])} | {pc(r['observed_valid_fraction'])} |"
                  for r in sr["model_only"]["reliability_table"])
par = "\n".join(f"| `{n}` | {m:.3f} | {s:.3f} | {a:+.2f} | {b:+.2f} | {'active' if abs(b) > 1e-6 else 'inert (b = 0)'} |"
                for n, m, s, a, b in zip(sm["features"], sm["mean"], sm["std"], sm["a"], sm["b"]))
kinds = "\n".join(f"| {k.replace('_', ' ')} | {v['n']} | {pc(v['accept_rate'])} |" for k, v in sr["per_kind"].items())
cf = sr["confusion"]; dev = sm["dev_summary"]; fin = sm["validation_summary"]; mo = sr["model_only"]

doc = f"""# How HydroScan's confidence numbers are calculated and validated

HydroScan shows **two different confidence numbers**. They answer different questions and are computed by
different, fully documented formulas.

| Number | Question it answers | Where |
|---|---|---|
| **Model confidence** | "Given these 4 water readings, how sure is the model about Safe / Caution / Unsafe?" | `ml/calibration.py`, `ml/train_model.py` |
| **Strip validity** | "Is this photo a clear, usable 4-pad test strip?" | `vision/strip_analyzer.py`, `vision/train_confidence.py` |

Both are shown in the app under *"How is this … calculated?"* with the live numbers for that result.

---

## 1. Model confidence (water-quality class)

### Definition
A RandomForest with T = 200 trees gives, for sample **x**, class probabilities

    p_k(x) = (1/T) * sum_t p_(t,k)(x)          (mean of the per-tree leaf class frequencies)

Temperature scaling then calibrates them:

    q_k(x) = softmax_k( ln p_k(x) / tau ),     confidence(x) = max_k q_k(x)

`tau` is one scalar fitted by minimising the negative log-likelihood of **out-of-fold** predictions on the
training split only, so the number is never tuned on the data it is later evaluated on.
`tau = 1` means the forest was already calibrated; `tau > 1` softens over-confidence; `tau < 1` sharpens
under-confidence. Temperature scaling never changes *which* class is predicted, only how sure the model claims to be.

### Validation protocol
1. Stratified 80/20 train/test split ({mm['n_train']} train / {mm['n_test']} test records).
2. 5-fold out-of-fold probabilities on the train split to fit `tau`.
3. All metrics below are computed on the untouched test split, plus 5-fold cross-validated accuracy on all {mm['n_records']} records.

### Results
| Metric | Value |
|---|---|
| Test accuracy | {pc(mm['test_accuracy'])} |
| Test macro-F1 | {pc(mm['test_macro_f1'])} |
| 5-fold CV accuracy | {pc(mm['cv5_accuracy_mean'])} ± {pc(mm['cv5_accuracy_std'])} |
| Fitted temperature `tau` | {mm['temperature_tau']} |
| Log-loss raw → calibrated | {mm['nll_raw']} → {mm['nll_calibrated']} |
| Brier score raw → calibrated | {mm['brier_raw']} → {mm['brier_calibrated']} |
| **Expected calibration error (ECE)** raw → calibrated | {pc(mm['ece_raw'], 2)} → {pc(mm['ece_calibrated'], 2)} |
| Mean confidence on test set | {pc(mm['mean_confidence_test'])} (accuracy {pc(mm['test_accuracy'])}) |

ECE = sum over 10 confidence bins of (n_bin / N) * |accuracy_bin − mean confidence_bin|.
Mean confidence ≈ accuracy and a small ECE mean the number is honest: *when the model says ~90 %, it is right ~90 % of the time.*

Reliability table (test split, calibrated):

| Confidence bin | n | Mean confidence | Actual accuracy |
|---|---|---|---|
{rel_m}

`tau` is close to 1, i.e. a random forest's vote fractions were already well calibrated on this data; the calibration
step is therefore a *check with a documented correction*, not a large adjustment.

### Limits (stated plainly)
* The training data is **synthetic**: labels come from WHO/EPA-inspired threshold rules plus ~4 % random label noise
  (so confidence can never honestly reach 100 %). The metrics measure internal consistency of the model, **not**
  agreement with real laboratory classifications.
* The thresholds themselves are simplified approximations, not regulatory limits.

---

## 2. Strip validity (is this a usable test-strip photo?)

### Definition: a series (weakest-link) system
Nine physical quantities are measured from the photo (`x_j`, table below). Each is standardised,
`z_j = (x_j − mu_j) / s_j`, and converted to the probability that criterion *j* is satisfied:

    f_j = 1 / (1 + exp(−(a_j + b_j * z_j)))
    P(valid) = product over j of f_j                (reliability of a series system)

Quality-control logic is "all criteria must pass", so one failed criterion cannot be out-voted by good ones
(an additive logistic model was tried first and let good scores mask a single catastrophic one). The sign of every
`b_j` is **fixed by physics** (more noise or a wider edge can never help); only the magnitudes and offsets are fitted,
by maximum likelihood with an L1 penalty chosen by 5-fold cross-validation on the training corpus.

A photo is accepted when `P(valid) ≥ c_FA / (c_FA + c_FR) = 1.5 / (1.5 + 1) = 0.60`: wrongly accepting a bad photo
is costed 1.5× wrongly rejecting a good one. Before any of this, deterministic gates reject undecodable files,
images under 150 px, and photos in which no strip-shaped region can be found.

### Measured quantities
| Feature | Definition |
|---|---|
| `ln_aspect` | ln(long side / short side) of the detected strip |
| `coverage` | strip area / frame area |
| `ln_edge_width` | ln(edge-spread width in px), width = grey-level step / 90th-percentile gradient across the strip boundary (a blur of σ px widens it ≈ 2.5σ) |
| `ln_noise` | ln(1 + σₙ), σₙ by Immerkaer's estimator: σ = √(π/2) · Σ\\|I∗N\\| / (6(W−2)(H−2)), N = [[1,−2,1],[−2,4,−2],[1,−2,1]] |
| `exposure` | mean brightness of the strip backing / 255 |
| `pad_texture` | √(max(σ_pad² − σₙ², 0)): pad colour variance with the noise variance removed |
| `ln_pad_distinctiveness` | ln(1 + mean pairwise CIE76 ΔE between the 4 pad colours) |
| `ln_gap_contrast` | ln(1 + ΔE between pad colour and the backing visible in the gaps between pads) |
| `edge_contrast` | \\|mean grey inside strip − mean grey of the surrounding ring\\| / 255 |

### Fitted parameters (`vision/confidence_model.json`)
| Feature | mu | s | a | b | Role |
|---|---|---|---|---|---|
{par}

Features marked *inert* received `b = 0`: the deterministic strip-detection gate already enforces them
(e.g. aspect ratio, minimum size), so the fit gives them no extra weight. They stay in the file for transparency.

### Data and protocol
No labelled corpus of real strip photos exists for this project, so the model is fitted and validated on a large
**synthetic** corpus (`vision/synthetic.py`) with an explicit, auditable label definition (see that file's docstring).
Valid photos vary tilt (±35°, plus vertical), light colour cast, exposure, vignette, noise, blur ≤ 1.6 px, JPEG quality
45–95, backgrounds and pad colours (drawn from the Lab calibration chart). Invalid photos include noise, flat walls,
gradients, random shapes, single-colour bars, two-colour bars, blurred (≥ 3.5 px) strips, noisy strips, low-contrast
strips, under-exposed strips, strips with textured pads, and tiny strips.

| Set | Seed | Size | Used for |
|---|---|---|---|
| Training | 1000 | {sm['fit']['n_images_generated']} photos | fitting `a_j, b_j` (only photos clearing the gates) |
| Development | 2000 | 500 + 500 | looking at errors while designing the features |
| **Final test** | **3000** | **500 + 500** | **never used for any decision: the headline numbers** |

### Results on the final test set (end-to-end: gates + model)
| Metric | Value |
|---|---|
| Overall accuracy | {pc(fin['overall_accuracy'])} (development set: {pc(dev['overall_accuracy'])}) |
| Valid strips accepted (true-accept rate) | {pc(fin['true_accept_rate'])} |
| Invalid photos wrongly accepted (false-accept rate) | {pc(fin['false_accept_rate'])} |
| Precision of accepted photos | {pc(fin['precision'])} |
| ROC-AUC of P(valid) (photos clearing the gates) | {mo['roc_auc']} |
| Brier score of P(valid) | {mo['brier']} |
| ECE of P(valid) | {pc(mo['ece'], 2)} |
| Confusion matrix | TP {cf['TP']}, FP {cf['FP']}, FN {cf['FN']}, TN {cf['TN']} |

Development and final-test results agree closely, so the headline figures are not inflated by design iteration.

Accept rate by photo type (valid should be high, everything else low):

| Photo type | n | Accepted |
|---|---|---|
{kinds}

Reliability of P(valid) (photos clearing the gates):

| Predicted P bin | n | Mean predicted P | Observed valid fraction |
|---|---|---|---|
{rel_s}

### Known weaknesses
* **Synthetic only.** These numbers prove the *method* is internally consistent and well-calibrated against a
  controlled, auditable test; they are **not** a guarantee for arbitrary real phone photos. Run
  `python -m vision.validate --real DIR` (folders `DIR/valid` and `DIR/invalid`) to obtain the same metrics on real photos.
* A photographed **patterned/printed surface** can still be accepted: {pc(sr['per_kind']['textured_pads']['accept_rate'])} of the
  "textured pads" look-alikes passed on the final test set. Genuine pad texture overlaps with JPEG artefacts in valid
  photos, so no single criterion separates them completely. This is the model's weakest case.
* The layout assumed is four pads in equal segments along the strip with backing between them. Strips with a handle
  region or a different pad count/spacing will score lower or be rejected.
* The colour chart that converts pad colour to pH / chlorine / hardness / nitrate is a generic approximation, not a
  brand calibration; pad colours far from the chart (ΔE > 30) are flagged as extrapolated. Validity says the photo is
  *usable*, not that the estimated values are chemically exact, which is why the values stay editable.

## Reproduce
```
python -m ml.train_model                 # model confidence: training + calibration + metrics
python -m vision.train_confidence        # strip validity: fit + dev/final-test evaluation (resumable)
python -m vision.validate --seed 3000    # re-run the final-test evaluation
python -m tests.test_strip_scenarios     # 10 hand-built regression scenarios
python -m scripts.build_methodology      # regenerate this document from the stored results
```
"""
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
open(os.path.join(ROOT, "docs", "CONFIDENCE_METHODOLOGY.md"), "w").write(doc)
print("written", len(doc), "chars")
