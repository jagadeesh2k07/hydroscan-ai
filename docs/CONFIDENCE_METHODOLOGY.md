# How HydroScan's confidence numbers are calculated and validated

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
1. Stratified 80/20 train/test split (960 train / 240 test records).
2. 5-fold out-of-fold probabilities on the train split to fit `tau`.
3. All metrics below are computed on the untouched test split, plus 5-fold cross-validated accuracy on all 1200 records.

### Results
| Metric | Value |
|---|---|
| Test accuracy | 93.3% |
| Test macro-F1 | 93.3% |
| 5-fold CV accuracy | 93.5% ± 0.7% |
| Fitted temperature `tau` | 1.0239 |
| Log-loss raw → calibrated | 0.3013 → 0.2993 |
| Brier score raw → calibrated | 0.1069 → 0.1072 |
| **Expected calibration error (ECE)** raw → calibrated | 1.47% → 1.31% |
| Mean confidence on test set | 92.8% (accuracy 93.3%) |

ECE = sum over 10 confidence bins of (n_bin / N) * |accuracy_bin − mean confidence_bin|.
Mean confidence ≈ accuracy and a small ECE mean the number is honest: *when the model says ~90 %, it is right ~90 % of the time.*

Reliability table (test split, calibrated):

| Confidence bin | n | Mean confidence | Actual accuracy |
|---|---|---|---|
| 0.4-0.5 | 1 | 48.3% | 0.0% |
| 0.5-0.6 | 5 | 56.6% | 60.0% |
| 0.6-0.7 | 12 | 65.0% | 75.0% |
| 0.7-0.8 | 13 | 74.7% | 76.9% |
| 0.8-0.9 | 18 | 85.8% | 88.9% |
| 0.9-1.0 | 191 | 97.6% | 97.4% |

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
| `ln_noise` | ln(1 + σₙ), σₙ by Immerkaer's estimator: σ = √(π/2) · Σ\|I∗N\| / (6(W−2)(H−2)), N = [[1,−2,1],[−2,4,−2],[1,−2,1]] |
| `exposure` | mean brightness of the strip backing / 255 |
| `pad_texture` | √(max(σ_pad² − σₙ², 0)): pad colour variance with the noise variance removed |
| `ln_pad_distinctiveness` | ln(1 + mean pairwise CIE76 ΔE between the 4 pad colours) |
| `ln_gap_contrast` | ln(1 + ΔE between pad colour and the backing visible in the gaps between pads) |
| `edge_contrast` | \|mean grey inside strip − mean grey of the surrounding ring\| / 255 |

### Fitted parameters (`vision/confidence_model.json`)
| Feature | mu | s | a | b | Role |
|---|---|---|---|---|---|
| `ln_aspect` | 1.684 | 0.280 | +12.00 | +0.00 | inert (b = 0) |
| `coverage` | 0.104 | 0.055 | +12.00 | +0.00 | inert (b = 0) |
| `ln_edge_width` | 0.853 | 0.665 | +4.46 | -2.46 | active |
| `ln_noise` | 0.686 | 0.769 | +12.00 | -5.25 | active |
| `exposure` | 0.576 | 0.199 | +12.00 | +10.06 | active |
| `pad_texture` | 6.275 | 8.443 | +4.01 | -1.60 | active |
| `ln_pad_distinctiveness` | 3.347 | 0.934 | +12.00 | +0.00 | inert (b = 0) |
| `ln_gap_contrast` | 3.077 | 1.010 | +4.62 | +3.92 | active |
| `edge_contrast` | 0.255 | 0.157 | +12.00 | +0.00 | inert (b = 0) |

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
| Training | 1000 | 1800 photos | fitting `a_j, b_j` (only photos clearing the gates) |
| Development | 2000 | 500 + 500 | looking at errors while designing the features |
| **Final test** | **3000** | **500 + 500** | **never used for any decision: the headline numbers** |

### Results on the final test set (end-to-end: gates + model)
| Metric | Value |
|---|---|
| Overall accuracy | 93.0% (development set: 93.9%) |
| Valid strips accepted (true-accept rate) | 92.4% |
| Invalid photos wrongly accepted (false-accept rate) | 6.4% |
| Precision of accepted photos | 93.5% |
| ROC-AUC of P(valid) (photos clearing the gates) | 0.9829 |
| Brier score of P(valid) | 0.0561 |
| ECE of P(valid) | 5.23% |
| Confusion matrix | TP 462, FP 32, FN 38, TN 468 |

Development and final-test results agree closely, so the headline figures are not inflated by design iteration.

Accept rate by photo type (valid should be high, everything else low):

| Photo type | n | Accepted |
|---|---|---|
| blobs | 44 | 0.0% |
| blurred strip | 42 | 9.5% |
| dark strip | 40 | 0.0% |
| flat | 42 | 0.0% |
| gradient | 43 | 0.0% |
| low contrast strip | 40 | 0.0% |
| noise | 41 | 0.0% |
| noisy strip | 41 | 0.0% |
| textured pads | 40 | 62.5% |
| tiny strip | 40 | 0.0% |
| two tone bar | 43 | 4.7% |
| uniform bar | 44 | 2.3% |
| valid | 500 | 92.4% |

Reliability of P(valid) (photos clearing the gates):

| Predicted P bin | n | Mean predicted P | Observed valid fraction |
|---|---|---|---|
| 0.0-0.1 | 187 | 0.8% | 3.7% |
| 0.1-0.2 | 22 | 14.6% | 13.6% |
| 0.2-0.3 | 16 | 23.5% | 31.2% |
| 0.3-0.4 | 14 | 34.1% | 21.4% |
| 0.4-0.5 | 17 | 44.5% | 29.4% |
| 0.5-0.6 | 11 | 53.9% | 45.5% |
| 0.6-0.7 | 15 | 64.6% | 33.3% |
| 0.7-0.8 | 16 | 74.9% | 37.5% |
| 0.8-0.9 | 27 | 84.9% | 59.3% |
| 0.9-1.0 | 436 | 97.5% | 99.8% |

### Known weaknesses
* **Synthetic only.** These numbers prove the *method* is internally consistent and well-calibrated against a
  controlled, auditable test; they are **not** a guarantee for arbitrary real phone photos. Run
  `python -m vision.validate --real DIR` (folders `DIR/valid` and `DIR/invalid`) to obtain the same metrics on real photos.
* A photographed **patterned/printed surface** can still be accepted: 62.5% of the
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
