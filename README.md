# HydroScan AI — Explainable Water Quality Analysis System

HydroScan AI analyzes drinking-water quality from manual parameter readings
(pH, chlorine, hardness, nitrate) **or** a photo of a 4-pad test strip, and
predicts **Safe / Caution / Unsafe** — with a full, genuine explanation of
*why*, not just a label and a confidence score.

Explainability is the point of this project, not model complexity: a
lightweight RandomForest + SHAP is used deliberately so every prediction can
be attributed, in plain English, back to the parameters that produced it.

---

## Features

- **Manual entry** — pH, chlorine (mg/L), hardness (ppm), nitrate (mg/L)
- **Test-strip photo scan** — OpenCV detects the strip, validates the photo
  (strip shape, focus, noise, exposure, pad texture and colour separation), samples the 4 pad
  colors, and estimates parameter values via Lab-color calibration matching.
  Invalid photos (no strip, blurry, blank, random objects) are rejected with
  a clear reason instead of guessing. Valid scans auto-fill the form with a
  visible confidence score — you can still review/edit before analyzing.
- **Prediction** — RandomForestClassifier trained on a 1,200-record
  synthetic-but-realistic dataset, ~93% test accuracy
- **Explainable AI** — SHAP-based per-parameter contribution percentages,
  the single most influential parameter, plain-English reasons, probable
  causes, and actionable recommendations
- **Dashboard** — prediction banner, confidence ring, risk indicator track,
  parameter cards, an "Attribution Spectrum" contribution visualization,
  full XAI panel, causes, recommendations, and analysis summary
- **PDF reports** — downloadable, professional report per analysis (ReportLab)

---

## Tech Stack

| Layer          | Choice                                             |
|----------------|-----------------------------------------------------|
| Frontend       | Vanilla HTML / CSS / JavaScript (no framework)      |
| Backend        | Flask (Python)                                      |
| Database       | SQLite                                              |
| ML             | scikit-learn (RandomForestClassifier)               |
| Explainability | SHAP (TreeExplainer)                                |
| Computer Vision| OpenCV                                              |
| PDF Reports    | ReportLab                                           |
| Data           | NumPy / Pandas                                      |

---

## Project Structure

```
hydroscan-ai/
├── app.py                     # Flask app entry point / bootstrap
├── config.py                  # Configuration (secret key, paths)
├── requirements.txt
├── blueprints/
│   └── main.py                 # Routes: dashboard, predict, strip-scan, reports
├── ml/
│   ├── domain_knowledge.py       # Safety thresholds, reasons, causes, recommendations
│   ├── dataset_generator.py       # Synthetic 1200-record dataset generator
│   ├── train_model.py              # RandomForest training + metrics
│   ├── explainability.py            # SHAP-based Explainer (the XAI core)
│   ├── calibration.py                 # temperature scaling, ECE, Brier (confidence maths)
│   └── model_manager.py              # In-memory model singleton
├── vision/
│   ├── strip_analyzer.py           # OpenCV strip detection, features, validity score
│   ├── confidence_model.json       # fitted validity model + validation summary
│   ├── synthetic.py / train_confidence.py / validate.py   # corpus, fitting, evaluation
├── tests/test_strip_scenarios.py   # hand-built regression scenarios
├── docs/CONFIDENCE_METHODOLOGY.md  # formulas, validation tables, limits
├── reports_engine/
│   └── report_generator.py          # ReportLab PDF report builder
├── database/
│   ├── schema.sql                    # analyses table
│   └── db.py                          # Data access layer
├── templates/                        # Jinja2 templates
├── static/
│   ├── css/style.css                  # Design system + all component styles
│   ├── js/                             # dashboard.js, main.js
│   ├── images/logo.png, icon.png
│   ├── uploads/                        # Saved strip photos (runtime)
│   └── reports/                        # Generated PDF reports (runtime)
├── data/                              # Dataset + trained model (runtime)
└── instance/                          # SQLite database (runtime)
```

---

## Setup

**Requirements:** Python 3.10+

```bash
cd hydroscan-ai
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000**

On first run, the app automatically:
1. Generates the 1,200-record training dataset (`data/water_quality_dataset.csv`)
2. Trains the initial RandomForest model (`data/water_quality_model.pkl`)
3. Initializes the SQLite database (`instance/hydroscan.db`)

No manual setup steps beyond `pip install` are required.

---

## How the Explainability Works

1. A RandomForestClassifier predicts Safe / Caution / Unsafe from the four
   parameters, with a confidence score from `predict_proba`.
2. A SHAP `TreeExplainer` computes each parameter's signed contribution to
   the **predicted class specifically** (not just global feature importance).
3. Contributions are converted to percentages that sum to 100%, and the
   top contributor is identified.
4. Rule-based domain knowledge (`ml/domain_knowledge.py`) turns each
   flagged parameter's status into a plain-English reason, a set of likely
   causes, and concrete recommendations — ordered by how much that
   parameter actually drove *this* result.

This two-layer design (SHAP for *how much*, domain rules for *what it means*)
is what makes the output genuinely explainable rather than a black-box
label with a bolted-on confidence number.

**Note:** the domain thresholds (pH 6.5–8.5, chlorine 0.2–2.0 mg/L, hardness
≤150/≤300 ppm, nitrate ≤10/≤45 mg/L) are simplified approximations inspired
by WHO/EPA guidance for portfolio/educational purposes — not a substitute
for certified laboratory testing.

## How the Strip Scan Works

`vision/strip_analyzer.py` (OpenCV):

1. **Deterministic gates** — undecodable file, resolution < 150 px, or no strip-shaped
   region found (four independent segmentation methods; the best candidate is tightened
   to the strip's real boundary).
2. **White-balance** the pad colours using the strip's own backing as the neutral reference
   (the same strip under warm light reads the same: pH 8.77 vs 8.78 in testing).
3. **Measure nine physical quantities** — shape, size, edge-spread width (focus), Immerkaer
   noise σ, exposure, pad texture, pad-to-pad colour difference, pad/backing gap contrast,
   strip/background contrast.
4. **Strip validity = P(valid)** — a weakest-link (series-system) model,
   `P = Π σ(aⱼ + bⱼ·zⱼ)`, fitted by maximum likelihood with physics-fixed signs and
   **validated on a held-out test set** (93.0 % accuracy, ROC-AUC 0.983 on synthetic photos —
   see the methodology document for the full tables and limits). Accepted when
   `P ≥ 0.6` (false accept costed 1.5× a false reject). Rejections name the weakest criterion.
5. **Colour → value** by inverse-distance interpolation on a Lab chart; pads far from the
   chart (ΔE > 30) are flagged as extrapolated.

## Confidence methodology (formulas + validation)

Two different confidence numbers are shown, each with its formula, live values and validation
results in the app ("How is this … calculated?") and in the PDF report:

* **Model confidence** — `qₖ = softmax(ln pₖ / τ)`, `pₖ` = mean vote of 200 trees, `τ` fitted
  on out-of-fold data. Validated: 5-fold CV accuracy 93.5 % ± 0.7 %, calibration error (ECE) 1.3 %.
* **Strip validity** — the series-system model above.

Full derivation, parameter tables, reliability tables, per-photo-type results and known
weaknesses: **[docs/CONFIDENCE_METHODOLOGY.md](docs/CONFIDENCE_METHODOLOGY.md)**.
All metrics are on synthetic data (no labelled real-photo set exists); `python -m vision.validate --real DIR`
reproduces them on real photos you label (`DIR/valid`, `DIR/invalid`).

```bash
python -m ml.train_model               # model confidence: train + calibrate + metrics
python -m vision.train_confidence      # strip validity: fit + validate (resumable)
python -m vision.validate --seed 3000  # re-run the final-test evaluation
python -m tests.test_strip_scenarios   # 10 hand-built regression scenarios
```

The bundled calibration table is a reasonable general-purpose approximation,
not a certified calibration for any specific commercial test-strip brand —
which is why the UI always lets you review and edit the estimated values
before running the analysis, even on a high-confidence scan.

---

## Disclaimer

HydroScan AI is an educational/portfolio tool. It provides an AI-assisted
estimate, not a certified laboratory analysis. For regulatory, medical, or
legal decisions, confirm results with an accredited water-testing laboratory.

---

## Deploying to Render or Railway

The app ships with a `Procfile`, `requirements.txt` (including `gunicorn`),
`runtime.txt`, and a `render.yaml` blueprint — both platforms can run it
with no code changes.

### Render

1. Push this repo to GitHub (see below).
2. In Render: **New → Blueprint**, connect the repo, and it will read
   `render.yaml` automatically (build command, start command, and env vars
   are already configured). Or set up manually as a **Web Service**:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
3. Set the environment variable `HYDROSCAN_SECRET_KEY` in the Render
   dashboard to a random string (don't leave the `config.py` default for
   anything public-facing).

### Railway

This repository includes `railway.toml`, `Procfile`, `runtime.txt`, and
`requirements.txt` for Railway's Nixpacks Python deployment.

1. Push the latest code to GitHub, including `railway.toml`.
2. Open <https://railway.com/dashboard>, choose **New Project → Deploy from GitHub Repo**, and select this repository. Authorize GitHub access if asked.
3. Open the created service and check **Deployments → Build Logs** and **Deploy Logs**. The start command runs Gunicorn and listens on Railway's `$PORT`.
4. Under the service's **Variables**, add `HYDROSCAN_SECRET_KEY` with a long, random secret value. Do not commit the secret to GitHub. Redeploy if Railway does not automatically redeploy after the variable is added.
5. Under **Settings → Networking**, generate a public domain, then open it and test the dashboard, manual analysis, strip image scan, and PDF report download.
6. If you need analysis history to survive redeploys, add a Railway Volume mounted at `/app/instance` for the SQLite database. The generated model, dataset, uploaded images, and reports are also runtime files and may be regenerated/lost unless you configure persistent storage for their directories too.

Railway pricing and trial credits can change; check the current plan shown in your Railway account before relying on a free deployment.

### Important: free-tier storage is not persistent

Both platforms' free tiers use an **ephemeral filesystem** — it resets on
every redeploy (and on Render's free tier, after periods of inactivity too).
That means the SQLite database (`instance/hydroscan.db`), the trained model,
and any uploaded strip photos/reports will be **wiped and regenerated from
scratch** on redeploy. `app.py`'s bootstrap logic handles this automatically
(regenerates the default dataset + retrains + reinits the DB), so the app
always comes back up working — but analysis **history will not survive a
redeploy or restart** on the free tier. If you need real persistence, add a
paid persistent disk (Render) or volume (Railway), or swap SQLite for a
hosted database (e.g., Render Postgres, Supabase).

### Why `--workers 1`

`ml/model_manager.py` holds the trained model in memory. Multiple worker
processes would each load their own separate copy — harmless for this app
since there's no runtime retraining anymore, but `--workers 1 --threads 4`
keeps memory usage low and behavior simple/predictable at demo scale, which
is the right tradeoff here rather than optimizing for concurrent throughput.

---

## Replacing the contents of an existing GitHub repo

If you already have a `hydroscan-ai` (or similarly named) repo and want to
replace its contents entirely with this project:

```bash
# 1. Clone your existing repo (or cd into it if you already have it locally)
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Remove everything currently tracked (keep the .git folder!)
git rm -rf .
# On Windows PowerShell, use instead:
#   Get-ChildItem -Exclude .git | Remove-Item -Recurse -Force

# 3. Copy in the new hydroscan-ai project files
#    (copy every file from the extracted hydroscan-ai folder into this one)

# 4. Stage, commit, and push
git add .
git commit -m "Update HydroScan AI: remove admin panel, harden strip validation"
git push origin main
```

If your default branch is called `master` instead of `main`, use that name
in the last command instead.

**Before pushing publicly:** the `.gitignore` already excludes the SQLite
database, trained model, generated dataset, and uploaded files (these
regenerate automatically), so you won't accidentally commit runtime data.
