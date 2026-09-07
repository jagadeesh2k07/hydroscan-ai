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
  (resolution, blur, strip-shape, pad-color uniformity), samples the 4 pad
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
- **History** — browse & filter every past analysis

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
│   └── main.py                 # Routes: dashboard, predict, strip-scan, reports, history
├── ml/
│   ├── domain_knowledge.py       # Safety thresholds, reasons, causes, recommendations
│   ├── dataset_generator.py       # Synthetic 1200-record dataset generator
│   ├── train_model.py              # RandomForest training + metrics
│   ├── explainability.py            # SHAP-based Explainer (the XAI core)
│   └── model_manager.py              # In-memory model singleton
├── vision/
│   └── strip_analyzer.py           # OpenCV strip detection, validation + color calibration
├── reports_engine/
│   └── report_generator.py          # ReportLab PDF report builder
├── database/
│   ├── schema.sql                    # analyses table
│   └── db.py                          # Data access layer
├── templates/                        # Jinja2 templates
├── static/
│   ├── css/style.css                  # Design system + all component styles
│   ├── js/                             # dashboard.js, history.js, main.js
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

`vision/strip_analyzer.py` uses OpenCV to:
1. **Validate the image** — decode it, check minimum resolution, and check
   sharpness (Laplacian variance) to reject blurry or blank photos outright.
2. **Locate the strip** — threshold + contour detection to find an elongated,
   strip-shaped region (tries both light-on-dark and dark-on-light polarity).
   If nothing confidently strip-shaped is found, **the scan is rejected**
   with a clear message — it no longer falls back to guessing on the whole
   photo the way an earlier version did.
3. **Sample and validate the 4 pads** — rotate/crop to a straight band, sample
   4 evenly spaced pad regions, and check each region's color uniformity. A
   real reagent pad is a fairly solid color; a noisy/high-variance region is
   more likely background clutter than an actual pad, and lowers confidence.
4. **Match colors to values** — convert each pad's average color to Lab space
   and match it against a calibration table via inverse-distance-weighted
   interpolation between the two nearest anchor colors.
5. **Score and gate** — combine strip-shape confidence and pad-uniformity
   confidence into one overall score (0–100%). Scans below ~35% confidence
   are rejected outright rather than auto-filling numbers nobody should
   trust; scans that pass show their confidence score in the UI so you know
   how much to trust the auto-filled values.

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

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**. Railway detects
   the `Procfile` and `requirements.txt` automatically — no extra config needed.
3. Add the same `HYDROSCAN_SECRET_KEY` environment variable under **Variables**.
4. Note: Railway's free trial credit runs out quickly for an actively-used
   app — expect to need the Hobby plan (~$5/month) for anything beyond a
   quick demo. Render's free tier is free indefinitely (with cold starts).

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
