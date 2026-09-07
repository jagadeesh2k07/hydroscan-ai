"""
blueprints/main.py
====================
Public-facing routes: the dashboard page, and the JSON API used by its
JavaScript front-end to run predictions, scan strip images, fetch history,
and download PDF reports.
"""
import os
import uuid
from datetime import datetime

from flask import (
    Blueprint, render_template, request, jsonify, current_app,
    send_from_directory, abort,
)
from werkzeug.utils import secure_filename

from ml import model_manager
from ml.domain_knowledge import FEATURES, VALID_RANGES
from vision.strip_analyzer import analyze_strip_image
from reports_engine.report_generator import build_report
from database import db

main_bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------- #
# Pages
# ---------------------------------------------------------------------- #
@main_bp.route("/")
def dashboard():
    return render_template("index.html")


@main_bp.route("/history")
def history_page():
    return render_template("history.html")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _validate_params(data):
    errors = []
    params = {}
    for f in FEATURES:
        if f not in data or data[f] in (None, ""):
            errors.append(f"Missing value for '{f}'.")
            continue
        try:
            v = float(data[f])
        except (TypeError, ValueError):
            errors.append(f"'{f}' must be a number.")
            continue
        lo, hi = VALID_RANGES[f]
        if not (lo <= v <= hi):
            errors.append(f"'{f}' must be between {lo} and {hi}.")
            continue
        params[f] = v
    return params, errors


def _allowed_image(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_IMAGE_EXT"]


# ---------------------------------------------------------------------- #
# API: Predict
# ---------------------------------------------------------------------- #
@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    params, errors = _validate_params(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    sample_label = (data.get("sample_label") or "").strip()[:120] or None
    source = data.get("source", "manual")
    strip_image_path = data.get("strip_image_path")

    explainer = model_manager.get_explainer()
    explanation = explainer.explain(params)

    analysis_id = db.insert_analysis(
        params, explanation, source=source,
        sample_label=sample_label, strip_image_path=strip_image_path,
    )

    response = dict(explanation)
    response["params"] = params
    response["analysis_id"] = analysis_id
    response["sample_label"] = sample_label
    response["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    response["ok"] = True
    return jsonify(response)


# ---------------------------------------------------------------------- #
# API: Strip image scan (OpenCV) -> pre-fills the form, does NOT save yet
# ---------------------------------------------------------------------- #
@main_bp.route("/api/strip-scan", methods=["POST"])
def api_strip_scan():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image file uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "Empty filename."}), 400
    if not _allowed_image(file.filename):
        return jsonify({"ok": False, "error": "Unsupported file type. Use JPG, PNG, or WEBP."}), 400

    raw = file.read()
    try:
        result = analyze_strip_image(raw)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception:
        current_app.logger.exception("Strip analysis failed")
        return jsonify({"ok": False, "error": "Could not analyze the strip image. Please try a clearer photo."}), 500

    # Persist the original photo so it can be linked from the analysis record
    os.makedirs(current_app.config["UPLOAD_FOLDER"], exist_ok=True)
    ext = file.filename.rsplit(".", 1)[-1].lower()
    fname = f"strip_{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], fname)
    with open(save_path, "wb") as f:
        f.write(raw)

    result["ok"] = True
    result["strip_image_path"] = f"uploads/{fname}"
    return jsonify(result)


# ---------------------------------------------------------------------- #
# API: History
# ---------------------------------------------------------------------- #
@main_bp.route("/api/history")
def api_history():
    limit = min(int(request.args.get("limit", 20)), 100)
    offset = max(int(request.args.get("offset", 0)), 0)
    prediction = request.args.get("prediction") or None
    rows = db.list_analyses(limit=limit, offset=offset, prediction=prediction)
    return jsonify({"ok": True, "results": rows, "total": db.count_analyses(prediction)})


@main_bp.route("/api/analysis/<int:analysis_id>")
def api_get_analysis(analysis_id):
    row = db.get_analysis(analysis_id)
    if not row:
        abort(404)
    return jsonify({"ok": True, "analysis": row})


# ---------------------------------------------------------------------- #
# API: PDF report download
# ---------------------------------------------------------------------- #
@main_bp.route("/api/report/<int:analysis_id>")
def api_report(analysis_id):
    row = db.get_analysis(analysis_id)
    if not row:
        abort(404)

    os.makedirs(current_app.config["REPORTS_FOLDER"], exist_ok=True)
    fname = f"HydroScan_Report_{analysis_id}.pdf"
    out_path = os.path.join(current_app.config["REPORTS_FOLDER"], fname)

    report_data = {
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "plain_summary": row["plain_summary"],
        "parameter_statuses": row["statuses"],
        "reasons": row["reasons"],
        "contribution_percent": row["contribution"],
        "top_influential_parameter": row["top_feature"],
        "causes": row["causes"],
        "recommendations": row["recommendations"],
        "params": {f: row[f] for f in FEATURES},
        "analysis_id": row["id"],
        "created_at": row["created_at"],
        "sample_label": row["sample_label"],
    }
    build_report(report_data, out_path)
    db.set_report_path(analysis_id, f"reports/{fname}")

    return send_from_directory(current_app.config["REPORTS_FOLDER"], fname,
                                as_attachment=True, download_name=fname)


@main_bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)
