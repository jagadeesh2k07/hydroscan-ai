"""
db.py
======
Thin SQLite data-access layer for HydroScan AI. Uses plain sqlite3 (no ORM)
to keep the stack lightweight, with row_factory set to sqlite3.Row so
records behave like dicts.
"""
import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "instance", "hydroscan.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------- #
# Analyses
# ---------------------------------------------------------------------- #
def insert_analysis(params: dict, explanation: dict, source: str = "manual",
                     sample_label: str = None, strip_image_path: str = None) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO analyses
           (created_at, source, sample_label, ph, chlorine, hardness, nitrate,
            prediction, confidence, top_feature, contribution_json, statuses_json,
            reasons_json, causes_json, recommendations_json, plain_summary,
            strip_image_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            source,
            sample_label,
            float(params["ph"]), float(params["chlorine"]),
            float(params["hardness"]), float(params["nitrate"]),
            explanation["prediction"], explanation["confidence"],
            explanation["top_influential_parameter"],
            json.dumps(explanation["contribution_percent"]),
            json.dumps(explanation["parameter_statuses"]),
            json.dumps(explanation["reasons"]),
            json.dumps(explanation["causes"]),
            json.dumps(explanation["recommendations"]),
            explanation["plain_summary"],
            strip_image_path,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def set_report_path(analysis_id: int, report_path: str):
    conn = get_db()
    conn.execute("UPDATE analyses SET report_path=? WHERE id=?", (report_path, analysis_id))
    conn.commit()
    conn.close()


def get_analysis(analysis_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    conn.close()
    return _row_to_analysis_dict(row) if row else None


def list_analyses(limit: int = 100, offset: int = 0, prediction: str = None):
    conn = get_db()
    if prediction:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE prediction=? ORDER BY id DESC LIMIT ? OFFSET ?",
            (prediction, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    conn.close()
    return [_row_to_analysis_dict(r) for r in rows]


def count_analyses(prediction: str = None):
    conn = get_db()
    if prediction:
        n = conn.execute("SELECT COUNT(*) AS c FROM analyses WHERE prediction=?", (prediction,)).fetchone()["c"]
    else:
        n = conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]
    conn.close()
    return n


def _row_to_analysis_dict(row):
    d = dict(row)
    for jf in ["contribution_json", "statuses_json", "reasons_json", "causes_json", "recommendations_json"]:
        key = jf.replace("_json", "")
        try:
            d[key] = json.loads(d.pop(jf))
        except (TypeError, json.JSONDecodeError):
            d[key] = None
    return d

