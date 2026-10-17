-- HydroScan AI — SQLite schema
-- Single table: analyses (every prediction run, manual or strip-scan).

CREATE TABLE IF NOT EXISTS analyses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    source              TEXT NOT NULL DEFAULT 'manual',   -- 'manual' | 'strip'
    sample_label        TEXT,                             -- optional user-given name, e.g. "Kitchen tap"
    ph                  REAL NOT NULL,
    chlorine            REAL NOT NULL,
    hardness            REAL NOT NULL,
    nitrate             REAL NOT NULL,
    prediction          TEXT NOT NULL,
    confidence          REAL NOT NULL,
    top_feature         TEXT,
    contribution_json    TEXT,
    statuses_json        TEXT,
    reasons_json          TEXT,
    causes_json            TEXT,
    recommendations_json    TEXT,
    plain_summary       TEXT,
    strip_image_path    TEXT,
    report_path         TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_prediction ON analyses(prediction);
