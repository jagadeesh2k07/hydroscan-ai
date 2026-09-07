"""
config.py — central configuration for the HydroScan AI Flask app.
Values can be overridden with environment variables for production
deployments (see README.md).
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("HYDROSCAN_SECRET_KEY", "hydroscan-dev-secret-change-me")

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")

    MODEL_PATH = os.path.join(BASE_DIR, "data", "water_quality_model.pkl")
    DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "water_quality_dataset.csv")

    MAX_CONTENT_LENGTH = 12 * 1024 * 1024  # 12 MB upload limit
    ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "webp"}
