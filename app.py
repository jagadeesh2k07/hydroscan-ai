"""
app.py
=======
HydroScan AI — Explainable Water Quality Analysis System
Flask application entry point.

Run with:
    python app.py
Then open http://localhost:5000
"""
import os
from flask import Flask

from config import Config
from database import db
from ml import model_manager, train_model
from ml.dataset_generator import generate_dataset


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["REPORTS_FOLDER"], exist_ok=True)
    os.makedirs(os.path.dirname(app.config["MODEL_PATH"]), exist_ok=True)

    # ---- First-run bootstrap: dataset + model + database -------------
    db.init_db()

    if not os.path.exists(app.config["DEFAULT_DATASET_PATH"]):
        generate_dataset(1200).to_csv(app.config["DEFAULT_DATASET_PATH"], index=False)

    if not os.path.exists(app.config["MODEL_PATH"]):
        train_model.train(dataset_path=app.config["DEFAULT_DATASET_PATH"],
                           model_path=app.config["MODEL_PATH"])

    model_manager.load(app.config["MODEL_PATH"])

    # ---- Blueprints ----------------------------------------------------
    from blueprints.main import main_bp
    app.register_blueprint(main_bp)

    # ---- Error handlers --------------------------------------------------
    @app.errorhandler(404)
    def not_found(e):
        from flask import jsonify, request
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Not found."}), 404
        from flask import render_template
        return render_template("404.html"), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

