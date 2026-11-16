from pathlib import Path
import json
import sys
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairxplain.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PROCESSED_DIR
from fairxplain.explain import make_explanation_artifacts
from fairxplain.faithfulness import perturbation_faithfulness_audit

if __name__ == "__main__":
    for task in ("multiclass", "binary", "resolved_binary"):
        task_dir = MODELS_DIR / task
        report_dir = REPORTS_DIR / task
        figure_dir = FIGURES_DIR / task
        shap_csv = report_dir / "shap_global_importance.csv"
        if shap_csv.exists():
            result = {"status": "Existing SHAP global importance retained"}
        else:
            result = make_explanation_artifacts(task_dir, report_dir, figure_dir, max_rows=60)
        print(task, "explanation:", result)
        faith_path = report_dir / "faithfulness_perturbation_audit.csv"
        summary_path = report_dir / "faithfulness_summary.json"
        if faith_path.exists() and summary_path.exists():
            print(task, "faithfulness: existing validation-only audit retained")
            continue
        try:
            model = joblib.load(task_dir / "pipeline.joblib")
            data = pd.read_csv(PROCESSED_DIR / task / "validation_features.csv")
            y = data.pop("__target").astype(str)
            encoder = joblib.load(task_dir / "label_encoder.joblib")
            y_encoded = encoder.transform(y)
            audit, summary = perturbation_faithfulness_audit(model, data, y_encoded, repeats=1)
            audit.to_csv(faith_path, index=False)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(task, "faithfulness:", summary)
        except Exception as exc:
            print(task, "faithfulness audit skipped:", type(exc).__name__, exc)
