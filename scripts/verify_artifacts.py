"""Load saved models and check one representative inference per task."""
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairxplain.config import MODELS_DIR, DATA_FILE
from fairxplain.data import load_dataset
from fairxplain.features import engineer_features


def main():
    df = load_dataset(DATA_FILE)
    row = df.drop(columns=["Target"]).iloc[[0]].copy()
    for task in ("multiclass", "binary", "resolved_binary"):
        folder = MODELS_DIR / task
        model = joblib.load(folder / "pipeline.joblib")
        encoder = joblib.load(folder / "label_encoder.joblib")
        meta = json.loads((folder / "model_metadata.json").read_text(encoding="utf-8"))
        x = engineer_features(row)[meta["engineered_feature_columns"]]
        pred = int(model.predict(x)[0])
        proba = model.predict_proba(x)[0]
        print(f"{task}: {encoder.inverse_transform([pred])[0]} | probability_sum={proba.sum():.6f} | model={meta['selected_model']}")
        assert abs(float(proba.sum()) - 1.0) < 1e-5
    print("All saved model artifacts loaded and produced valid probability outputs.")

if __name__ == "__main__":
    main()
