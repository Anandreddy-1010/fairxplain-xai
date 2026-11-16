from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print("Python:", sys.executable)
print("Project:", ROOT)
for package in ["pandas", "numpy", "sklearn", "xgboost", "lightgbm", "catboost", "imblearn", "shap", "streamlit", "plotly"]:
    print(f"{package:12} {'OK' if importlib.util.find_spec(package) else 'MISSING'}")
print("Dataset:", "FOUND" if (ROOT / "data/raw/data.csv").exists() else "MISSING — add official UCI data.csv to data/raw/")
