from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
from .config import PROCESSED_DIR


def local_explanation(model, X_one: pd.DataFrame, class_index: int = 0) -> pd.DataFrame | None:
    """Return top local SHAP attributions for one row when a single tree model is selected."""
    if not (hasattr(model, "named_steps") and "preprocess" in model.named_steps and "classifier" in model.named_steps):
        return None
    try:
        import shap
        pre = model.named_steps["preprocess"]
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "model_"):
            classifier = classifier.model_
        transformed = pre.transform(X_one)
        names = list(pre.get_feature_names_out())
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            idx = min(max(int(class_index), 0), len(values)-1)
            vals = np.asarray(values[idx])[0]
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                idx = min(max(int(class_index), 0), arr.shape[2]-1)
                vals = arr[0, :, idx]
            else:
                vals = arr[0]
        raw_values = np.asarray(transformed)[0]
        result = pd.DataFrame({"feature": names, "feature_value_encoded": raw_values,
                               "shap_value": np.asarray(vals).ravel()})
        return result.reindex(result.shap_value.abs().sort_values(ascending=False).index).head(15)
    except Exception:
        return None


def make_explanation_artifacts(task_dir: Path, report_dir: Path, figure_dir: Path, max_rows: int = 80):
    """Create global SHAP summary on held-out validation features when SHAP is usable.

    The prediction model is not tuned from these explanations. SHAP is descriptive.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = task_dir / "pipeline.joblib"
    metadata_path = task_dir / "model_metadata.json"
    validation_path = PROCESSED_DIR / task_dir.name / "validation_features.csv"
    if not (pipeline_path.exists() and metadata_path.exists() and validation_path.exists()):
        return {"status": "Missing model or validation feature artifact; train the model first."}
    model = joblib.load(pipeline_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = pd.read_csv(validation_path)
    y = data.pop("__target") if "__target" in data else None
    X = data.iloc[:max_rows].copy()
    try:
        import shap
        # Tree explainers operate on the transformed feature matrix and avoid explaining
        # the entire preprocessing pipeline as an opaque black box.
        if hasattr(model, "named_steps") and "preprocess" in model.named_steps and "classifier" in model.named_steps:
            transformed = model.named_steps["preprocess"].transform(X)
            classifier = model.named_steps["classifier"]
            if hasattr(classifier, "model_"):
                classifier = classifier.model_
            names = list(model.named_steps["preprocess"].get_feature_names_out())
            explainer = shap.TreeExplainer(classifier)
            values = explainer.shap_values(transformed)
            if isinstance(values, list):
                arr = np.mean([np.abs(v) for v in values], axis=0)
            else:
                arr = np.asarray(values)
                if arr.ndim == 3:
                    arr = np.mean(np.abs(arr), axis=2)
                else:
                    arr = np.abs(arr)
            importance = np.mean(np.abs(arr), axis=0).ravel()
            imp = pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
            imp.to_csv(report_dir / "shap_global_importance.csv", index=False)
            top = imp.head(18).sort_values("mean_abs_shap")
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(top["feature"], top["mean_abs_shap"])
            ax.set_title("Global SHAP importance (mean absolute value)")
            ax.set_xlabel("Mean |SHAP value|")
            fig.tight_layout()
            fig.savefig(figure_dir / "shap_global_importance.png", dpi=160)
            plt.close(fig)
            return {"status": "SHAP TreeExplainer completed", "rows_explained": len(X), "features": len(names)}
        return _permutation_fallback(model, X, y, report_dir, figure_dir, "Selected model is an ensemble or does not expose a direct tree estimator.")
    except Exception as exc:
        return _permutation_fallback(model, X, y, report_dir, figure_dir,
                                     f"SHAP could not be computed ({type(exc).__name__}: {exc}).")


def _permutation_fallback(model, X, y_text, report_dir, figure_dir, reason):
    try:
        if y_text is None:
            return {"status": reason + " No validation labels available for fallback."}
        # Resolve label encoder beside the model artifact using the standard project tree.
        task = report_dir.name
        from .config import MODELS_DIR
        encoder_path = MODELS_DIR / task / "label_encoder.joblib"
        encoder = joblib.load(encoder_path)
        y = encoder.transform(y_text.astype(str))
        result = permutation_importance(model, X, y, scoring="accuracy", n_repeats=3,
                                        random_state=42, n_jobs=1)
        imp = pd.DataFrame({"feature": X.columns, "permutation_importance": result.importances_mean})
        imp = imp.sort_values("permutation_importance", ascending=False)
        imp.to_csv(report_dir / "permutation_importance.csv", index=False)
        top = imp.head(18).sort_values("permutation_importance")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"], top["permutation_importance"])
        ax.set_title("Validation permutation importance")
        ax.set_xlabel("Mean decrease in accuracy")
        fig.tight_layout()
        fig.savefig(figure_dir / "shap_global_importance.png", dpi=160)
        plt.close(fig)
        return {"status": reason + " Used validation permutation importance fallback.", "features": len(imp)}
    except Exception as exc:
        return {"status": f"{reason} Permutation fallback failed: {type(exc).__name__}: {exc}"}
