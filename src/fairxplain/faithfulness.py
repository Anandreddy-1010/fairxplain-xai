from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def perturbation_faithfulness_audit(model, X: pd.DataFrame, y_encoded, repeats: int = 5, seed: int = 42):
    """Measure validation accuracy drop when individual fields are permuted.

    This is a perturbation/reliance audit, not a causal test. Higher drop means the
    fitted model's validation predictions rely more on that field under permutation.
    The function never changes model parameters or uses the test partition.
    """
    if X.empty or len(X) < 20:
        return pd.DataFrame(), {"status": "Not enough validation rows for perturbation audit."}
    result = permutation_importance(model, X, y_encoded, scoring="accuracy",
                                    n_repeats=repeats, random_state=seed, n_jobs=1)
    frame = pd.DataFrame({
        "feature": X.columns,
        "accuracy_drop_mean": result.importances_mean,
        "accuracy_drop_std": result.importances_std,
    }).sort_values("accuracy_drop_mean", ascending=False).reset_index(drop=True)
    top_n = min(10, len(frame))
    top = frame.head(top_n)
    remaining = frame.iloc[top_n:]
    rng = np.random.default_rng(seed)
    random = remaining.sample(n=min(top_n, len(remaining)), random_state=seed) if len(remaining) else pd.DataFrame()
    top_mean = float(top["accuracy_drop_mean"].mean()) if len(top) else 0.0
    random_mean = float(random["accuracy_drop_mean"].mean()) if len(random) else 0.0
    summary = {
        "status": "Completed on validation data only",
        "top_feature_count": int(len(top)),
        "mean_top_feature_accuracy_drop": top_mean,
        "mean_random_feature_accuracy_drop": random_mean,
        "top_minus_random_drop": top_mean - random_mean,
        "interpretation": "A perturbation sensitivity check of model reliance; it is not a causal explanation and does not prove that SHAP is faithful.",
    }
    return frame, summary
