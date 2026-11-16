from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from .config import AUDIT_COLUMNS


def audit_groups(X_raw: pd.DataFrame, y_true, y_pred, labels: list[str], task: str) -> pd.DataFrame:
    """Descriptive subgroup performance audit; metrics are not a legal fairness verdict."""
    rows = []
    temp = X_raw.copy().reset_index(drop=True)
    temp["__actual"] = np.asarray(y_true, dtype=str)
    temp["__predicted"] = np.asarray(y_pred, dtype=str)
    group_cols = [c for c in AUDIT_COLUMNS if c in temp.columns]
    if "Age at enrollment" in temp.columns:
        temp["Age group"] = pd.cut(pd.to_numeric(temp["Age at enrollment"], errors="coerce"),
                                   bins=[-np.inf, 20, 25, 30, 40, np.inf],
                                   labels=["<=20", "21-25", "26-30", "31-40", "41+"])
        if "Age group" not in group_cols:
            group_cols.append("Age group")
    for col in group_cols:
        for group_value, part in temp.groupby(col, dropna=False, observed=True):
            if len(part) < 20:
                continue
            yt = part["__actual"].to_numpy()
            yp = part["__predicted"].to_numpy()
            predicted_dropout = float(np.mean(yp == "Dropout"))
            row = {
                "attribute": col,
                "group": str(group_value),
                "support": int(len(part)),
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
                "predicted_dropout_rate": predicted_dropout,
            }
            for label in labels:
                mask = yt == label
                row[f"recall_{label}"] = float(np.mean(yp[mask] == label)) if mask.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def fairness_summary(audit: pd.DataFrame) -> dict:
    if audit.empty:
        return {"status": "No groups met the minimum support threshold (20)."}
    out = {"status": "Descriptive subgroup analysis; differences are not causal and are not a legal compliance assessment."}
    for attribute, group in audit.groupby("attribute"):
        out[attribute] = {
            "groups": int(len(group)),
            "accuracy_gap_max_minus_min": float(group["accuracy"].max() - group["accuracy"].min()),
            "macro_f1_gap_max_minus_min": float(group["macro_f1"].max() - group["macro_f1"].min()),
            "predicted_dropout_rate_gap": float(group["predicted_dropout_rate"].max() - group["predicted_dropout_rate"].min()),
        }
    return out
