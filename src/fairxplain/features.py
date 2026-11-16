from __future__ import annotations
import numpy as np
import pandas as pd


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent ratios/deltas from existing semester aggregates.

    These derived features are still based on semester-level records. They must not
    be described as admission-time or early-course features.
    """
    df = frame.copy()
    eps = 1e-6
    for sem in ("1st", "2nd"):
        enrolled = f"Curricular units {sem} sem (enrolled)"
        approved = f"Curricular units {sem} sem (approved)"
        evals = f"Curricular units {sem} sem (evaluations)"
        without = f"Curricular units {sem} sem (without evaluations)"
        if enrolled in df and approved in df:
            df[f"Derived {sem} sem approval rate"] = df[approved] / (df[enrolled] + eps)
        if enrolled in df and evals in df:
            df[f"Derived {sem} sem evaluation rate"] = df[evals] / (df[enrolled] + eps)
        if enrolled in df and without in df:
            df[f"Derived {sem} sem no-evaluation rate"] = df[without] / (df[enrolled] + eps)
    if "Curricular units 1st sem (grade)" in df and "Curricular units 2nd sem (grade)" in df:
        df["Derived semester grade change"] = (
            df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
        )
    if "Curricular units 1st sem (approved)" in df and "Curricular units 2nd sem (approved)" in df:
        df["Derived approved-units change"] = (
            df["Curricular units 2nd sem (approved)"] - df["Curricular units 1st sem (approved)"]
        )
    if "Curricular units 1st sem (grade)" in df and "Curricular units 2nd sem (grade)" in df:
        df["Derived mean semester grade"] = df[[
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"
        ]].mean(axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def prepare_target(y: pd.Series, task: str) -> pd.Series:
    y = y.astype(str).str.strip()
    if task == "multiclass":
        return y
    if task == "binary":
        # Explicit task: Dropout vs all other outcomes (Enrolled + Graduate).
        return (y == "Dropout").map({True: "Dropout", False: "Not dropout"})
    if task == "resolved_binary":
        # Separate benchmark task: exclude currently Enrolled; classify Dropout vs Graduate.
        if not set(y.unique()).issubset({"Dropout", "Graduate"}):
            raise ValueError("resolved_binary requires the Enrolled class to be excluded before target preparation")
        return y
    raise ValueError("task must be 'multiclass', 'binary', or 'resolved_binary'")
