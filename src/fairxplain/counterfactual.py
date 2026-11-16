from __future__ import annotations
import numpy as np
import pandas as pd


def nearest_observed_contrast(query: pd.DataFrame, reference: pd.DataFrame, target_class: str,
                              target_col: str = "Target", limit: int = 3) -> pd.DataFrame:
    """Return nearest observed profiles from a requested outcome class.

    This is a descriptive contrast, not a causal or actionable intervention.
    Sensitive/protected attributes are deliberately omitted from the displayed changes.
    """
    candidates = reference[reference[target_col].astype(str) == str(target_class)].copy()
    if candidates.empty:
        return pd.DataFrame()
    query = query.iloc[0]
    protected = {"Gender", "Nacionality", "Age at enrollment", "Marital status", "International",
                 "Scholarship holder", "Tuition fees up to date", "Displaced",
                 "Educational special needs", "Debtor"}
    usable = [c for c in reference.columns if c != target_col and c not in protected and c in query.index]
    numeric = [c for c in usable if pd.api.types.is_numeric_dtype(reference[c])]
    categorical = [c for c in usable if c not in numeric]
    distances = np.zeros(len(candidates), dtype=float)
    for col in numeric:
        vals = pd.to_numeric(reference[col], errors="coerce")
        scale = float(vals.std()) or 1.0
        q = pd.to_numeric(pd.Series([query[col]]), errors="coerce").iloc[0]
        cand = pd.to_numeric(candidates[col], errors="coerce").fillna(vals.median()).to_numpy()
        distances += ((cand - q) / scale) ** 2
    for col in categorical:
        distances += (candidates[col].astype(str).to_numpy() != str(query[col])).astype(float)
    result = candidates.copy()
    result["distance"] = np.sqrt(distances)
    return result.sort_values("distance").head(limit)
