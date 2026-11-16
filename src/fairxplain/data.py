from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import TARGET, CATEGORICAL_COLUMNS, DATA_FILE

EXPECTED_CLASSES = {"Dropout", "Enrolled", "Graduate"}


def load_dataset(path: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load the UCI CSV (semicolon-delimited) and validate its core schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download data.csv from the official UCI page: "
            "https://uci-ics-mlr-prod.aws.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess/files "
            "and place it at data/raw/data.csv."
        )
    # Official UCI file uses semicolon separators. Sniff fallback supports comma CSV.
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found columns: {list(df.columns)}")
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(str).str.strip()
    observed = set(df[TARGET].unique())
    if not observed.issubset(EXPECTED_CLASSES) or len(observed) < 2:
        raise ValueError(f"Unexpected target classes: {sorted(observed)}")
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c != TARGET]
    cats = [c for c in CATEGORICAL_COLUMNS if c in features]
    nums = [c for c in features if c not in cats]
    return nums, cats
