import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from fairxplain.data import load_dataset, get_feature_columns
from fairxplain.features import engineer_features, prepare_target
from fairxplain.metrics import evaluate
from fairxplain.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier


def test_load_semicolon_dataset(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"Age at enrollment": [19, 22, 25], "Target": ["Graduate", "Dropout", "Enrolled"]}).to_csv(path, sep=";", index=False)
    df = load_dataset(path)
    assert df.shape == (3, 2)
    assert set(df["Target"]) == {"Graduate", "Dropout", "Enrolled"}


def test_binary_target_is_separate_task():
    y = pd.Series(["Dropout", "Enrolled", "Graduate"])
    out = prepare_target(y, "binary")
    assert list(out) == ["Dropout", "Not dropout", "Not dropout"]


def test_engineered_ratios_and_deltas():
    df = pd.DataFrame({
        "Curricular units 1st sem (enrolled)": [4, 2],
        "Curricular units 1st sem (approved)": [3, 1],
        "Curricular units 1st sem (evaluations)": [4, 2],
        "Curricular units 1st sem (without evaluations)": [0, 1],
        "Curricular units 1st sem (grade)": [15.0, 8.0],
        "Curricular units 2nd sem (enrolled)": [4, 3],
        "Curricular units 2nd sem (approved)": [4, 1],
        "Curricular units 2nd sem (evaluations)": [4, 3],
        "Curricular units 2nd sem (without evaluations)": [0, 1],
        "Curricular units 2nd sem (grade)": [16.0, 9.0],
    })
    out = engineer_features(df)
    assert np.isclose(out.loc[0, "Derived 1st sem approval rate"], 0.75)
    assert np.isclose(out.loc[0, "Derived semester grade change"], 1.0)


def test_metric_calculation():
    result = evaluate([0, 1, 2, 2], [0, 1, 1, 2], [0, 1, 2])
    assert result["accuracy"] == 0.75
    assert 0 <= result["macro_f1"] <= 1


def test_pipeline_fit_smoke():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"score": rng.normal(size=90), "age": rng.integers(18, 40, 90),
                      "Course": rng.choice([1, 2, 3], 90), "Gender": rng.choice([0, 1], 90)})
    y = np.array([0, 1, 2] * 30)
    pipe = make_pipeline(ExtraTreesClassifier(n_estimators=10, random_state=4),
                         ["score", "age"], ["Course", "Gender"])
    pipe.fit(X, y)
    assert len(pipe.predict(X.iloc[:5])) == 5
    assert pipe.predict_proba(X.iloc[:5]).shape == (5, 3)


def test_resolved_binary_target_excludes_enrolled_class():
    y = pd.Series(["Dropout", "Graduate"])
    out = prepare_target(y, "resolved_binary")
    assert list(out) == ["Dropout", "Graduate"]


def test_resolved_binary_rejects_enrolled_records():
    import pytest
    y = pd.Series(["Dropout", "Enrolled", "Graduate"])
    with pytest.raises(ValueError):
        prepare_target(y, "resolved_binary")
