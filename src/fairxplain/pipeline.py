from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostAdapter(ClassifierMixin, BaseEstimator):
    """Small sklearn-compatible adapter for CatBoost across sklearn tag versions."""
    def __init__(self, iterations=350, depth=6, learning_rate=0.04, loss_function="MultiClass",
                 auto_class_weights=None, random_seed=42, thread_count=4):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        params = dict(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                      loss_function=self.loss_function, verbose=False, allow_writing_files=False,
                      random_seed=self.random_seed, thread_count=self.thread_count)
        if self.auto_class_weights is not None:
            params["auto_class_weights"] = self.auto_class_weights
        self.model_ = CatBoostClassifier(**params)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1).astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)



def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ], remainder="drop", verbose_feature_names_out=False)


def build_estimators(seed: int = 42) -> dict:
    """Fixed, reproducible starting configurations; tune top candidates after baseline."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
            l2_regularization=0.1, max_leaf_nodes=20, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
            objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
            n_jobs=4, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=350, num_leaves=20, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.9, reg_lambda=2.0, verbosity=-1,
            n_jobs=4, random_state=seed),
        "CatBoost": CatBoostAdapter(iterations=350, depth=6, learning_rate=0.04,
            loss_function="MultiClass", random_seed=seed, thread_count=4),
    }


def make_pipeline(model, numeric_cols, categorical_cols, use_smote: bool = False, seed: int = 42):
    if use_smote:
        # SMOTENC is applied to the raw mixed-type feature frame after training-only
        # imputation, before one-hot encoding. This avoids interpolating fractional
        # one-hot category indicators as ordinary SMOTE would.
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTENC
        imputer = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            ("categorical", SimpleImputer(strategy="most_frequent"), categorical_cols),
        ], remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
        encoder = ColumnTransformer([
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ], remainder="drop", verbose_feature_names_out=False)
        sampler = SMOTENC(categorical_features=categorical_cols, random_state=seed, k_neighbors=3)
        return ImbPipeline([
            ("impute", imputer),
            ("sampler", sampler),
            ("preprocess", encoder),
            ("classifier", model),
        ])
    return Pipeline([("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                     ("classifier", model)])
