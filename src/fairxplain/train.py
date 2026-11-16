from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier

from .config import DATA_FILE, MODELS_DIR, REPORTS_DIR, PROCESSED_DIR, RANDOM_STATE, AUDIT_COLUMNS
from .data import load_dataset, get_feature_columns
from .features import engineer_features, prepare_target
from .metrics import evaluate, report_frame, confusion_frame
from .pipeline import build_estimators, make_pipeline
from .fairness import audit_groups, fairness_summary

warnings.filterwarnings("ignore", category=UserWarning)


def _json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def _schema_for_inputs(raw_X: pd.DataFrame, raw_feature_cols: list[str], categorical_cols: list[str]) -> dict:
    schema = {}
    for col in raw_feature_cols:
        s = raw_X[col]
        if col in categorical_cols:
            values = sorted(s.dropna().unique().tolist(), key=lambda x: str(x))
            schema[col] = {"kind": "categorical", "values": [int(v) if isinstance(v, (np.integer, int)) else str(v) for v in values[:250]],
                           "default": int(s.mode(dropna=True).iloc[0]) if len(s.mode(dropna=True)) else 0}
        else:
            numeric = pd.to_numeric(s, errors="coerce")
            schema[col] = {"kind": "numeric", "min": float(numeric.min()), "max": float(numeric.max()),
                           "median": float(numeric.median()), "mean": float(numeric.mean())}
    return schema


def _make_voting(name_to_model: dict, selected_names: list[str]):
    estimators = [(f"m{i}", clone(name_to_model[name])) for i, name in enumerate(selected_names)]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)


def _suggest_tuned_model(name: str, trial, task: str):
    from .pipeline import CatBoostAdapter
    if name == "XGBoost":
        from xgboost import XGBClassifier
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 500, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.75, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 8.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            eval_metric="logloss" if task == "binary" else "mlogloss",
            tree_method="hist", n_jobs=4, random_state=RANDOM_STATE,
        )
        params["objective"] = "binary:logistic" if task == "binary" else "multi:softprob"
        return XGBClassifier(**params)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 500, step=100),
            num_leaves=trial.suggest_int("num_leaves", 15, 40),
            max_depth=trial.suggest_categorical("max_depth", [-1, 6, 9]),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.75, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            objective="binary" if task == "binary" else "multiclass",
            verbosity=-1, n_jobs=4, random_state=RANDOM_STATE,
        )
    if name in ("Random Forest", "Extra Trees"):
        cls = __import__("sklearn.ensemble", fromlist=["RandomForestClassifier" if name == "Random Forest" else "ExtraTreesClassifier"])
        estimator_cls = getattr(cls, "RandomForestClassifier" if name == "Random Forest" else "ExtraTreesClassifier")
        return estimator_cls(
            n_estimators=trial.suggest_int("n_estimators", 250, 500, step=125),
            max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.7, 1.0]),
            class_weight="balanced_subsample" if name == "Random Forest" else "balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    if name == "HistGradientBoosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("max_iter", 150, 350, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 40),
            l2_regularization=trial.suggest_float("l2_regularization", 0.01, 5.0, log=True),
            random_state=RANDOM_STATE,
        )
    if name == "CatBoost":
        return CatBoostAdapter(
            iterations=trial.suggest_int("iterations", 200, 500, step=100),
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            loss_function="Logloss" if task == "binary" else "MultiClass",
            random_seed=RANDOM_STATE, thread_count=4,
        )
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=3000, class_weight="balanced",
                                  C=trial.suggest_float("C", 0.05, 10.0, log=True))
    raise ValueError(f"No tuning search space for {name}")


def tune_top_candidates(top_names, task, X_train, y_train, X_val, y_val,
                        numeric_cols, categorical_cols, n_trials, fitted, validation_rows):
    if n_trials <= 0:
        return []
    try:
        import optuna
    except ImportError:
        print("Optuna not installed; skipping hyperparameter tuning.")
        return []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    saved_trials = []
    for model_name in top_names[:2]:
        print(f"Optuna tuning {model_name} ({n_trials} trials; 3-fold training CV) ...", flush=True)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        def objective(trial):
            estimator = _suggest_tuned_model(model_name, trial, task)
            pipe = make_pipeline(estimator, numeric_cols, categorical_cols, use_smote=False)
            scores = cross_val_score(pipe, X_train, y_train, cv=fold, scoring="accuracy", n_jobs=1)
            return float(scores.mean())
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            estimator = _suggest_tuned_model(model_name, optuna.trial.FixedTrial(study.best_params), task)
            pipe = make_pipeline(estimator, numeric_cols, categorical_cols, use_smote=False)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val).astype(int)
            metrics = evaluate(y_val, pred, list(range(len(np.unique(np.concatenate([y_train, y_val]))))))
            tuned_name = f"{model_name} (Optuna)"
            fitted[tuned_name] = pipe
            validation_rows.append({"model": tuned_name, **metrics, "variant": "Optuna tuned"})
            for t in study.trials:
                saved_trials.append({"base_model": model_name, "trial": t.number,
                                     "training_cv_accuracy": t.value, "params": json.dumps(t.params, sort_keys=True)})
            print(f"  best training CV={study.best_value:.4f}; validation accuracy={metrics['accuracy']:.4f}")
        except Exception as exc:
            print(f"  Optuna skipped for {model_name}: {type(exc).__name__}: {exc}")
    return saved_trials


def train_task(df: pd.DataFrame, task: str, data_path: Path, use_smote_variants: bool = True, optuna_trials: int = 5) -> dict:
    (REPORTS_DIR / task).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*68}\nTASK: {task.upper()}\n{'='*68}")
    y_text = prepare_target(df["Target"], task)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    labels = list(label_encoder.classes_)
    X_raw = df.drop(columns=["Target"]).copy()
    raw_feature_cols = list(X_raw.columns)
    X = engineer_features(X_raw)
    numeric_cols, categorical_cols = get_feature_columns(pd.concat([X, y_text.rename("Target")], axis=1))
    # Treat newly engineered ratios/deltas as numerical.
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1875, stratify=y_trainval, random_state=RANDOM_STATE
    )
    print(f"Rows={len(df):,} | train={len(X_train):,} | validation={len(X_val):,} | test={len(X_test):,}")
    print("Classes:", {str(label_encoder.inverse_transform([i])[0]): int((y == i).sum()) for i in range(len(labels))})
    print(f"Input features={len(raw_feature_cols)}; engineered features={X.shape[1]}")

    estimators = build_estimators(RANDOM_STATE)
    validation_rows = []
    fitted: dict[str, object] = {}
    for name, estimator in estimators.items():
        print(f"Training {name} ...", flush=True)
        # XGB needs a task-specific objective for binary classification.
        if name == "XGBoost" and task == "binary":
            estimator.set_params(objective="binary:logistic", eval_metric="logloss")
        if name == "LightGBM" and task == "binary":
            estimator.set_params(objective="binary", class_weight="balanced")
        if name == "CatBoost" and task == "binary":
            estimator.set_params(loss_function="Logloss", auto_class_weights="Balanced")
        if name == "HistGradientBoosting":
            # This estimator does not expose class_weight in all supported sklearn versions.
            pass
        pipe = make_pipeline(estimator, numeric_cols, categorical_cols, use_smote=False)
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val).astype(int)
            metrics = evaluate(y_val, pred, list(range(len(labels))))
            validation_rows.append({"model": name, **metrics, "variant": "baseline"})
            fitted[name] = pipe
            print(f"  validation accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}")
        except Exception as exc:
            print(f"  SKIPPED {name}: {type(exc).__name__}: {exc}")

    if not fitted:
        raise RuntimeError("No candidate model trained successfully. Check dependencies and dataset format.")

    # Run small Optuna searches for the two strongest baseline candidates using only
    # training data and stratified CV. Validation is used to compare tuned candidates.
    baseline_validation = pd.DataFrame(validation_rows)
    top_names = baseline_validation.sort_values(["accuracy", "macro_f1"], ascending=False).head(2)["model"].tolist()
    optuna_rows = tune_top_candidates(top_names, task, X_train, y_train, X_val, y_val,
                                      numeric_cols, categorical_cols, optuna_trials, fitted, validation_rows)
    if optuna_rows:
        pd.DataFrame(optuna_rows).to_csv(REPORTS_DIR / task / "optuna_trials.csv", index=False)

    # Try SMOTE-NC variants for the two strongest baseline candidates; resampling is
    # inside the training pipeline and preserves nominal categories.
    validation_df = pd.DataFrame(validation_rows)
    top_names = baseline_validation.sort_values(["accuracy", "macro_f1"], ascending=False).head(2)["model"].tolist()
    if use_smote_variants:
        for name in top_names:
            print(f"Testing SMOTE variant for {name} ...", flush=True)
            base = clone(estimators[name])
            if name == "XGBoost" and task == "binary":
                base.set_params(objective="binary:logistic", eval_metric="logloss")
            if name == "LightGBM" and task == "binary":
                base.set_params(objective="binary", class_weight=None)
            if name == "CatBoost" and task == "binary":
                base.set_params(loss_function="Logloss", auto_class_weights=None)
            try:
                pipe = make_pipeline(base, numeric_cols, categorical_cols, use_smote=True)
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_val).astype(int)
                metrics = evaluate(y_val, pred, list(range(len(labels))))
                key = f"{name} + SMOTE"
                validation_rows.append({"model": key, **metrics, "variant": "SMOTE"})
                fitted[key] = pipe
                print(f"  validation accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}")
            except Exception as exc:
                print(f"  SMOTE variant skipped for {name}: {type(exc).__name__}: {exc}")

    # Include a soft-voting ensemble of the three leading baseline models.
    base_rank = validation_df.sort_values(["accuracy", "macro_f1"], ascending=False)
    vote_names = [n for n in base_rank["model"].tolist() if n in fitted][:3]
    if len(vote_names) >= 2:
        try:
            print(f"Training soft-voting ensemble: {', '.join(vote_names)} ...", flush=True)
            voting = _make_voting(fitted, vote_names)
            voting.fit(X_train, y_train)
            pred = voting.predict(X_val).astype(int)
            metrics = evaluate(y_val, pred, list(range(len(labels))))
            validation_rows.append({"model": "Soft Voting (top 3)", **metrics, "variant": "ensemble"})
            fitted["Soft Voting (top 3)"] = voting
            print(f"  validation accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}")
        except Exception as exc:
            print(f"  Voting ensemble skipped: {type(exc).__name__}: {exc}")

    # Stacking ensemble with out-of-fold base predictions restricted to training data.
    if len(vote_names) >= 2:
        try:
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            print(f"Training stacking ensemble: {', '.join(vote_names)} ...", flush=True)
            stack_estimators = [(f"base_{i}", clone(fitted[name])) for i, name in enumerate(vote_names)]
            stacking = StackingClassifier(estimators=stack_estimators,
                final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
                cv=3, stack_method="predict_proba", n_jobs=1, passthrough=False)
            stacking.fit(X_train, y_train)
            pred = stacking.predict(X_val).astype(int)
            metrics = evaluate(y_val, pred, list(range(len(labels))))
            validation_rows.append({"model": "Stacking Ensemble (top 3)", **metrics, "variant": "ensemble"})
            fitted["Stacking Ensemble (top 3)"] = stacking
            print(f"  validation accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}")
        except Exception as exc:
            print(f"  Stacking ensemble skipped: {type(exc).__name__}: {exc}")

    validation = pd.DataFrame(validation_rows).sort_values(["accuracy", "macro_f1"], ascending=False).reset_index(drop=True)
    output_dir = MODELS_DIR / task
    report_dir = REPORTS_DIR / task
    process_dir = PROCESSED_DIR / task
    for folder in (output_dir, report_dir, process_dir):
        folder.mkdir(parents=True, exist_ok=True)
    validation.to_csv(report_dir / "validation_model_comparison.csv", index=False)

    # Small, training-only CV stability estimate for the two leading non-ensemble candidates.
    cv_rows = []
    cv_names = [n for n in validation["model"].tolist() if n in fitted and not n.startswith(("Soft Voting", "Stacking Ensemble"))][:2]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name in cv_names:
        print(f"3-fold training CV for {name} ...", flush=True)
        try:
            scores = cross_val_score(clone(fitted[name]), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
            cv_rows.append({"model": name, "cv_accuracy_mean": float(scores.mean()),
                            "cv_accuracy_std": float(scores.std()), "fold_scores": scores.round(4).tolist()})
        except Exception as exc:
            cv_rows.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
    _json_dump(cv_rows, report_dir / "training_cv_stability.json")

    selected_name = str(validation.iloc[0]["model"])
    selected = clone(fitted[selected_name])
    print(f"Selected on validation only: {selected_name}")
    # Refit once on train + validation; final test partition remains untouched until here.
    selected.fit(X_trainval, y_trainval)
    test_pred_enc = selected.predict(X_test).astype(int)
    test_pred = label_encoder.inverse_transform(test_pred_enc)
    y_test_text = label_encoder.inverse_transform(y_test)
    test_metrics = evaluate(y_test, test_pred_enc, list(range(len(labels))))
    test_metrics.update({"task": task, "selected_model": selected_name,
                         "classes": labels, "split_seed": RANDOM_STATE,
                         "test_evaluated_once": True,
                         "test_size": float(len(X_test) / len(X)),
                         "train_size": int(len(X_train)), "validation_size": int(len(X_val)),
                         "final_fit_size": int(len(X_trainval)),
                         "feature_count_raw": int(len(raw_feature_cols)),
                         "feature_count_engineered": int(X.shape[1])})
    _json_dump(test_metrics, report_dir / "final_test_metrics.json")
    report_frame(y_test_text, test_pred, labels).to_csv(report_dir / "classification_report.csv", index=False)
    confusion_frame(y_test_text, test_pred, labels).to_csv(report_dir / "confusion_matrix.csv")
    joblib.dump(selected, output_dir / "pipeline.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    input_schema = _schema_for_inputs(X_raw, raw_feature_cols, categorical_cols)
    _json_dump({
        "task": task, "classes": labels, "selected_model": selected_name,
        "raw_feature_columns": raw_feature_cols,
        "engineered_feature_columns": list(X.columns),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "input_schema": input_schema,
        "audit_columns": [c for c in AUDIT_COLUMNS if c in X_raw.columns],
        "target_definition": "Dropout / Enrolled / Graduate" if task == "multiclass" else "Dropout vs Enrolled+Graduate",
        "prediction_timing_note": "The dataset includes first- and second-semester academic performance. This is end-of-course outcome classification, not admission-time early warning.",
        "feature_engineering_note": "Derived semester ratios and deltas are deterministic transformations of available input fields.",
    }, output_dir / "model_metadata.json")

    # Save the validation features separately for post-hoc explanations. They are not
    # used to alter the selected model after this point.
    val_artifact = X_val.copy()
    val_artifact["__target"] = label_encoder.inverse_transform(y_val)
    val_artifact.to_csv(process_dir / "validation_features.csv", index=False)
    train_reference = X_raw.loc[X_trainval.index].copy()
    train_reference["Target"] = label_encoder.inverse_transform(y_trainval)
    train_reference.to_csv(process_dir / "train_reference.csv", index=False)

    # Descriptive fairness audit on the untouched test predictions.
    test_raw = X_raw.loc[X_test.index].copy()
    audit = audit_groups(test_raw, y_test_text, test_pred, labels, task)
    audit.to_csv(report_dir / "fairness_group_metrics.csv", index=False)
    _json_dump(fairness_summary(audit), report_dir / "fairness_summary.json")

    # Holdout rows are saved only for the final evaluation/audit artefacts.
    holdout = X_test.copy()
    holdout["actual_target"] = y_test_text
    holdout["predicted_target"] = test_pred
    holdout.to_csv(process_dir / "test_predictions.csv", index=False)
    print("\nFINAL TEST METRICS")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
    print(f"Saved model: {output_dir / 'pipeline.joblib'}")
    return {"task": task, "selected_model": selected_name, "metrics": test_metrics,
            "validation_comparison": validation.to_dict(orient="records")}


def main():
    parser = argparse.ArgumentParser(description="Train leakage-aware FairXplain models.")
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--task", choices=["multiclass", "binary", "both"], default="both")
    parser.add_argument("--skip-smote", action="store_true", help="Skip SMOTE-NC comparison variants")
    parser.add_argument("--optuna-trials", type=int, default=5, help="Optuna trials per top baseline model (0 disables tuning)")
    args = parser.parse_args()
    df = load_dataset(args.data)
    print(f"Loaded {len(df):,} rows and {len(df.columns)-1} features from {args.data}")
    summaries = []
    tasks = ["multiclass", "binary"] if args.task == "both" else [args.task]
    for task in tasks:
        summaries.append(train_task(df, task, args.data, use_smote_variants=not args.skip_smote, optuna_trials=args.optuna_trials))
    _json_dump(summaries, REPORTS_DIR / "all_task_summaries.json")
    print("\nTraining complete. Review reports/<task>/final_test_metrics.json.")


if __name__ == "__main__":
    main()
