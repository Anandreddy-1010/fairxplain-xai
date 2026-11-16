from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
    }


def report_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.reset_index(names="class_or_average")


def confusion_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"actual_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
