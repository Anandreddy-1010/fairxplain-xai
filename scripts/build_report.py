from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TASK_TITLES = {
    "multiclass": "Primary three-class task (Dropout / Enrolled / Graduate)",
    "binary": "Secondary binary task (Dropout vs Enrolled + Graduate)",
    "resolved_binary": "Resolved-outcome binary task (Dropout vs Graduate; Enrolled excluded)",
}

def main():
    sections = [
        "# FairXplain — Experimental Research Report",
        "\n## Dataset and scope\nThe study uses the UCI Predict Students' Dropout and Academic Success dataset (ID 697; DOI 10.24432/C5MC89). It contains 4,424 enrolments, 36 original predictors and a three-class outcome. The data comes from one higher-education institution in Portugal. It includes first- and second-semester academic performance; therefore, the principal analysis is end-of-course outcome classification, not admission-time early warning.",
        "\n## Methodology\nThe executed compact workflow uses a stratified 65% train / 15% validation / 20% test split (random state 42). Imputation, scaling and one-hot encoding are fitted inside each model pipeline using training data. Five candidate learners (XGBoost, LightGBM, CatBoost, Random Forest and Extra Trees) are compared. Validation accuracy is the primary selection criterion, with macro-F1 also reported. The selected pipeline is refitted on train+validation and evaluated once on the untouched test set. The compact executed run did not apply SMOTE or Optuna tuning; these are not claimed as completed experiment steps.",
        "\n## Results",
    ]
    for task, title in TASK_TITLES.items():
        path = REPORTS / task / "final_test_metrics.json"
        if not path.exists():
            sections.append(f"\n### {title}\nNot executed.\n")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        sections.append(
            f"\n### {title}\n\n- Selected model: **{m['selected_model']}**\n"
            f"- Test accuracy: **{m['accuracy']:.4%}**\n"
            f"- Balanced accuracy: {m['balanced_accuracy']:.4f}\n"
            f"- Macro precision: {m['macro_precision']:.4f}\n"
            f"- Macro recall: {m['macro_recall']:.4f}\n"
            f"- Macro F1: {m['macro_f1']:.4f}\n"
            f"- Weighted F1: {m['weighted_f1']:.4f}\n"
            f"- Test sample count: {m['n_test']}\n"
        )
    sections.append("\n## Task distinctions\nThe `binary` task groups Enrolled and Graduate as Not dropout. The `resolved_binary` task excludes all Enrolled records and compares Dropout only with Graduate. These are distinct prediction populations and must not be conflated. The three-class task remains the primary task.")
    sections.append("\n## Explainability and faithfulness\nSHAP global importance is computed from a sample of validation rows. The perturbation audit is also run on validation data only. Its results measure model reliance under feature permutation; they do not prove causal relationships or fully establish explanation faithfulness.")
    sections.append("\n## Fairness\nThe saved fairness tables report descriptive performance across groups on the untouched test predictions. These comparisons are not a legal compliance assessment or proof of fairness. Interpret group results with subgroup support, class prevalence and the single-institution data source in mind.")
    sections.append("\n## Published benchmark comparison\nA 2026 SOD-FE paper reports 97.78% accuracy for XGBoost on a related resolved-outcome Dropout-vs-Graduate task. FairXplain did not reproduce SOD-FE in this run. The public implementation combines prior train and test partitions and splits the combined data again before its final k-fold evaluation, so the published cross-validation figure is not directly comparable to this project’s untouched holdout estimate. See `docs/published_benchmark_note.md`.\n")
    sections.append("\n## Accuracy target and research limitations\nThe 95% threshold was a target, not a guarantee. None of the three independently evaluated tasks in this run achieved 95% test accuracy. Published results are not evidence that this implementation achieved the same score. The dataset is small and comes from one institution; second-semester features make the task end-of-course outcome classification rather than early warning. Do not report binary accuracy as multiclass accuracy.")
    out = REPORTS / "FairXplain_Research_Report.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written: {out}")

if __name__ == "__main__":
    main()
