# FairXplain — Experimental Research Report

## Dataset and scope
The study uses the UCI Predict Students' Dropout and Academic Success dataset (ID 697; DOI 10.24432/C5MC89). It contains 4,424 enrolments, 36 original predictors and a three-class outcome. The data comes from one higher-education institution in Portugal. It includes first- and second-semester academic performance; therefore, the principal analysis is end-of-course outcome classification, not admission-time early warning.

## Methodology
The executed compact workflow uses a stratified 65% train / 15% validation / 20% test split (random state 42). Imputation, scaling and one-hot encoding are fitted inside each model pipeline using training data. Five candidate learners (XGBoost, LightGBM, CatBoost, Random Forest and Extra Trees) are compared. Validation accuracy is the primary selection criterion, with macro-F1 also reported. The selected pipeline is refitted on train+validation and evaluated once on the untouched test set. The compact executed run did not apply SMOTE or Optuna tuning; these are not claimed as completed experiment steps.

## Results

### Primary three-class task (Dropout / Enrolled / Graduate)

- Selected model: **XGBoost**
- Test accuracy: **75.2542%**
- Balanced accuracy: 0.6683
- Macro precision: 0.6876
- Macro recall: 0.6683
- Macro F1: 0.6744
- Weighted F1: 0.7426
- Test sample count: 885


### Secondary binary task (Dropout vs Enrolled + Graduate)

- Selected model: **Random Forest**
- Test accuracy: **87.9096%**
- Balanced accuracy: 0.8488
- Macro precision: 0.8688
- Macro recall: 0.8488
- Macro F1: 0.8576
- Weighted F1: 0.8774
- Test sample count: 885


### Resolved-outcome binary task (Dropout vs Graduate; Enrolled excluded)

- Selected model: **Extra Trees**
- Test accuracy: **91.1846%**
- Balanced accuracy: 0.9018
- Macro precision: 0.9122
- Macro recall: 0.9018
- Macro F1: 0.9063
- Weighted F1: 0.9113
- Test sample count: 726


## Task distinctions
The `binary` task groups Enrolled and Graduate as Not dropout. The `resolved_binary` task excludes all Enrolled records and compares Dropout only with Graduate. These are distinct prediction populations and must not be conflated. The three-class task remains the primary task.

## Explainability and faithfulness
SHAP global importance is computed from a sample of validation rows. The perturbation audit is also run on validation data only. Its results measure model reliance under feature permutation; they do not prove causal relationships or fully establish explanation faithfulness.

## Fairness
The saved fairness tables report descriptive performance across groups on the untouched test predictions. These comparisons are not a legal compliance assessment or proof of fairness. Interpret group results with subgroup support, class prevalence and the single-institution data source in mind.

## Published benchmark comparison
A 2026 SOD-FE paper reports 97.78% accuracy for XGBoost on a related resolved-outcome Dropout-vs-Graduate task. FairXplain did not reproduce SOD-FE in this run. The public implementation combines prior train and test partitions and splits the combined data again before its final k-fold evaluation, so the published cross-validation figure is not directly comparable to this project’s untouched holdout estimate. See `docs/published_benchmark_note.md`.


## Accuracy target and research limitations
The 95% threshold was a target, not a guarantee. None of the three independently evaluated tasks in this run achieved 95% test accuracy. Published results are not evidence that this implementation achieved the same score. The dataset is small and comes from one institution; second-semester features make the task end-of-course outcome classification rather than early warning. Do not report binary accuracy as multiclass accuracy.