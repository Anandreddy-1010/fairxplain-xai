# FairXplain research protocol

## Dataset and task

The primary dataset is **Predict Students' Dropout and Academic Success** from the UCI Machine Learning Repository (dataset 697; DOI: 10.24432/C5MC89). It contains 4,424 records, 36 predictor fields, and a three-class target (`Dropout`, `Enrolled`, `Graduate`). The data is from one higher-education institution in Portugal.

The original dataset includes student demographics, admission/enrollment information, socioeconomic variables and academic performance recorded at the end of the first and second semesters. Therefore, the main experiment is end-of-course student-outcome classification. It must not be described as an admission-time early-warning model.

## Tasks reported separately

1. **Primary multiclass task:** Dropout vs Enrolled vs Graduate.
2. **Secondary binary task:** Dropout vs Not dropout (Enrolled + Graduate).

Accuracy from the binary task must never be represented as the multiclass accuracy.

## Leakage controls

- Split data using stratification before model fitting.
- Fit imputers, scalers and encoders only inside training pipelines.
- SMOTE, when compared, is embedded inside the training pipeline and is never applied to validation or test data.
- Hyperparameter/model selection uses validation data and/or cross-validation restricted to the training partition.
- The final holdout test set is not used to select features, models or thresholds.
- Derived ratios and deltas are deterministic transformations of dataset features; the outcome column is never used to create predictors.

## Evaluation

Report accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, confusion matrix and per-class metrics. Accuracy alone is insufficient for an imbalanced three-class problem.

## Explainability

SHAP values and feature attribution describe patterns used by the fitted model. They do not establish causation or guarantee that changing a feature would change a real student's outcome.

## Fairness

Subgroup comparisons are descriptive and should be interpreted with group sizes, class prevalence and the one-institution sampling context in mind. They are not a legal compliance certification. Report subgroup support and multiple performance measures; do not rely on one parity statistic alone.

## Counterfactual profile contrasts

The application displays nearest observed training profiles from alternative outcome classes. These are examples of observed differences, not causal interventions or prescriptions. Certain protected attributes are omitted from displayed contrast fields.

## Target accuracy

95%+ is an experimental target, not a promised result. Any claimed score must come from the untouched test partition and must be reproduced from saved code, data, dependencies and random seeds. Published results are not evidence that this implementation achieved the same score.

## Additional resolved-outcome binary benchmark

A separate `resolved_binary` task excludes all `Enrolled` rows and classifies `Dropout` versus `Graduate`. It is maintained separately because it changes the population and target definition. It must not be described as Dropout versus all non-dropout students. In the executed run, it achieved 91.18% accuracy on an untouched 20% test set (n=726) with Extra Trees.

A 2026 SOD-FE paper reports higher scores for a similar resolved-outcome task. This project does not claim to reproduce SOD-FE. Its reported cross-validation result is not directly comparable to the present holdout result because the public repository code combines the prior training and test partitions and splits the combined data again before its final k-fold stage. See `published_benchmark_note.md`.
