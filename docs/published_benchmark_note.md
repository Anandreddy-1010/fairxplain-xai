# Published SOD-FE benchmark: interpretation and reproduction boundary

A 2026 Scientific Reports article titled *SOD-FE: a supervised outlier detection and feature engineering approach for student dropout prediction* reports high scores on the Portugal student-outcome dataset. Its resolved binary task excludes the `Enrolled` class and compares Dropout with Graduate. The article reports 97.78% accuracy for XGBoost in its Portugal benchmark results.

Reference: Eissa Alzabidi et al. (2026), Scientific Reports 16, Article 25906. DOI: https://doi.org/10.1038/s41598-026-56591-6.
Official implementation repository: https://github.com/AlzabidiEissa/SOD-FE_Student_Dropout.

## Why this project does not claim the published score

The task definition matters: the published binary result excludes Enrolled records. FairXplain reports this as a separate `resolved_binary` task and does not label it as three-class or Dropout-vs-all-outcomes performance.

The published repository's `main.py` creates a temporary combination of its training and test partitions during the SOD stage, then assigns `pipeline.X` / `pipeline.y` from that combined frame and calls its split routine again before final k-fold evaluation. Thus the original holdout test partition is not preserved as the final independent test set in that code path. The reported cross-validation result is therefore not directly comparable to this project's untouched holdout test estimate.

FairXplain's executed resolved-binary run instead uses a stratified 65/15/20 train/validation/test split with seed 42, validation-only model selection, refit on train+validation, and one final evaluation on an untouched test set. It achieved 91.18% test accuracy with Extra Trees (n=726). The 95% target was not reached. This is reported transparently rather than changing the test protocol to match a published number.

This note does not assert that every result in the paper is invalid. It identifies a specific difference in the publicly available implementation's final evaluation flow that limits direct comparability.
