# FairXplain — Student Outcome Intelligence

FairXplain is a research prototype for student outcome classification with model explanations, descriptive subgroup fairness audits, perturbation-based model-reliance checks, observed-profile contrasts, and a Streamlit dashboard.

## Dataset bundled in this ZIP

The official UCI `data.csv` is included at `data/raw/data.csv` (4,424 rows, 36 predictors, 1 target). The original outcome classes are `Dropout`, `Enrolled`, and `Graduate`. The source is the UCI Machine Learning Repository dataset 697 (DOI: 10.24432/C5MC89), licensed CC BY 4.0. Cite the dataset if you reuse it.

**Scope limitation:** The data contains first- and second-semester academic performance. The prediction task is end-of-course student-outcome classification, not admission-time or early-semester warning. The dataset represents one higher-education institution in Portugal.

## Executed test results

These numbers were produced in this execution environment using a stratified 65% training / 15% validation / 20% held-out test split (seed 42). Five candidates (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees) were compared. The validation-selected model was refit on train+validation and evaluated once on the untouched test split.

| Task | Population | Selected model | Test accuracy | Macro F1 | Test n |
|---|---|---|---:|---:|---:|
| Multiclass | Dropout / Enrolled / Graduate | XGBoost | **75.25%** | 0.6744 | 885 |
| Binary | Dropout vs (Enrolled + Graduate) | Random Forest | **87.91%** | 0.8576 | 885 |
| Resolved binary | Dropout vs Graduate; Enrolled excluded | Extra Trees | **91.18%** | 0.9063 | 726 |

**The 95% target was not achieved in these runs.** Binary and resolved-binary metrics are separate tasks and must not be reported as the primary three-class accuracy. The resolved-binary task excludes 794 currently enrolled records, so it applies to a narrower population.

Detailed metrics, per-class reports, confusion matrices, validation comparisons, fairness tables, SHAP importance tables and faithfulness sensitivity checks are saved under `reports/`, `models/`, `figures/`, and `data/processed/`.

## Windows setup

Open PowerShell in the extracted folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dataset is already included, so no download is required.

## Train models

The default entrypoint uses the tested compact training workflow. `--task both` runs all three task variants listed below:

```powershell
python scripts\train.py --task both
```

Task definitions:
- `multiclass`: Dropout / Enrolled / Graduate (primary task).
- `binary`: Dropout vs Enrolled + Graduate.
- `resolved_binary`: Dropout vs Graduate, excluding Enrolled records.

The executed compact run compares XGBoost, LightGBM, CatBoost, Random Forest and Extra Trees. Selection is based on validation accuracy (macro-F1 is reported as a secondary metric). The final test split is not used for model selection. The compact run does not apply SMOTE or Optuna search; those optional experiments are not claimed as completed.

## Explainability, faithfulness, fairness, and report

```powershell
python scripts\create_explanations.py
python scripts\build_report.py
```

- SHAP global importance is generated from a sample of validation records.
- Individual prediction explanations are available in the dashboard for compatible tree pipelines.
- The perturbation audit measures model reliance when a validation feature is permuted. It is not a causal test or proof of SHAP faithfulness.
- Fairness tables compare subgroup performance descriptively on the untouched test predictions. They are not legal fairness certification.
- Counterfactual profile comparisons use nearest observed training profiles from other outcome classes. They are descriptive contrasts, not causal interventions.

## Launch dashboard

```powershell
python -m streamlit run app.py
```

The Streamlit dashboard provides overview, student prediction, model analytics, SHAP/global explanation views, fairness audit, dataset explorer, and research notes. The UI source is included. Streamlit was not installed in the isolated build environment, so it could not be launched and runtime-tested here; install from `requirements.txt` on Windows before launching.

## Important research integrity notes

1. Do not claim 95% accuracy: the executed held-out test results above are the measured results.
2. Do not describe the UCI task as early-warning prediction because the predictor set includes second-semester performance.
3. Do not conflate the three-class, all-non-dropout binary, and resolved-outcome binary populations.
4. Features such as Gender, Age at enrollment, Nationality, scholarship and tuition status are included among raw predictors in the current baseline and are also subgroup-audit attributes. Any deployment study should compare models with/without these fields and evaluate proxy effects.
5. Feature associations are not causes. This prototype is not suitable for automated high-stakes student decisions.

## Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89. License: CC BY 4.0.

## Tested environment

The model runs and artifact checks were executed with Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, XGBoost 3.1.3, LightGBM 4.6.0, CatBoost 1.2.8, imbalanced-learn 0.14.1, SHAP 0.50.0, and Plotly 6.5.2. Exact core versions are pinned in `requirements.txt` to reduce joblib model-version mismatch risk. Streamlit was unavailable in the build sandbox; it is specified as an installable dependency but the dashboard launch remains to be verified on the target Windows machine.
