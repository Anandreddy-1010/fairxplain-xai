# Build and execution status

- Official UCI CSV: included and schema-checked (4,424 rows; 36 predictors; 3 target classes; no missing cells or duplicate rows observed).
- Unit tests: 5 passed.
- Executed task runs: multiclass, binary (Dropout vs Enrolled+Graduate), resolved binary (Dropout vs Graduate; Enrolled excluded).
- Executed test accuracy: 75.25%, 87.91%, and 91.18%, respectively, on stratified 20% held-out test partitions (seed 42).
- Selected models: XGBoost, Random Forest, Extra Trees, respectively.
- SHAP global importance: generated for all three tasks.
- Perturbation-based faithfulness sensitivity checks: generated for all three tasks; validation-only and not a causal proof.
- Descriptive fairness group metrics: generated for all three tasks on held-out test predictions.
- Per-class classification reports and confusion matrices: saved.
- Research report: generated from executed test metrics.
- Dashboard source: included; Streamlit could not be installed/launched in the offline build container, so UI runtime test remains outstanding.
- 95% target: NOT achieved in these executions; no fabricated scores are included.
- Dataset scope: end-of-course classification with first- and second-semester information, not admission-time early warning.
- Environment: Python 3.13.5; pandas 2.2.3; NumPy 2.3.5; scikit-learn 1.8.0; XGBoost 3.1.3; LightGBM 4.6.0; CatBoost 1.2.8; SHAP 0.50.0; Plotly 6.5.2.
- Runtime caveat: Streamlit is listed in requirements but unavailable in this isolated container because network package installation failed; dashboard source is syntax-checked but not browser-launched here.
