from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
TARGET = "Target"
DATA_FILE = RAW_DIR / "data.csv"
RANDOM_STATE = 42

# Nominal integer-coded fields from the UCI data dictionary.
CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
]

# Variables reviewed as potentially sensitive/proxy attributes for subgroup auditing.
AUDIT_COLUMNS = ["Gender", "Scholarship holder", "Tuition fees up to date", "Age at enrollment", "Nacionality"]

# Features that are not suitable for an admission-time early-warning claim because
# they contain semester-end performance. The public UCI task itself is end-of-course
# outcome classification, not admission-time prediction.
SEMESTER_FEATURES = [
    c for c in [
        "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    ]
]
