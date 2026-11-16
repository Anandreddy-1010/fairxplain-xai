"""Local, deterministic evidence lookup based on the UCI dataset feature dictionary.

This is a small retrieval layer over documented field definitions, not an LLM or
causal knowledge base. Returned evidence explains what a field represents only.
"""
from __future__ import annotations

UCI_SOURCE = "https://uci-ics-mlr-prod.aws.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess"

EVIDENCE = [
    {"match": "Curricular units", "topic": "Semester academic activity",
     "definition": "The UCI data dictionary describes counts of curricular units credited, enrolled, evaluated, approved, and without evaluations, plus semester grade averages (0–20).",
     "boundary": "These are semester-level academic records; they do not represent admission-time information."},
    {"match": "Tuition fees up to date", "topic": "Tuition status",
     "definition": "The dataset records whether tuition fees were up to date (1=yes, 0=no).",
     "boundary": "This field is an observed association in this dataset, not proof that tuition status causes an outcome."},
    {"match": "Scholarship holder", "topic": "Scholarship status",
     "definition": "The dataset records scholarship-holder status (1=yes, 0=no).",
     "boundary": "This field may reflect socioeconomic context and should be included in subgroup fairness review."},
    {"match": "Age at enrollment", "topic": "Age at enrollment",
     "definition": "The dataset records the student's age at enrollment.",
     "boundary": "Age is an audit-sensitive attribute; model attribution is not a justification for unequal treatment."},
    {"match": "Admission grade", "topic": "Admission grade",
     "definition": "The dataset describes admission grade on a 0–200 scale.",
     "boundary": "This is a recorded historical score, not an intervention target."},
    {"match": "Previous qualification", "topic": "Previous education",
     "definition": "The data dictionary describes the student's previous qualification category and its grade.",
     "boundary": "The category codes describe educational background; they do not establish causal effects."},
    {"match": "Unemployment rate", "topic": "Economic context",
     "definition": "The dataset includes the unemployment rate, inflation rate, and GDP for the surrounding economic context.",
     "boundary": "These contextual fields are not student-level causal measurements."},
    {"match": "GDP", "topic": "Economic context",
     "definition": "The dataset includes GDP as an economic context variable.",
     "boundary": "This aggregate context should not be interpreted as an individual student's behaviour."},
]


def retrieve_feature_evidence(feature_names: list[str]) -> list[dict]:
    found = []
    seen = set()
    for feature in feature_names:
        for item in EVIDENCE:
            if item["match"].lower() in str(feature).lower() and item["topic"] not in seen:
                found.append({**item, "matched_feature": feature, "source": UCI_SOURCE})
                seen.add(item["topic"])
    return found
