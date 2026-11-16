# Dataset note

The UCI dataset contains 4,424 student records, 36 original predictor features and the `Target` field with three values: Dropout, Enrolled and Graduate. It was collected from a higher-education institution in Portugal and includes demographic, socioeconomic, admission, and first/second-semester academic-performance information.

The primary task is end-of-course student-outcome classification. It is not an admission-time early-warning task because second-semester performance fields are present.

Three tasks are kept distinct:
1. `multiclass`: Dropout / Enrolled / Graduate.
2. `binary`: Dropout vs Enrolled+Graduate.
3. `resolved_binary`: Dropout vs Graduate, with Enrolled records excluded. This is a narrower, resolved-outcome population and is not interchangeable with the other tasks.

Dataset citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5MC89. License: CC BY 4.0.
