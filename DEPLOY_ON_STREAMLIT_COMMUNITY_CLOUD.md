# Deploy FairXplain on Streamlit Community Cloud

1. Create a GitHub repository named `fairxplain-xai` (Public is simplest for a shareable public demo).
2. Upload the **contents of this folder** to the repository root. Do not upload this ZIP as a single file.
3. Confirm that `app.py`, `requirements.txt`, `src/`, `models/`, `data/`, `reports/`, and `figures/` are visible at the repository root.
4. Open https://share.streamlit.io and sign in with GitHub.
5. Choose **Create app** → **Yup, I have an app**.
6. Select repository `fairxplain-xai`, branch `main`, and entrypoint `app.py`.
7. Open **Advanced settings** and select Python **3.13** (the checked model artifacts were built in Python 3.13.5).
8. Deploy and check the app logs if the first build fails.

`requirements.txt` pins the checked Python package versions and Streamlit 1.56.0. The Git ignore file intentionally does not ignore model pipelines, dataset files, reports, or figures because the deployed app reads them at runtime.

Do not commit credentials or `.streamlit/secrets.toml`. No secrets file is needed for the current local-file-based prototype.
