$ErrorActionPreference = 'Stop'
Write-Host 'Setting up FairXplain...' -ForegroundColor Cyan
if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host 'Setup complete. Download data, train, then run Streamlit.' -ForegroundColor Green
