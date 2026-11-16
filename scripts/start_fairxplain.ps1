$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    py -3.12 -m venv .venv
    $python = Join-Path $root '.venv\Scripts\python.exe'
}
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
if (-not (Test-Path '.\data\raw\data.csv')) {
    & $python '.\scripts\download_data.py'
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Automatic download did not complete. Download data.csv from the official UCI page and save it to data\raw\data.csv, then rerun this script.' -ForegroundColor Yellow
        exit 2
    }
}
& $python '.\scripts\train.py' --task both --optuna-trials 5
& $python '.\scripts\create_explanations.py'
& $python '.\scripts\build_report.py'
& $python -m streamlit run app.py
