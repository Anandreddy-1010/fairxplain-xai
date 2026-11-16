from __future__ import annotations
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "data.csv"
URLS = [
    # Official UCI dataset static file endpoint.
    "https://archive.ics.uci.edu/static/public/697/data.csv",
]


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in URLS:
        try:
            print(f"Downloading from {url}")
            response = requests.get(url, timeout=40, headers={"User-Agent": "FairXplain research project"})
            response.raise_for_status()
            if len(response.content) < 100_000:
                raise ValueError(f"Download unexpectedly small ({len(response.content)} bytes).")
            temp = DEST.with_suffix(".download")
            temp.write_bytes(response.content)
            # Validate delimiter, target and row/column shape before accepting file.
            import pandas as pd
            df = pd.read_csv(temp, sep=";", encoding="utf-8-sig")
            if len(df.columns) == 1:
                df = pd.read_csv(temp, sep=",", encoding="utf-8-sig")
            if "Target" not in df.columns or len(df) < 4000:
                raise ValueError(f"Unexpected file schema: shape={df.shape}, target_present={'Target' in df.columns}")
            temp.replace(DEST)
            print(f"Dataset saved and validated: {DEST} | shape={df.shape}")
            return
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    print("Automatic download failed:")
    print("\n".join(errors))
    print("\nManual fallback: open the official UCI files page, download data.csv, and save it as:")
    print(DEST)
    print("Official UCI page: https://uci-ics-mlr-prod.aws.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess/files")
    sys.exit(2)


if __name__ == "__main__":
    main()
