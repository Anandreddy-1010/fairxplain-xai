"""Default compact, reproducible training entry point.

The optional extended search remains in src/fairxplain/train.py. This default script
runs the tested lightweight workflow that writes the same model/report artifacts.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
from train_compact import main

if __name__ == "__main__":
    main()
