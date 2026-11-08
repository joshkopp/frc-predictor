"""Central config. Change YEAR or paths here."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
YEAR = 2026
# Set FRC_DATA_MODE=synthetic to run the whole pipeline on fake data (code testing only).
MODE = os.environ.get("FRC_DATA_MODE", "real")

DATA = ROOT / "data" / ("synthetic" if MODE == "synthetic" else "real")
RAW = DATA / "raw"
VALIDATED = DATA / "validated"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models" / MODE
REPORTS = ROOT / "reports" / MODE

TBA_BASE = "https://www.thebluealliance.com/api/v3"

# Only train/evaluate on official events (offseason rules/fields differ and distort the split)
OFFICIAL_ONLY = True

# Chronological split fractions (by match time, not random)
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2   # remaining 0.2 = test

# Online rating model
RATING_LR = 0.05          # how fast team ratings move after each match
RATING_PRIOR_MATCHES = 4  # shrinkage: ratings count as "full" after ~this many matches
RECENT_N = 5

for p in (RAW, VALIDATED, PROCESSED, MODELS, REPORTS):
    try:                      # may be read-only on a web host; the app never needs these folders
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
