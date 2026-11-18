

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

df = pd.read_csv(TRAIN_PATH)

NUMERIC_COLUMNS = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked",
    "kg_weighed_at_factory",
    "kg_per_worker",
]

print("=== SENTINEL VALUE INVESTIGATION ===")

for column in NUMERIC_COLUMNS:
    print(f"\nColumn: {column}")

    # Count how frequently each value occurs
    counts = df[column].value_counts(dropna=False)

    print(counts.head(5).to_string())

print("\nInvestigation completed.")

