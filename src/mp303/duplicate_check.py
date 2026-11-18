

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

df = pd.read_csv(TRAIN_PATH)

# Count extra copies of completely duplicated rows
duplicate_count = int(df.duplicated().sum())

print("=== DUPLICATE ROW INVESTIGATION ===")
print("Total training records:", len(df))
print("Extra duplicate copies:", duplicate_count)
print("Duplicates present:", duplicate_count > 0)
