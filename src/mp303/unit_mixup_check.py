
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

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

with TRAIN_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

sections = sorted(set(row["section"] for row in rows))

print("=== UNIT MIX-UP INVESTIGATION ===")

for column in NUMERIC_COLUMNS:
    print(f"\nColumn: {column}")

    for section in sections:
        values = [
            float(row[column])
            for row in rows
            if row["section"] == section
            and row[column].strip()
        ]

        print(
            f"{section:15} | "
            f"Min: {min(values):10.2f} | "
            f"Mean: {mean(values):10.2f} | "
            f"Max: {max(values):10.2f}"
        )
