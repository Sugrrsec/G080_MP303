
import csv
from pathlib import Path
from datetime import date

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

# Sort records chronologically
rows.sort(key=lambda r: (r["date"], r["record_id"]))

print("=== STUCK SENSOR INVESTIGATION ===")

for column in NUMERIC_COLUMNS:
    best_run = []
    current_run = []

    for row in rows:
        if current_run and row[column] == current_run[-1][column]:
            current_run.append(row)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = [row]

    if len(current_run) > len(best_run):
        best_run = current_run

    start = date.fromisoformat(best_run[0]["date"])
    end = date.fromisoformat(best_run[-1]["date"])
    duration = (end - start).days

    print(f"\nColumn: {column}")
    print("Repeated value:", best_run[0][column])
    print("Consecutive records:", len(best_run))
    print("Start date:", start)
    print("End date:", end)
    print("Duration:", duration, "days")
