
import csv
from pathlib import Path
from statistics import median

# Find the project root directory
ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"

# Read training dataset
with TRAIN_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    train_rows = list(csv.DictReader(f))

# Read holdout dataset
with HOLDOUT_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    holdout_reader = csv.DictReader(f)
    holdout_columns = holdout_reader.fieldnames

# Candidate leakage column and target
candidate = "kg_weighed_at_factory"
target = "kg_per_worker"

print("=== DATA LEAKAGE INVESTIGATION ===")

print(f"\nCandidate column: {candidate}")
print(f"Target column: {target}")

print("\nPresent in training data:", candidate in train_rows[0])
print("Present in holdout data:", candidate in holdout_columns)

# Compare candidate against target
differences = [
    abs(float(row[candidate]) - float(row[target]))
    for row in train_rows
]

print("\nTotal records:", len(differences))
print("Median absolute difference:", round(median(differences), 2))
print("Records within 1.5 kg:", sum(d <= 1.5 for d in differences))
print("Maximum absolute difference:", round(max(differences), 2))

print("\nInvestigation completed.")
