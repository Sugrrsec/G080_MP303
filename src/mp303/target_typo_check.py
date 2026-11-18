
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

with TRAIN_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# Sort records by the target value
rows.sort(key=lambda r: float(r["kg_per_worker"]))

values = [float(row["kg_per_worker"]) for row in rows]

print("=== TARGET TYPO INVESTIGATION ===")

print("\nHighest 15 target values:")
for row in rows[-15:]:
    print(row["record_id"], row["kg_per_worker"])

# Find the largest gap between consecutive target values
gaps = [
    (values[i + 1] - values[i], i)
    for i in range(len(values) - 1)
]

largest_gap, gap_index = max(gaps)

lower_value = values[gap_index]
upper_value = values[gap_index + 1]

print("\nLargest gap:", round(largest_gap, 2))
print("Value below gap:", lower_value)
print("Value above gap:", upper_value)

# Investigate values above the large gap
suspects = rows[gap_index + 1:]

print("\n=== POSSIBLE EXTRA-ZERO TARGETS ===")
print("Number of candidate records:", len(suspects))

for row in suspects:
    target = float(row["kg_per_worker"])
    factory = float(row["kg_weighed_at_factory"])

    print(
        row["record_id"],
        "| Target:", target,
        "| Target / 10:", round(target / 10, 2),
        "| Factory weight:", factory,
        "| Difference:", round(abs(target / 10 - factory), 2)
    )
