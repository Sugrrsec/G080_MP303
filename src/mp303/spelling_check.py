
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"

train = pd.read_csv(TRAIN_PATH)
later = pd.read_csv(LATER_PATH)

TEXT_COLUMNS = ["flush", "section"]

print("=== SPELLING VARIANT INVESTIGATION ===")

for column in TEXT_COLUMNS:

    print(f"\n{'=' * 45}")
    print(f"COLUMN: {column}")
    print("=" * 45)

    print("\nTRAIN DATA - value_counts():")
    for value, count in train[column].value_counts(dropna=False).items():
        print(f"{repr(value):25} : {count}")

    print("\nLATER DATA - clean reference spellings:")
    for value, count in later[column].value_counts(dropna=False).items():
        print(f"{repr(value):25} : {count}")

    print("\nTrue categories:", later[column].nunique())
