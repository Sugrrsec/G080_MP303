
import numpy as np
import pandas as pd


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the original MP303 training dataset.

    Operations:
    1. Remove duplicate rows.
    2. Replace the temperature sentinel 9999 with NaN.
    3. Correct target values typed 10 times too large.
    4. Standardize flush spelling variants.
    """

    required_columns = [
        "temp_c",
        "flush",
        "kg_per_worker",
        "kg_weighed_at_factory",
    ]

    missing = [
        col for col in required_columns if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()

    # 1. Remove exact duplicate records.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Replace impossible temperature sentinel.
    cleaned["temp_c"] = pd.to_numeric(
        cleaned["temp_c"], errors="coerce"
    )

    cleaned["temp_c"] = cleaned["temp_c"].replace(
        9999.0, np.nan
    )

    # 3. Standardize flush spelling variants.
    cleaned["flush"] = (
        cleaned["flush"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    spelling_map = {
        "fir.": "first",
        "sec.": "second",
        "mon.": "monsoon",
    }

    cleaned["flush"] = cleaned["flush"].replace(
        spelling_map
    )

    # 4. Correct target values with an extra zero.
    target = pd.to_numeric(
        cleaned["kg_per_worker"], errors="coerce"
    )

    factory_weight = pd.to_numeric(
        cleaned["kg_weighed_at_factory"], errors="coerce"
    )

    typo_mask = (
        (target >= 100)
        & ((target / 10 - factory_weight).abs() <= 1.5)
    )

    cleaned.loc[typo_mask, "kg_per_worker"] = (
        target[typo_mask] / 10
    )

    return cleaned
