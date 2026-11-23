
import pandas as pd

from mp303.cleaning import clean_training_data


def sample_data():

    return pd.DataFrame({
        "record_id": ["R001"],
        "temp_c": [9999.0],
        "flush": ["fir."],
        "kg_weighed_at_factory": [25.2],
        "kg_per_worker": [252.0],
    })


def test_sentinel_value_replaced():

    df = sample_data()

    cleaned = clean_training_data(df)

    assert pd.isna(cleaned.loc[0, "temp_c"])


def test_flush_spelling_standardized():

    df = sample_data()

    cleaned = clean_training_data(df)

    assert cleaned.loc[0, "flush"] == "first"


def test_target_typo_corrected():

    df = sample_data()

    cleaned = clean_training_data(df)

    assert cleaned.loc[0, "kg_per_worker"] == 25.2


def test_duplicate_rows_removed():

    df = pd.concat(
        [sample_data(), sample_data()],
        ignore_index=True,
    )

    cleaned = clean_training_data(df)

    assert len(cleaned) == 1
