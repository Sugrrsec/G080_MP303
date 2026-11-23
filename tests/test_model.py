
import numpy as np
import pandas as pd

from mp303.train import build_pipeline


def test_model_returns_valid_predictions():

    X = pd.DataFrame({

        "worker_exp_yr": [2, 5, 8, 10, 12, 15],

        "rain_mm_week": [20, 40, 60, 80, 100, 120],

        "slope_deg": [5, 10, 15, 20, 25, 30],

        "temp_c": [20, np.nan, 25, 28, 30, 32],

        "bush_age_yr": [10, 15, 20, 25, 30, 35],

        "hours_worked": [5, 6, 7, 8, 9, 10],

        "flush": [
            "first",
            "second",
            "monsoon",
            "first",
            "second",
            "monsoon",
        ],

        "section": [
            "Upper Garden",
            "Lower Garden",
            "River Side",
            "Upper Garden",
            "Lower Garden",
            "River Side",
        ],
    })

    y = [15, 20, 25, 30, 35, 40]

    model = build_pipeline(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)

    assert np.isfinite(predictions).all()
