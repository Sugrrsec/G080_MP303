
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mp303.cleaning import clean_training_data


# Project root
ROOT = Path(__file__).resolve().parents[2]


# Default feature definitions
NUMERIC_FEATURES = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked",
]

CATEGORICAL_FEATURES = [
    "flush",
    "section",
]


def load_config():
    """Load project configuration."""

    config_path = ROOT / "params.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_pipeline(
    n_estimators,
    random_state=42,
    numerical_features=None,
    categorical_features=None,
):
    """Build preprocessing and Random Forest pipeline."""

    numerical_features = (
        numerical_features or NUMERIC_FEATURES
    )

    categorical_features = (
        categorical_features or CATEGORICAL_FEATURES
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return model


def run_training():

    config = load_config()

    # Read configuration.
    train_path = ROOT / config["data"]["train_path"]

    target_column = config["data"]["target"]

    numeric_features = config["features"]["numerical"]

    categorical_features = config["features"]["categorical"]

    random_state = config["model"]["random_state"]

    validation_size = config["model"]["validation_size"]

    candidates = config["model"]["candidates"]

    experiment_name = config["mlflow"]["experiment_name"]

    model_name = config["mlflow"]["registered_model_name"]

    tracking_uri = config["mlflow"]["tracking_uri"]

    # Calculate hash from ORIGINAL training CSV.
    original_bytes = train_path.read_bytes()

    data_sha256 = hashlib.sha256(
        original_bytes
    ).hexdigest()

    print("Original dataset SHA256:", data_sha256)

    # Load and clean data.
    raw_data = pd.read_csv(train_path)

    cleaned_data = clean_training_data(raw_data)

    print("Original rows:", len(raw_data))
    print("Cleaned rows:", len(cleaned_data))

    # Separate features and target.
    feature_columns = (
        numeric_features + categorical_features
    )

    X = cleaned_data[feature_columns].copy()

    y = pd.to_numeric(
        cleaned_data[target_column],
        errors="coerce",
    )

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    # Validation split.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
    )

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))

    # Configure MLflow.
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Experiment tracking.
    results = []

    for candidate in candidates:

        n_estimators = candidate["n_estimators"]

        model = build_pipeline(
            n_estimators=n_estimators,
            random_state=random_state,
            numerical_features=numeric_features,
            categorical_features=categorical_features,
        )

        with mlflow.start_run(
            run_name=f"RandomForest_{n_estimators}"
        ) as run:

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            mae = mean_absolute_error(
                y_val, predictions
            )

            # Log required original dataset fingerprint.
            mlflow.log_param(
                "data_sha256", data_sha256
            )

            mlflow.log_param(
                "model_type", "RandomForestRegressor"
            )

            mlflow.log_params({
                "n_estimators": n_estimators,
                "random_state": random_state,
                "validation_size": validation_size,
                "training_rows": len(X_train),
            })

            # Log MAE metric.
            mlflow.log_metric(
                "mae_validation", float(mae)
            )

            results.append({
                "n_estimators": n_estimators,
                "mae": float(mae),
                "run_id": run.info.run_id,
            })

            print(
                f"Model: {n_estimators} trees | "
                f"Validation MAE: {mae:.4f}"
            )

    # Select model with lowest MAE.
    best_result = min(
        results,
        key=lambda item: item["mae"],
    )

    best_n_estimators = best_result["n_estimators"]

    best_mae = best_result["mae"]

    print("\nBest configuration:", best_n_estimators)

    print("Best validation MAE:", round(best_mae, 4))

    # Retrain selected configuration on ALL cleaned
    # original training rows to create v1.
    final_model = build_pipeline(
        n_estimators=best_n_estimators,
        random_state=random_state,
        numerical_features=numeric_features,
        categorical_features=categorical_features,
    )

    final_model.fit(X, y)

    # Log and register the final v1 model.
    with mlflow.start_run(
        run_name="v1_best_model"
    ) as run:

        mlflow.log_param(
            "data_sha256", data_sha256
        )

        mlflow.log_param(
            "model_type", "RandomForestRegressor"
        )

        mlflow.log_params({
            "n_estimators": best_n_estimators,
            "random_state": random_state,
            "training_rows": len(X),
            "validation_run_id": best_result["run_id"],
        })

        mlflow.log_metric(
            "mae_validation", best_mae
        )

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            registered_model_name=model_name,
            serialization_format="cloudpickle",
        )

        v1_run_id = run.info.run_id

    # Assign champion alias.
    client = MlflowClient()

    versions = client.search_model_versions(
        filter_string=f"name='{model_name}'"
    )

    if not versions:
        raise RuntimeError(
            "Model registration failed."
        )

    version = max(
        versions,
        key=lambda item: int(item.version),
    ).version

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version,
    )

    print("\n========== MLflow Summary ==========")

    print("Experiment:", experiment_name)

    print("Registered model:", model_name)

    print("Registered version:", version)

    print("Champion alias: champion")

    print("V1 run ID:", v1_run_id)

    print("Data SHA256:", data_sha256)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    run_training()
