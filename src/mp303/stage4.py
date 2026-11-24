
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from mp303.cleaning import clean_training_data
from mp303.train import ROOT, build_pipeline, load_config


def calculate_psi(reference, comparison, bins=10):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    comparison = pd.to_numeric(comparison, errors="coerce").dropna()

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = np.histogram(reference, bins=edges)[0]
    comparison_counts = np.histogram(comparison, bins=edges)[0]

    reference_pct = reference_counts / len(reference)
    comparison_pct = comparison_counts / len(comparison)

    epsilon = 0.0001
    reference_pct = np.clip(reference_pct, epsilon, None)
    comparison_pct = np.clip(comparison_pct, epsilon, None)

    return float(
        np.sum(
            (comparison_pct - reference_pct)
            * np.log(comparison_pct / reference_pct)
        )
    )


def get_features(config):
    return (
        config["features"]["numerical"]
        + config["features"]["categorical"]
    )


def main():
    config = load_config()

    target = config["data"]["target"]
    numeric_features = config["features"]["numerical"]
    categorical_features = config["features"]["categorical"]
    feature_columns = get_features(config)

    model_name = config["mlflow"]["registered_model_name"]
    experiment_name = config["mlflow"]["experiment_name"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    train_path = ROOT / config["data"]["train_path"]
    later_path = ROOT / "data" / "raw" / "later.csv"
    holdout_path = ROOT / "data" / "raw" / "holdout_features.csv"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    v1 = mlflow.sklearn.load_model(
        f"models:/{model_name}/1"
    )

    raw_train = pd.read_csv(train_path)
    raw_later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    cleaned_train = clean_training_data(raw_train)
    cleaned_later = clean_training_data(raw_later)

    print("\n========== PSI ANALYSIS ==========")

    psi_results = {}

    for column in numeric_features:
        psi = calculate_psi(
            cleaned_train[column],
            cleaned_later[column],
            bins=10,
        )
        psi_results[column] = psi
        print(f"{column}: {psi:.6f}")

    most_drifted = max(psi_results, key=psi_results.get)
    print("Most drifted feature:", most_drifted)

    X_later = cleaned_later[feature_columns]
    y_later = pd.to_numeric(
        cleaned_later[target], errors="coerce"
    )

    v1_predictions = v1.predict(X_later)

    group_mae = {}

    print("\n========== V1 MAE BY SECTION ==========")

    for section in sorted(cleaned_later["section"].unique()):
        mask = cleaned_later["section"] == section
        score = mean_absolute_error(
            y_later[mask],
            v1_predictions[mask],
        )
        group_mae[section] = float(score)
        print(f"{section}: {score:.6f}")

    worst_group = max(group_mae, key=group_mae.get)
    print("Worst served group:", worst_group)

    ordered_later = cleaned_later.copy()
    ordered_later["_original_order"] = np.arange(len(ordered_later))
    ordered_later["date"] = pd.to_datetime(
        ordered_later["date"], errors="raise"
    )
    ordered_later = ordered_later.sort_values(
        ["date", "_original_order"], kind="stable"
    ).drop(columns="_original_order").reset_index(drop=True)

    split_index = int(len(ordered_later) * 0.70)

    later_train = ordered_later.iloc[:split_index].copy()
    later_check = ordered_later.iloc[split_index:].copy()

    combined_train = pd.concat(
        [cleaned_train, later_train],
        ignore_index=True,
    )

    X_train = combined_train[feature_columns]
    y_train = pd.to_numeric(
        combined_train[target], errors="coerce"
    )

    X_check = later_check[feature_columns]
    y_check = pd.to_numeric(
        later_check[target], errors="coerce"
    )

    if y_train.isna().any() or y_check.isna().any():
        raise ValueError("Missing target values found.")

    v1_check_predictions = v1.predict(X_check)
    mae_v1 = mean_absolute_error(
        y_check, v1_check_predictions
    )

    n_estimators = 300
    random_state = config["model"]["random_state"]

    v2 = build_pipeline(
        n_estimators=n_estimators,
        random_state=random_state,
        numerical_features=numeric_features,
        categorical_features=categorical_features,
    )

    v2.fit(X_train, y_train)

    v2_predictions = v2.predict(X_check)
    mae_v2 = mean_absolute_error(y_check, v2_predictions)

    improvement = ((mae_v1 - mae_v2) / mae_v1) * 100
    promote = improvement >= 8.0

    print("\n========== V1 VS V2 ==========")
    print("V2 training rows:", len(combined_train))
    print("V2 evaluation rows:", len(later_check))
    print("V1 MAE:", round(mae_v1, 6))
    print("V2 MAE:", round(mae_v2, 6))
    print("Improvement percentage:", round(improvement, 4))
    print("Decision:", "PROMOTE" if promote else "KEEP V1")

    with mlflow.start_run(run_name="v2_stage4") as run:
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "random_state": random_state,
            "original_training_rows": len(cleaned_train),
            "later_training_rows": len(later_train),
            "total_training_rows": len(combined_train),
            "evaluation_rows": len(later_check),
            "later_training_fraction": 0.70,
            "promotion_threshold_percent": 8.0,
        })

        mlflow.log_metrics({
            "mae_v1_check": float(mae_v1),
            "mae_v2_check": float(mae_v2),
            "improvement_percent": float(improvement),
        })

        mlflow.sklearn.log_model(
            sk_model=v2,
            name="model",
            registered_model_name=model_name,
            serialization_format="cloudpickle",
        )

        v2_run_id = run.info.run_id

    versions = client.search_model_versions(
        filter_string=f"name='{model_name}'"
    )

    v2_version = max(
        int(version.version) for version in versions
    )

    if promote:
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=str(v2_version),
        )

    champion = client.get_model_version_by_alias(
        model_name, "champion"
    )

    champion_model = mlflow.sklearn.load_model(
        f"models:/{model_name}@champion"
    )

    holdout_predictions = champion_model.predict(
        holdout[feature_columns]
    )

    predictions_dir = ROOT / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": holdout_predictions,
    })

    output.to_csv(
        predictions_dir / "holdout_predictions.csv",
        index=False,
    )

    findings_path = ROOT / "reports" / "findings.yaml"

    with findings_path.open("r", encoding="utf-8") as file:
        findings = yaml.safe_load(file)

    findings["monitoring"]["psi"] = psi_results
    findings["monitoring"]["most_drifted_feature"] = most_drifted
    findings["monitoring"]["mae_by_group"] = group_mae
    findings["monitoring"]["worst_served_group"] = worst_group
    findings["monitoring"]["mae_v1_on_check"] = float(mae_v1)
    findings["monitoring"]["mae_v2_on_check"] = float(mae_v2)
    findings["monitoring"]["decision"] = (
        "promote" if promote else "keep"
    )


    with findings_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            findings,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\n========== FINAL OUTPUT ==========")
    print("V2 registered version:", v2_version)
    print("V2 run ID:", v2_run_id)
    print("Champion version:", champion.version)
    print("Holdout predictions:", len(output))
    print("Output file:", predictions_dir / "holdout_predictions.csv")
    print("Findings updated:", findings_path)


if __name__ == "__main__":
    main()
