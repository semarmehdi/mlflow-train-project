import argparse
import time
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


load_dotenv()

# Tracking server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # experiment_name = "california_housing_regressor" # fixé dans le terminal au moment du run
    registered_model_name = "california_housing_regressor"
    alias_name = "challenger"

    # mlflow.set_experiment(experiment_name) # fixé dans le terminal au moment du run
    client = MlflowClient()

    print("Training model...")
    start_time = time.time()

    # Keep autolog basic, but log model manually
    mlflow.sklearn.autolog(log_models=False)

    # ------------------------------------------------------------------
    # Dataset: California Housing
    # ------------------------------------------------------------------
    df = pd.read_csv(
        "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"
    )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # ------------------------------------------------------------------
    # Model pipeline
    # ------------------------------------------------------------------
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    min_samples_split=args.min_samples_split,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_param("dataset", "california_housing")

        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(5)

        # MLflow 3.x: prefer `name=` instead of deprecated `artifact_path=`
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )

        model_version = model_info.registered_model_version
        print(f"[INFO] Model logged as version {model_version}")

        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias_name,
            version=model_version,
        )
        print(f"[INFO] Alias '{alias_name}' now points to version {model_version}")

        # Optional: handy tags for the registry/UI
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="dataset",
            value="california_housing",
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="metric:test_rmse",
            value=f"{rmse:.4f}",
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="metric:test_r2",
            value=f"{r2:.4f}",
        )

        print(f"[INFO] Run ID: {run.info.run_id}")
        print(f"[INFO] Test RMSE: {rmse:.4f}")
        print(f"[INFO] Test R2: {r2:.4f}")

    print("...Done!")
    print(f"--- Total training time: {time.time() - start_time:.2f} seconds")
