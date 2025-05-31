import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import mlflow
import mlflow.xgboost
import os
import joblib
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

# --- Main Modelling Function ---
def train_model(data_filename="processed_heart_data.csv", n_estimators=100, learning_rate=0.1, max_depth=3, model_artifact_name="xgboost-model"):
    """
    Loads data, trains an XGBoost model, and logs with MLflow.
    Data file is expected in the same directory as this script.
    Assumes it's running within an active MLflow run when called by 'mlflow run'.
    """
    
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    data_path = os.path.join(script_dir, data_filename)
    
    print(f"Attempting to load data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print(f"Ensure '{data_filename}' is in the same directory as modelling.py (e.g., MLProject_Heart/).")
        return # Exit if data not found
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if 'HeartDisease' not in df.columns:
        print("Error: Target column 'HeartDisease' not found in dataset.")
        return

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # The MLflow run is already started by `mlflow run .`
    # We can get the current run's info if needed, but logging will happen to the active run.
    active_run = mlflow.active_run()
    if active_run:
        print(f"Logging to active MLflow Run ID: {active_run.info.run_id}")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    else:
        print("Warning: Not running within an active MLflow run. Metrics/params might not be logged as expected by 'mlflow run'.")
        # This case is more for when you run the script directly without an outer mlflow.start_run()

    mlflow.log_param("data_file", data_filename)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    
    print(f"Training XGBoost with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"\nTest Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path=model_artifact_name
    )
    print(f"Model logged to MLflow artifact path: {model_artifact_name}")

    # Example of logging a simple text artifact (like a model summary)
    model_summary = f"Model: XGBoost\nParameters: {model.get_params()}\nTest Accuracy: {accuracy:.4f}"
    mlflow.log_text(model_summary, "model_summary.txt")
    print("Model summary logged as model_summary.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str, default="processed_heart_data.csv", 
                        help="Filename of the preprocessed CSV data, expected in the same directory as this script (MLProject_Heart/).")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--model_artifact_name", type=str, default="xgboost-ci-model-default") # Consistent default
    
    args = parser.parse_args()

    # When this script is called by `mlflow run .`, an MLflow run is already active.
    # We just call the train_model function, and it will log to that active run.
    # The `mlflow.set_experiment()` and `mlflow.start_run()` below are for when you
    # run `python modelling.py` directly from your terminal for local testing/debugging,
    # *outside* of an `mlflow run` context.

    if mlflow.active_run():
        print("Detected active MLflow run (likely started by 'mlflow run .'). Using it.")
        # The experiment is already set by `mlflow run --experiment-name ...`
        train_model(
            data_filename=args.data_filename,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            model_artifact_name=args.model_artifact_name
        )
    else:
        # This block is for when you run `python modelling.py` directly
        print("No active MLflow run detected. Starting a new one for direct script execution.")
        mlflow.set_experiment("Direct_Script_Test_Experiment") 
        with mlflow.start_run(run_name="Direct_Python_Script_Run_XGBoost") as run:
            print(f"Direct script run started. MLflow Run ID: {run.info.run_id}")
            train_model(
                data_filename=args.data_filename,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                model_artifact_name=args.model_artifact_name
            )