# -*- coding: utf-8 -*-
"""
modelling.py

This script trains an XGBoost model and logs with MLflow.
It's designed to be run as an MLflow Project entry point.
"""

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
def train_model(data_filename, n_estimators, learning_rate, max_depth, model_artifact_name):
    """
    Loads data, trains an XGBoost model, and logs with MLflow.
    Assumes it is running within an active MLflow run context when called by 'mlflow run'.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    data_path = os.path.join(script_dir, data_filename)
    
    print(f"Attempting to load data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print(f"Ensure '{data_filename}' is in the correct location relative to the script.")
        # In MLflow project context, this script is typically at the root of the project directory.
        # So, if data_filename is "heart_preprocessing/processed_heart_data.csv",
        # it looks for MLProject_Heart/heart_preprocessing/processed_heart_data.csv
        return 
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if 'HeartDisease' not in df.columns:
        print("Error: Target column 'HeartDisease' not found in dataset.")
        return

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Log parameters (these will go to the active run started by `mlflow run`)
    mlflow.log_param("data_file", data_filename)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    
    print(f"Training XGBoost with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False, # Recommended for newer XGBoost
        eval_metric='logloss',   # Common for binary classification
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

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"\nTest Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Log model
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path=model_artifact_name # Uses the name passed as parameter
    )
    print(f"Model logged to MLflow artifact path: {model_artifact_name}")

    # Log a simple text artifact example
    model_summary_text = f"Trained XGBoost model.\nParameters: n_estimators={n_estimators}, lr={learning_rate}, depth={max_depth}\nTest Accuracy: {accuracy:.4f}"
    mlflow.log_text(model_summary_text, "model_run_summary.txt")
    print("Model run summary logged as model_run_summary.txt")


if __name__ == '__main__':
    # This block is executed when `python modelling.py ...` is called.
    # `mlflow run .` effectively does this by constructing the command from MLproject.
    
    parser = argparse.ArgumentParser()
    # Default for data_filename should match the location of processed_heart_data.csv
    # If processed_heart_data.csv is directly in MLProject_Heart:
    parser.add_argument("--data_filename", type=str, default="processed_heart_data.csv", 
                        help="Filename of the preprocessed CSV data, expected in MLProject_Heart directory.")
    # If it's in MLProject_Heart/heart_preprocessing/:
    # parser.add_argument("--data_filename", type=str, default="heart_preprocessing/processed_heart_data.csv", 
    #                     help="Path to the preprocessed CSV data file relative to MLProject_Heart directory.")
    
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--model_artifact_name", type=str, default="xgboost-from-script-default") 
    
    args = parser.parse_args()

    # When `mlflow run` executes this script, it has already started a run
    # and set the experiment. We simply call the main logic.
    # The checks for `mlflow.active_run()` and `mlflow.set_experiment()` / `mlflow.start_run()`
    # were causing the conflict in the CI environment.
    # For pure direct execution (`python modelling.py`) outside of `mlflow run`,
    # you would typically wrap the call to train_model in its own `mlflow.start_run` block
    # if you wanted separate tracking for those direct runs.
    # However, for an MLflow Project entry point, we assume the context is managed.

    print(f"Script execution started with parameters: {args}")
    
    train_model(
        data_filename=args.data_filename,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        model_artifact_name=args.model_artifact_name
    )
    print("Script execution finished.")