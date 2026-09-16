"""
main.py
--------
Entry point for the Student Academic Risk Predictor application.

Workflow:
    1. Load the dataset (Data Handling module).
    2. Preprocess and split the data (Data Handling module).
    3. Train the DecisionTreeClassifier (Model Engine module).
    4. Evaluate the model and print accuracy (Model Engine module).
    5. Launch the interactive CLI loop for predictions (CLI Interface module).
"""

import os
import sys

from modules.data_handler import load_data, preprocess_data
from modules.model_engine import train_model, evaluate_model
from modules.cli_interface import run_cli

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: dataset.csv not found at {DATASET_PATH}")
        sys.exit(1)

    print("Loading dataset...")
    df = load_data(DATASET_PATH)

    print("Preprocessing data and splitting into train/test sets...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training DecisionTreeClassifier model...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    run_cli(model)


if __name__ == "__main__":
    main()
