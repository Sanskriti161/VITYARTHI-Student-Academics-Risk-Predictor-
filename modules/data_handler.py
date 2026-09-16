"""
data_handler.py
----------------
Module 1: Data Handling.

Responsible for loading the raw dataset, encoding the categorical target
variable, and splitting the data into training and testing sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Mapping between the human-readable risk label and a numeric class,
# used by the model internally and reversed again for display.
RISK_LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
RISK_LABEL_MAP_INVERSE = {v: k for k, v in RISK_LABEL_MAP.items()}

FEATURE_COLUMNS = ["Attendance", "Study_Hours", "Marks"]
TARGET_COLUMN = "Risk_Level"


def load_data(csv_path):
    """
    Load the student dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the dataset.csv file.

    Returns
    -------
    pandas.DataFrame
        The raw dataset.
    """
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Encode the categorical target column and split the dataset into
    training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset containing FEATURE_COLUMNS and TARGET_COLUMN.
    test_size : float
        Proportion of the dataset to use for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : pandas.DataFrame / pandas.Series
        Split feature and target sets.
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(RISK_LABEL_MAP)

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "Dataset contains a Risk_Level value not in "
            f"{list(RISK_LABEL_MAP.keys())}. Please clean dataset.csv."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
