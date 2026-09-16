"""
cli_interface.py
------------------
Module 3: CLI Interface.

Responsible for collecting and validating user input from the terminal,
formatting it into a DataFrame matching the model's expected feature
columns, and displaying the predicted risk level in a readable way.
"""

import pandas as pd

from modules.data_handler import FEATURE_COLUMNS, RISK_LABEL_MAP_INVERSE

EXIT_COMMANDS = {"exit", "quit", "q"}


def _prompt_float(prompt_text, min_value, max_value):
    """
    Prompt the user for a numeric value within a valid range, looping
    until valid input (or an exit command) is given.

    Returns
    -------
    float or None
        The parsed value, or None if the user chose to exit.
    """
    while True:
        raw = input(prompt_text).strip()
        if raw.lower() in EXIT_COMMANDS:
            return None
        try:
            value = float(raw)
        except ValueError:
            print(f"  Invalid input. Please enter a number (or 'exit' to quit).")
            continue
        if not (min_value <= value <= max_value):
            print(f"  Value must be between {min_value} and {max_value}. Try again.")
            continue
        return value


def get_user_input():
    """
    Collect Attendance, Study_Hours, and Marks from the user via the
    terminal, with validation.

    Returns
    -------
    dict or None
        Dictionary of collected values, or None if the user wants to exit.
    """
    print("\nEnter student details (type 'exit' at any point to quit):")

    attendance = _prompt_float("  Attendance (0-100 %): ", 0, 100)
    if attendance is None:
        return None

    study_hours = _prompt_float("  Study Hours per day (0-24): ", 0, 24)
    if study_hours is None:
        return None

    marks = _prompt_float("  Marks obtained (0-100): ", 0, 100)
    if marks is None:
        return None

    return {"Attendance": attendance, "Study_Hours": study_hours, "Marks": marks}


def predict_risk(model, student_data):
    """
    Predict the academic risk level for a single student.

    Parameters
    ----------
    model : sklearn.tree.DecisionTreeClassifier
        Trained model.
    student_data : dict
        Dictionary with keys matching FEATURE_COLUMNS.

    Returns
    -------
    str
        Human-readable risk level ("Low", "Medium", or "High").
    """
    input_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
    predicted_class = model.predict(input_df)[0]
    return RISK_LABEL_MAP_INVERSE[predicted_class]


def run_cli(model):
    """
    Run the interactive command-line loop: repeatedly ask for a student's
    details and display the predicted risk level, until the user exits.

    Parameters
    ----------
    model : sklearn.tree.DecisionTreeClassifier
        Trained model.
    """
    print("=" * 50)
    print(" STUDENT ACADEMIC RISK PREDICTOR")
    print("=" * 50)

    while True:
        student_data = get_user_input()
        if student_data is None:
            print("\nExiting. Goodbye!")
            break

        risk_level = predict_risk(model, student_data)
        print("\n  ---------------------------------")
        print(f"  Predicted Risk Level: {risk_level.upper()}")
        print("  ---------------------------------")
