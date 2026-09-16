# Student Academic Risk Predictor

## Overview
A command-line Python application that predicts a student's academic risk
level (**Low / Medium / High**) using **Attendance %**, **Study Hours per
day**, and **Marks**, based on a `DecisionTreeClassifier` trained with
`scikit-learn`.

Built for the CSA2001 – Fundamentals in AI and ML course project
(VITyarthi "Build Your Own Project").

## Features
- Loads and preprocesses a labelled student dataset with `pandas`.
- Encodes the categorical target (`Risk_Level`) and performs a
  train/test split.
- Trains a `DecisionTreeClassifier` and prints test-set accuracy on
  startup.
- Interactive CLI loop: enter a student's Attendance, Study Hours, and
  Marks, get an instant predicted risk level.
- Input validation with clear error messages and a clean exit option.
- Modular, package-based code structure (`modules/`) for maintainability.

## Technologies / Tools Used
- Python 3.8+
- pandas
- scikit-learn
- numpy

## Project Structure
```
VITYARTHI-Student-Risk-Predictor/
├── dataset.csv
├── requirements.txt
├── main.py
├── statement.md
├── README.md
└── modules/
    ├── __init__.py
    ├── data_handler.py
    ├── model_engine.py
    └── cli_interface.py
```

## Steps to Install & Run

1. **Clone / unzip the project**, then move into the folder:
   ```bash
   cd VITYARTHI-Student-Risk-Predictor
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

5. **Follow the on-screen prompts** to enter Attendance (%), Study Hours
   (per day), and Marks (0-100). Type `exit` at any prompt to quit.

## Instructions for Testing
- On every run, `main.py` trains the model fresh and prints the **test
  accuracy** achieved on the held-out 20% split — this is the quickest
  sanity check that the model is learning correctly.
- Manually test edge cases at the CLI prompt, e.g.:
  - High attendance + high study hours + high marks → expect **Low** risk.
  - Low attendance + low study hours + low marks → expect **High** risk.
  - Mid-range values → expect **Medium** risk.
- Try invalid inputs (e.g., letters instead of numbers, out-of-range
  percentages) to confirm the input validation in `cli_interface.py`
  handles them gracefully without crashing.

## Screenshots
<img width="913" height="685" alt="Screenshot 2026-09-17 020416" src="https://github.com/user-attachments/assets/b2e10693-b949-4e3a-b17e-d08218e070d7" />


## Future Enhancements
- Replace the synthetic dataset with real, anonymized institutional data.
- Add more features (assignment submission rate, LMS login frequency).
- Persist the trained model with `joblib` to avoid retraining every run.
- Add a simple web/GUI front-end.
