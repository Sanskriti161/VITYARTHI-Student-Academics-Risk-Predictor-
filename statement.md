# Problem Statement

## Title
Student Academic Risk Predictor

## Problem Statement
Educational institutions often identify at-risk students too late, after
attendance, study effort, and marks have already declined for an extended
period. There is a need for a lightweight, data-driven tool that can flag a
student's academic risk level early, based on easily available indicators,
so that mentors and faculty can intervene in time.

## Scope
This project is a command-line application that:
- Loads a dataset of historical student records (Attendance %, Study Hours
  per day, Marks obtained, and a labelled Risk Level).
- Trains a Decision Tree classification model on this data.
- Allows a user (faculty/mentor) to enter a new student's Attendance, Study
  Hours, and Marks through the terminal.
- Predicts and displays whether that student is at Low, Medium, or High
  academic risk.

The scope is limited to a single, self-contained CLI tool using a static
CSV dataset. It does not integrate with any live student information
system, and the dataset used is illustrative/synthetic rather than
institution-specific.

## Target Users
- Faculty advisors / mentors who track student performance.
- Academic coordinators who need a quick, explainable early-warning signal.
- Students themselves, for self-assessment of academic risk.

## High-Level Features
1. **Data Handling Module** — loads and preprocesses the dataset, encodes
   the categorical risk label, and splits data into training/testing sets.
2. **Model Engine Module** — trains a `DecisionTreeClassifier` and reports
   test-set accuracy.
3. **CLI Interface Module** — collects user input, validates it, and
   presents a predicted risk level in a readable format, in a loop so
   multiple students can be checked in one session.
