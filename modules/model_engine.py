"""
model_engine.py
-----------------
Module 2: Model Engine.

Responsible for training the DecisionTreeClassifier and evaluating its
performance on the held-out test set.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_model(X_train, y_train, random_state=42, max_depth=5):
    """
    Train a DecisionTreeClassifier on the training data.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / pandas.Series
        Training features and target.
    random_state : int
        Seed for reproducibility.
    max_depth : int
        Maximum depth of the tree (helps avoid overfitting on a small
        synthetic dataset).

    Returns
    -------
    sklearn.tree.DecisionTreeClassifier
        The trained model.
    """
    model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set and print its accuracy.

    Parameters
    ----------
    model : sklearn.tree.DecisionTreeClassifier
        Trained model.
    X_test, y_test : pandas.DataFrame / pandas.Series
        Test features and target.

    Returns
    -------
    float
        Accuracy score between 0 and 1.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[Model Engine] Test-set accuracy: {accuracy * 100:.2f}%")
    return accuracy
