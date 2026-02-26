"""
Model module.

Handles model creation, training, and evaluation.
Keeps model logic separate from data and pipeline logic.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def create_model(random_state: int = 42) -> RandomForestClassifier:
    """
    Create Random Forest model instance.

    Args:
        random_state (int): Random seed

    Returns:
        RandomForestClassifier: model instance
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train model.

    Args:
        model: model instance
        X_train: training features
        y_train: training labels

    Returns:
        trained model
    """

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model using F1 score.

    Args:
        model: trained model
        X_val: validation features
        y_val: validation labels

    Returns:
        float: F1 score
    """

    predictions = model.predict(X_val)
    score = f1_score(y_val, predictions)

    return score