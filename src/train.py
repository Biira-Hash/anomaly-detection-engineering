"""
Training pipeline.

Coordinates data loading, preprocessing, model training, and evaluation.
Acts as the main entry point for training.
"""

from sklearn.model_selection import train_test_split

from data_loader import load_train_data
from preprocessing import split_features_target, fill_missing_values
from model import create_model, train_model, evaluate_model


def run_training_pipeline(train_path: str):
    """
    Run full training pipeline.

    Args:
        train_path (str): Path to training data
    """

    # Step 1: Load data
    print("Loading data...")
    df = load_train_data(train_path)

    # Step 2: Split features and target
    print("Splitting features and target...")
    X, y = split_features_target(df)

    # Step 3: Handle missing values
    print("Filling missing values...")
    X = fill_missing_values(X)

    # Step 4: Train-validation split
    print("Splitting train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 5: Create model
    print("Creating model...")
    model = create_model()

    # Step 6: Train model
    print("Training model...")
    model = train_model(model, X_train, y_train)

    # Step 7: Evaluate model
    print("Evaluating model...")
    f1 = evaluate_model(model, X_val, y_val)

    print(f"Validation F1 Score: {f1:.4f}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run anomaly detection training pipeline")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset")

    args = parser.parse_args()

    run_training_pipeline(args.train_path)