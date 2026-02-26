"""
Data loading module.

Responsible for loading datasets from disk.
Separates I/O logic from training logic.
"""

import pandas as pd


def load_train_data(path: str) -> pd.DataFrame:
    """
    Load training dataset.

    Args:
        path (str): Path to training parquet file

    Returns:
        pd.DataFrame: Training dataset
    """
    return pd.read_parquet(path)


def load_test_data(path: str) -> pd.DataFrame:
    """
    Load test dataset.

    Args:
        path (str): Path to test parquet file

    Returns:
        pd.DataFrame: Test dataset
    """
    return pd.read_parquet(path)