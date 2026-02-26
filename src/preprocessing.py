"""
Preprocessing module.

Handles feature selection, missing values, and dataset preparation.
Separates preprocessing logic from model logic.
"""

import pandas as pd


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features and target.

    Args:
        df (pd.DataFrame): Input dataframe with target column

    Returns:
        tuple: (X, y)
    """

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def fill_missing_values(X: pd.DataFrame, reference_df: pd.DataFrame = None):
    """
    Fill missing values using median.

    Args:
        X (pd.DataFrame): Data to fill
        reference_df (pd.DataFrame): Optional reference dataframe for median calculation

    Returns:
        pd.DataFrame: Filled dataframe
    """

    if reference_df is None:
        return X.fillna(X.median())

    return X.fillna(reference_df.median())