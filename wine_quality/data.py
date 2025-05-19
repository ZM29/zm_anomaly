"""
This script focuses on data ingestion and feature engineering
"""

import pandas as pd
from pathlib import Path

def feature_engineer(
    raw: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Creates ratio features to capture relationships between chemical measurements
    These ratios are based on the original paper and other papers written on this
    dataset

    Args:
      raw: Raw DataFrame
    """

    eps = 1e-6
    data = raw.copy()
    data['alc_over_density'] = data['alcohol'] / (data['density'] + eps)
    data['free_tot_so2_ratio'] = data['free sulfur dioxide'] / (data['total sulfur dioxide'] + eps)
    data['sulphates_chlorides'] = data['sulphates'] / (data['chlorides'] + eps)
    data['citric_over_volatile'] = data['citric acid'] / (data['volatile acidity'] + eps)
    data['sugar_over_density'] = data['residual sugar'] / (data['density'] + eps)
    
    return data


def parse_csv(
    csv_path: Path,
    train_data: bool,
    drop_cols: list[str] = [],
    label_col: str = 'quality_bool',
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reads CSV, applies feature engineering and extract labels

    Args:
      csv_path: Path to the CSV file
      train_data: If True, expect 'quality' & engineer features
      drop_cols: List of columns to drop from X
      label_col: Name of the boolean label column
    """
    data = pd.read_csv(csv_path, sep=';')
    y = data.pop(label_col) if label_col in data.columns else None

    if train_data and 'quality' in data.columns:
        data['quality_bool'] = data['quality'] >= 5
        data = data.drop(columns='quality')
        data = feature_engineer(data)

    X = data.drop(columns = drop_cols, errors = 'ignore')
    
    return X, y
