"""Data loading and preprocessing utilities for churn prediction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Known valid values for categorical columns (from IBM Telco dataset)
CATEGORICAL_VALUE_MAP = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1, "0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def get_default_data_path() -> Path:
    """Return the expected path to the raw Telco churn CSV."""
    return get_project_root() / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(data_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file is missing.
    """
    path = Path(data_path) if data_path else get_default_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download the IBM Telco Customer Churn dataset from Kaggle "
            "(blastchar/Telco Customer Churn) and place the file as:\n"
            "  data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Telco data: coerce types, drop identifiers, encode target.

    Parameters
    ----------
    df:
        Raw dataframe including customerID and Churn columns.
    """
    data = df.copy()

    if ID_COLUMN in data.columns:
        data = data.drop(columns=[ID_COLUMN])

    # TotalCharges can contain blank strings for new customers
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

    valid_labels = {"Yes", "No", 1, 0, "1", "0"}
    invalid_mask = ~data[TARGET_COLUMN].isin(valid_labels)
    if invalid_mask.any():
        bad_values = data.loc[invalid_mask, TARGET_COLUMN].unique().tolist()
        raise ValueError(f"Invalid Churn labels found: {bad_values}")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].map(
        {"Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    )

    # SeniorCitizen is binary; keep as category-friendly string for OneHotEncoder
    if "SeniorCitizen" in data.columns:
        data["SeniorCitizen"] = data["SeniorCitizen"].astype(str)

    return data


def validate_feature_frame(df: pd.DataFrame, *, allow_missing_numeric: bool = True) -> None:
    """
    Validate that a feature dataframe contains required columns and valid categories.

    Parameters
    ----------
    df:
        Feature dataframe (without target).
    allow_missing_numeric:
        If True, NaN numerical values are allowed (imputed later).
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    for col in CATEGORICAL_FEATURES:
        allowed = CATEGORICAL_VALUE_MAP[col]
        series = df[col]
        # Normalize SeniorCitizen for comparison
        if col == "SeniorCitizen":
            series = series.astype(str)
            allowed = {str(v) for v in allowed}
        invalid = ~series.isin(allowed) & series.notna()
        if invalid.any():
            bad = series.loc[invalid].unique().tolist()
            raise ValueError(
                f"Invalid value(s) for '{col}': {bad}. Allowed values: {sorted(allowed, key=str)}"
            )

    if not allow_missing_numeric:
        for col in NUMERICAL_FEATURES:
            if df[col].isna().any():
                raise ValueError(f"Missing numerical values in column '{col}'.")


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer with imputation, scaling, and one-hot encoding.

    Fitted only on training data when used inside a Pipeline / CV fold.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split cleaned dataframe into features X and target y."""
    data = clean_data(df)
    validate_feature_frame(data[FEATURE_COLUMNS], allow_missing_numeric=True)
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].astype(int)
    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split to prevent leakage before fitting."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def transform_preview(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Fit and transform a feature matrix for shape/consistency checks in tests."""
    return preprocessor.fit_transform(X)
