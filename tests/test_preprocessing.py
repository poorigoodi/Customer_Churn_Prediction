"""Tests for data cleaning and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    build_preprocessor,
    clean_data,
    prepare_features_and_target,
    transform_preview,
    validate_feature_frame,
)


def _sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [1, 34, 2, 45],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "No", "Yes"],
            "InternetService": ["DSL", "DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "No", "Yes"],
            "TechSupport": ["No", "No", "No", "Yes"],
            "StreamingTV": ["No", "No", "Yes", "No"],
            "StreamingMovies": ["No", "No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": ["29.85", "1889.5", "108.15", ""],  # blank -> missing
            "Churn": ["No", "No", "Yes", "No"],
        }
    )


def test_clean_data_removes_id_and_encodes_target():
    cleaned = clean_data(_sample_raw_frame())
    assert "customerID" not in cleaned.columns
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert cleaned["TotalCharges"].isna().sum() == 1


def test_missing_numerical_values_are_imputed():
    X, y = prepare_features_and_target(_sample_raw_frame())
    assert X["TotalCharges"].isna().sum() == 1

    preprocessor = build_preprocessor()
    transformed = transform_preview(preprocessor, X)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_invalid_categorical_values_raise():
    X, _ = prepare_features_and_target(_sample_raw_frame())
    X.loc[0, "Contract"] = "Lifetimely"
    with pytest.raises(ValueError, match="Invalid value"):
        validate_feature_frame(X)


def test_feature_transformation_shape_consistency():
    X, _ = prepare_features_and_target(_sample_raw_frame())
    preprocessor = build_preprocessor()
    transformed = transform_preview(preprocessor, X)

    # One-hot expands categoricals; numerical stays at 3 columns
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] > len(NUMERICAL_FEATURES)
    assert set(FEATURE_COLUMNS) == set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)


def test_invalid_target_label_raises():
    df = _sample_raw_frame()
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Invalid Churn labels"):
        clean_data(df)
