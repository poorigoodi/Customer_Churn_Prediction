"""Tests for prediction helpers and risk classification."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from sklearn.pipeline import Pipeline

from src.predict import classify_risk, features_to_dataframe, predict_churn
from src.preprocessing import FEATURE_COLUMNS, build_preprocessor
from sklearn.linear_model import LogisticRegression


def _valid_customer() -> Dict[str, Any]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def _tiny_fitted_pipeline() -> Pipeline:
    """Fit a tiny deterministic pipeline for offline prediction tests."""
    rows = [_valid_customer() for _ in range(20)]
    # Flip a few labels via tenure/charges diversity
    for i, row in enumerate(rows):
        row["tenure"] = i
        row["MonthlyCharges"] = 20 + i * 3
        row["TotalCharges"] = row["MonthlyCharges"] * max(i, 1)
        row["Contract"] = "Month-to-month" if i % 2 == 0 else "Two year"

    import pandas as pd

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    X["SeniorCitizen"] = X["SeniorCitizen"].astype(str)
    y = pd.Series([1 if i % 3 == 0 else 0 for i in range(len(rows))])

    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "Low"),
        (0.29, "Low"),
        (0.30, "Medium"),
        (0.69, "Medium"),
        (0.70, "High"),
        (1.0, "High"),
    ],
)
def test_risk_classification(probability: float, expected: str):
    assert classify_risk(probability) == expected


def test_prediction_response_structure():
    model = _tiny_fitted_pipeline()
    result = predict_churn(_valid_customer(), model=model)

    assert set(result.keys()) == {"prediction", "churn_probability", "risk_level"}
    assert result["prediction"] in {"Churn", "No Churn"}
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}


def test_invalid_input_raises():
    bad = _valid_customer()
    bad["InternetService"] = "Satellite"
    with pytest.raises(ValueError):
        features_to_dataframe(bad)


def test_missing_field_raises():
    bad = _valid_customer()
    del bad["tenure"]
    with pytest.raises(ValueError, match="Missing required fields"):
        features_to_dataframe(bad)
