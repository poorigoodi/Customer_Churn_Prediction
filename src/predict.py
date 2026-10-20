"""Load the trained pipeline and generate churn predictions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Union

# Allow both `python -m src.predict` and `python src/predict.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    get_project_root,
    validate_feature_frame,
)

# Illustrative risk thresholds (not empirically optimized for a specific business).
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

ModelInput = Mapping[str, Any]
PathLike = Union[str, Path]


def get_default_model_path() -> Path:
    """Return the default serialized model path."""
    return get_project_root() / "models" / "churn_model.joblib"


def load_model(model_path: PathLike | None = None) -> Pipeline:
    """
    Load the trained preprocessing + classifier pipeline.

    Raises
    ------
    FileNotFoundError
        If the model artifact does not exist.
    """
    path = Path(model_path) if model_path else get_default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. "
            "Run `python -m src.train` to train and serialize the model first."
        )
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError("Loaded artifact is not a scikit-learn Pipeline.")
    return model


def classify_risk(probability: float) -> str:
    """
    Map churn probability to an illustrative risk level.

    Thresholds:
    - Low:    probability < 0.30
    - Medium: 0.30 <= probability < 0.70
    - High:   probability >= 0.70
    """
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    if probability < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def _normalize_input(features: ModelInput) -> Dict[str, Any]:
    """Normalize and lightly coerce a single customer feature mapping."""
    data = dict(features)

    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Coerce numerical fields
    for col in NUMERICAL_FEATURES:
        try:
            value = data[col]
            data[col] = float(value) if value is not None and value != "" else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for '{col}': {data[col]}") from exc

    if "SeniorCitizen" in data:
        data["SeniorCitizen"] = str(int(data["SeniorCitizen"])) if str(
            data["SeniorCitizen"]
        ).isdigit() else str(data["SeniorCitizen"])

    return data


def features_to_dataframe(features: ModelInput | list[ModelInput]) -> pd.DataFrame:
    """Convert one or more customer feature dicts into a model-ready DataFrame."""
    if isinstance(features, Mapping):
        records = [_normalize_input(features)]
    else:
        records = [_normalize_input(item) for item in features]

    df = pd.DataFrame(records)[FEATURE_COLUMNS]
    validate_feature_frame(df, allow_missing_numeric=True)
    return df


def predict_churn(
    features: ModelInput,
    model: Pipeline | None = None,
    model_path: PathLike | None = None,
) -> Dict[str, Any]:
    """
    Generate a churn prediction for a single customer.

    Returns
    -------
    dict with keys: prediction, churn_probability, risk_level
    """
    pipeline = model if model is not None else load_model(model_path)
    X = features_to_dataframe(features)

    proba = float(pipeline.predict_proba(X)[0, 1])
    label = int(pipeline.predict(X)[0])
    prediction = "Churn" if label == 1 else "No Churn"

    return {
        "prediction": prediction,
        "churn_probability": round(proba, 4),
        "risk_level": classify_risk(proba),
    }


def predict_batch(
    feature_list: list[ModelInput],
    model: Pipeline | None = None,
    model_path: PathLike | None = None,
) -> list[Dict[str, Any]]:
    """Generate predictions for multiple customers."""
    pipeline = model if model is not None else load_model(model_path)
    X = features_to_dataframe(feature_list)
    probas = pipeline.predict_proba(X)[:, 1]
    labels = pipeline.predict(X)

    results = []
    for proba, label in zip(probas, labels):
        p = float(proba)
        results.append(
            {
                "prediction": "Churn" if int(label) == 1 else "No Churn",
                "churn_probability": round(p, 4),
                "risk_level": classify_risk(p),
            }
        )
    return results


def _demo_customer() -> Dict[str, Any]:
    """Sample customer used for a quick local prediction demo."""
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


if __name__ == "__main__":
    result = predict_churn(_demo_customer())
    print("Local churn prediction demo")
    print(f"  Prediction:         {result['prediction']}")
    print(f"  Churn probability:  {result['churn_probability']:.4f}")
    print(f"  Risk level:         {result['risk_level']}")
    print(
        "\nRisk bands (illustrative): "
        f"Low < {LOW_RISK_THRESHOLD:.2f}, "
        f"Medium < {HIGH_RISK_THRESHOLD:.2f}, "
        f"High >= {HIGH_RISK_THRESHOLD:.2f}"
    )
