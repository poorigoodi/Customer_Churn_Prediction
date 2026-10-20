"""Local Streamlit dashboard for Customer Churn Prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_churn
from src.preprocessing import CATEGORICAL_VALUE_MAP, get_project_root

ROOT = get_project_root()
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
EXCEL_PATH = REPORTS / "model_results.xlsx"
JSON_PATH = REPORTS / "evaluation_report.json"


st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource
def get_cached_model():
    """Load the trained pipeline once per session."""
    return load_model()


def _show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing figure: `{path.name}`. Run `python -m src.train` first.")


def page_overview() -> None:
    st.title("Customer Churn Prediction Dashboard")
    st.write(
        "Local dashboard for model evaluation, confusion matrix, feature importance, "
        "SHAP explainability, and single-customer prediction."
    )

    if JSON_PATH.exists():
        report = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        best = report.get("best_model", "N/A")
        metrics = report.get("test_metrics_best_model", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Model", best)
        c2.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
        c3.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
        c4.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
        c5.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.4f}")
    else:
        st.warning("No evaluation report found. Train the model with `python -m src.train`.")

    st.subheader("Excel Results")
    if EXCEL_PATH.exists():
        st.success(f"Results workbook: `{EXCEL_PATH}`")
        st.download_button(
            "Download model_results.xlsx",
            data=EXCEL_PATH.read_bytes(),
            file_name="model_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Excel file not found yet. It is created automatically after training.")


def page_model_evaluation() -> None:
    st.header("Model Evaluation")

    if EXCEL_PATH.exists():
        cv_df = pd.read_excel(EXCEL_PATH, sheet_name="CV_Comparison")
        test_df = pd.read_excel(EXCEL_PATH, sheet_name="Test_Comparison")
        best_df = pd.read_excel(EXCEL_PATH, sheet_name="Model_Evaluation")

        st.subheader("Best Model (Test Set)")
        st.dataframe(best_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cross-Validation Comparison")
            st.dataframe(cv_df, use_container_width=True)
        with col2:
            st.subheader("Test-Set Comparison")
            st.dataframe(test_df, use_container_width=True)
    else:
        st.warning("Run training to generate Excel evaluation sheets.")

    _show_image_if_exists(FIGURES / "model_comparison.png", "Model Comparison (CV ROC-AUC)")
    _show_image_if_exists(FIGURES / "roc_curve.png", "ROC Curve")


def page_confusion_matrix() -> None:
    st.header("Confusion Matrix")
    if EXCEL_PATH.exists():
        cm_df = pd.read_excel(EXCEL_PATH, sheet_name="Confusion_Matrix", index_col=0)
        st.dataframe(cm_df, use_container_width=True)
    _show_image_if_exists(FIGURES / "confusion_matrix.png", "Confusion Matrix")
    st.markdown(
        """
**How to read it**
- **False Positive:** predicted Churn, customer stayed → extra retention cost  
- **False Negative:** predicted No Churn, customer left → missed intervention  
"""
    )


def page_feature_importance() -> None:
    st.header("Feature Importance")
    if EXCEL_PATH.exists():
        fi_df = pd.read_excel(EXCEL_PATH, sheet_name="Feature_Importance")
        st.dataframe(fi_df, use_container_width=True)
    _show_image_if_exists(FIGURES / "feature_importance.png", "Top Feature Importance")


def page_shap() -> None:
    st.header("SHAP Explainability")
    st.write(
        "Mean absolute SHAP values summarize global feature impact on churn probability "
        "for a held-out sample."
    )
    if EXCEL_PATH.exists():
        shap_df = pd.read_excel(EXCEL_PATH, sheet_name="SHAP_Explainability")
        st.dataframe(shap_df, use_container_width=True)
    _show_image_if_exists(FIGURES / "shap_summary.png", "SHAP Summary (bar)")
    _show_image_if_exists(FIGURES / "shap_beeswarm.png", "SHAP Beeswarm")


def page_predict() -> None:
    st.header("Local Prediction")
    st.write("Enter customer attributes and get churn prediction locally.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("gender", sorted(CATEGORICAL_VALUE_MAP["gender"]))
            senior = st.selectbox("SeniorCitizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone = st.selectbox("PhoneService", ["Yes", "No"])
            multiple = st.selectbox(
                "MultipleLines",
                sorted(CATEGORICAL_VALUE_MAP["MultipleLines"], key=str),
            )
        with c2:
            internet = st.selectbox(
                "InternetService",
                sorted(CATEGORICAL_VALUE_MAP["InternetService"]),
            )
            online_sec = st.selectbox(
                "OnlineSecurity",
                sorted(CATEGORICAL_VALUE_MAP["OnlineSecurity"]),
            )
            online_bak = st.selectbox(
                "OnlineBackup",
                sorted(CATEGORICAL_VALUE_MAP["OnlineBackup"]),
            )
            device = st.selectbox(
                "DeviceProtection",
                sorted(CATEGORICAL_VALUE_MAP["DeviceProtection"]),
            )
            tech = st.selectbox(
                "TechSupport",
                sorted(CATEGORICAL_VALUE_MAP["TechSupport"]),
            )
            stream_tv = st.selectbox(
                "StreamingTV",
                sorted(CATEGORICAL_VALUE_MAP["StreamingTV"]),
            )
        with c3:
            stream_movies = st.selectbox(
                "StreamingMovies",
                sorted(CATEGORICAL_VALUE_MAP["StreamingMovies"]),
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
            payment = st.selectbox(
                "PaymentMethod",
                sorted(CATEGORICAL_VALUE_MAP["PaymentMethod"]),
            )
            tenure = st.number_input("tenure (months)", min_value=0, value=1, step=1)
            monthly = st.number_input("MonthlyCharges", min_value=0.0, value=29.85)
            total = st.number_input("TotalCharges", min_value=0.0, value=29.85)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }
        try:
            model = get_cached_model()
            result = predict_churn(payload, model=model)
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", result["prediction"])
            m2.metric("Churn Probability", f"{result['churn_probability']:.4f}")
            m3.metric("Risk Level", result["risk_level"])
            st.json(result)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover
            st.error(f"Prediction failed: {exc}")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Model Evaluation",
            "Confusion Matrix",
            "Feature Importance",
            "SHAP Explainability",
            "Predict",
        ],
    )

    if page == "Overview":
        page_overview()
    elif page == "Model Evaluation":
        page_model_evaluation()
    elif page == "Confusion Matrix":
        page_confusion_matrix()
    elif page == "Feature Importance":
        page_feature_importance()
    elif page == "SHAP Explainability":
        page_shap()
    else:
        page_predict()


if __name__ == "__main__":
    main()
