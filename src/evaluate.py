"""Model evaluation helpers, Excel export, and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a fitted pipeline on the held-out test set.

    Returns
    -------
    metrics, y_proba, y_pred
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics, y_proba, y_pred


def confusion_matrix_dataframe(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a labeled confusion-matrix dataframe."""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    )


def save_confusion_matrix_figure(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_roc_curve_figure(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
) -> Path:
    """Save an ROC curve figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_comparison_chart(
    comparison_df: pd.DataFrame,
    output_path: Path,
    title: str = "Model Comparison (CV ROC-AUC)",
) -> Path:
    """Save a bar chart comparing model ROC-AUC scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = comparison_df.sort_values("ROC-AUC", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(plot_df["Model"], plot_df["ROC-AUC"], color="#2a9d8f")
    ax.set_xlabel("Mean CV ROC-AUC")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    for bar, value in zip(bars, plot_df["ROC-AUC"]):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_evaluation_report(
    best_model_name: str,
    cv_comparison: pd.DataFrame,
    test_metrics: Dict[str, float],
    test_comparison: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a machine-readable evaluation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    false_positive_note = (
        "False Positive: predicting Churn when the customer stays. "
        "May trigger unnecessary retention offers and increase campaign cost."
    )
    false_negative_note = (
        "False Negative: predicting No Churn when the customer leaves. "
        "Missed intervention opportunity and higher revenue risk."
    )

    payload: Dict[str, Any] = {
        "best_model_selected_by": "CV ROC-AUC",
        "best_model": best_model_name,
        "cross_validation_metrics": cv_comparison.round(4).to_dict(orient="records"),
        "test_metrics_best_model": {k: round(v, 4) for k, v in test_metrics.items()},
        "test_metrics_all_models": test_comparison.round(4).to_dict(orient="records"),
        "error_analysis": {
            "false_positives": false_positive_note,
            "false_negatives": false_negative_note,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def format_metrics_table(df: pd.DataFrame) -> str:
    """Return a markdown table string for documentation."""
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    display = df[cols].copy()
    for col in cols[1:]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[c]) for c in cols) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def save_results_to_excel(
    *,
    output_path: Path,
    best_model_name: str,
    cv_comparison: pd.DataFrame,
    test_comparison: pd.DataFrame,
    test_metrics: Dict[str, float],
    confusion_df: pd.DataFrame,
    feature_importance: Optional[pd.DataFrame] = None,
    shap_importance: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Save evaluation outputs to a multi-sheet Excel workbook.

    Sheets:
    - Model_Evaluation (best-model test metrics)
    - CV_Comparison
    - Test_Comparison
    - Confusion_Matrix
    - Feature_Importance
    - SHAP_Explainability
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_metrics_df = pd.DataFrame(
        [{"Model": best_model_name, **{k: round(v, 4) for k, v in test_metrics.items()}}]
    )
    notes_df = pd.DataFrame(
        [
            {
                "note": (
                    "False Positive: predicting Churn when the customer stays. "
                    "May trigger unnecessary retention offers."
                )
            },
            {
                "note": (
                    "False Negative: predicting No Churn when the customer leaves. "
                    "Missed intervention opportunity."
                )
            },
            {
                "note": (
                    "Risk bands are illustrative: Low < 0.30, "
                    "Medium [0.30, 0.70), High >= 0.70."
                )
            },
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        best_metrics_df.to_excel(writer, sheet_name="Model_Evaluation", index=False)
        cv_comparison.round(4).to_excel(writer, sheet_name="CV_Comparison", index=False)
        test_comparison.round(4).to_excel(writer, sheet_name="Test_Comparison", index=False)
        confusion_df.to_excel(writer, sheet_name="Confusion_Matrix")
        notes_df.to_excel(writer, sheet_name="Notes", index=False)

        if feature_importance is not None and not feature_importance.empty:
            feature_importance.to_excel(
                writer, sheet_name="Feature_Importance", index=False
            )
        else:
            pd.DataFrame({"message": ["Feature importance unavailable"]}).to_excel(
                writer, sheet_name="Feature_Importance", index=False
            )

        if shap_importance is not None and not shap_importance.empty:
            shap_importance.to_excel(
                writer, sheet_name="SHAP_Explainability", index=False
            )
        else:
            pd.DataFrame({"message": ["SHAP values unavailable"]}).to_excel(
                writer, sheet_name="SHAP_Explainability", index=False
            )

    return output_path
