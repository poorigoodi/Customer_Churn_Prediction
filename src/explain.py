"""Feature importance and SHAP explainability helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return feature names after the preprocessing step."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply fitted preprocessor to raw features."""
    return pipeline.named_steps["preprocessor"].transform(X)


def compute_feature_importance(
    pipeline: Pipeline,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract model-native feature importance.

    - Tree models: feature_importances_
    - Linear models: absolute coefficients
    """
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_transformed_feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        method = "feature_importances_"
    elif hasattr(classifier, "coef_"):
        coef = np.asarray(classifier.coef_, dtype=float)
        values = np.abs(coef[0] if coef.ndim > 1 else coef)
        method = "abs_coefficients"
    else:
        raise AttributeError(
            f"Classifier {type(classifier).__name__} has no supported importance attribute."
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["method"] = method
    return importance_df.head(top_n).copy()


def save_feature_importance_figure(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str = "Feature Importance",
) -> Path:
    """Save a horizontal bar chart of feature importance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = importance_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#264653")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def compute_shap_summary(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    max_display: int = 20,
) -> Tuple[pd.DataFrame, Any]:
    """
    Compute mean |SHAP| values for a sample of customers.

    Returns
    -------
    shap_importance_df, shap_values
    """
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_transformed_feature_names(pipeline)
    X_transformed = transform_features(pipeline, X_sample)

    # TreeExplainer is fast for RF/XGBoost; Explainer falls back for linear models
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        explainer = shap.Explainer(classifier, X_transformed)
        explanation = explainer(X_transformed)
        shap_values = explanation.values

    # Binary classifiers may return list [class0, class1]
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]

    mean_abs = np.mean(np.abs(values), axis=0)
    shap_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .head(max_display)
    )
    return shap_df, values


def save_shap_summary_figure(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 20,
) -> Tuple[Path, pd.DataFrame]:
    """Save a SHAP bar summary plot and return mean-|SHAP| table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier = pipeline.named_steps["classifier"]
    feature_names = get_transformed_feature_names(pipeline)
    X_transformed = transform_features(pipeline, X_sample)

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        explainer = shap.Explainer(classifier, X_transformed)
        explanation = explainer(X_transformed)
        shap_values = explanation.values
        if getattr(shap_values, "ndim", 0) == 3:
            shap_values = shap_values[:, :, 1]

    shap_values = np.asarray(shap_values)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .head(max_display)
    )

    plt.figure(figsize=(9, max(4, 0.35 * max_display)))
    shap.summary_plot(
        shap_values,
        features=X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Also save a beeswarm-style summary when feasible
    beeswarm_path = output_path.with_name("shap_beeswarm.png")
    try:
        plt.figure(figsize=(9, max(4, 0.35 * max_display)))
        shap.summary_plot(
            shap_values,
            features=X_transformed,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        plt.close()

    return output_path, shap_df


def run_explainability(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    figures_dir: Path,
    sample_size: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute feature importance + SHAP and save figures/tables.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    importance_df = compute_feature_importance(pipeline, top_n=20)
    importance_fig = save_feature_importance_figure(
        importance_df,
        figures_dir / "feature_importance.png",
        title="Feature Importance (Top 20)",
    )

    n = min(sample_size, len(X_test))
    X_sample = X_test.sample(n=n, random_state=random_state)

    shap_fig, shap_df = save_shap_summary_figure(
        pipeline,
        X_sample,
        figures_dir / "shap_summary.png",
        max_display=20,
    )

    return {
        "feature_importance": importance_df,
        "shap_importance": shap_df,
        "feature_importance_figure": importance_fig,
        "shap_figure": shap_fig,
        "shap_sample_size": n,
    }
