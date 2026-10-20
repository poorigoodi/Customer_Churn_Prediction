"""Train, cross-validate, and serialize the best churn prediction model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow both `python -m src.train` and `python src/train.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import (
    confusion_matrix_dataframe,
    evaluate_on_test,
    save_comparison_chart,
    save_confusion_matrix_figure,
    save_evaluation_report,
    save_results_to_excel,
    save_roc_curve_figure,
)
from src.explain import run_explainability
from src.preprocessing import (
    RANDOM_STATE,
    build_preprocessor,
    get_project_root,
    load_raw_data,
    prepare_features_and_target,
    stratified_train_test_split,
)


def get_model_candidates() -> Dict[str, Any]:
    """Return classifier instances used for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_model_pipeline(classifier: Any) -> Pipeline:
    """Combine preprocessing and a classifier into a single pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, str, Dict[str, Pipeline]]:
    """
    Compare models with stratified 5-fold CV.

    Selection uses mean ROC-AUC on validation folds only.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows: List[Dict[str, float | str]] = []
    fitted_templates: Dict[str, Pipeline] = {}

    for name, classifier in get_model_candidates().items():
        pipeline = build_model_pipeline(classifier)
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        row = {
            "Model": name,
            "Accuracy": float(np.mean(cv_results["test_accuracy"])),
            "Precision": float(np.mean(cv_results["test_precision"])),
            "Recall": float(np.mean(cv_results["test_recall"])),
            "F1": float(np.mean(cv_results["test_f1"])),
            "ROC-AUC": float(np.mean(cv_results["test_roc_auc"])),
        }
        rows.append(row)
        fitted_templates[name] = pipeline
        print(
            f"[CV] {name}: "
            f"ROC-AUC={row['ROC-AUC']:.4f}, "
            f"F1={row['F1']:.4f}, "
            f"Precision={row['Precision']:.4f}, "
            f"Recall={row['Recall']:.4f}"
        )

    comparison_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    best_model_name = str(comparison_df.iloc[0]["Model"])
    return comparison_df, best_model_name, fitted_templates


def fit_best_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the selected model pipeline on the full training set."""
    classifier = get_model_candidates()[model_name]
    pipeline = build_model_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, output_path: Path | None = None) -> Path:
    """Serialize the complete preprocessing + model pipeline with joblib."""
    root = get_project_root()
    path = output_path or (root / "models" / "churn_model.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def save_cv_comparison(comparison_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Persist cross-validation comparison metrics as JSON."""
    root = get_project_root()
    path = output_path or (root / "reports" / "cv_comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = comparison_df.round(4).to_dict(orient="records")
    payload = {"selection_metric": "ROC-AUC", "cross_validation": records}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_training(data_path: Path | str | None = None) -> Dict[str, Any]:
    """
    End-to-end training workflow:

    1. Load and clean data
    2. Stratified train/test split
    3. Cross-validate candidate models on training folds
    4. Fit best model on full training set
    5. Evaluate on held-out test set
    6. Serialize the best pipeline
    """
    root = get_project_root()
    figures_dir = root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    raw_df = load_raw_data(data_path)
    X, y = prepare_features_and_target(raw_df)

    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Persist a lightweight processed snapshot for inspection (features only metadata)
    processed_meta = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    (root / "data" / "processed" / "dataset_meta.json").write_text(
        json.dumps(processed_meta, indent=2),
        encoding="utf-8",
    )

    print("Running stratified 5-fold cross-validation...")
    comparison_df, best_model_name, _ = cross_validate_models(X_train, y_train)
    print(f"\nBest model by CV ROC-AUC: {best_model_name}")
    save_cv_comparison(comparison_df)

    print(f"Fitting {best_model_name} on the full training set...")
    best_pipeline = fit_best_model(best_model_name, X_train, y_train)

    print("Evaluating on held-out test set...")
    test_metrics, y_proba, y_pred = evaluate_on_test(best_pipeline, X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Also score all models on test for reporting transparency (selection already done via CV)
    test_rows = []
    for name in comparison_df["Model"]:
        pipe = fit_best_model(str(name), X_train, y_train)
        metrics, _, _ = evaluate_on_test(pipe, X_test, y_test)
        test_rows.append({"Model": name, **metrics})
    test_comparison_df = pd.DataFrame(test_rows)

    confusion_df = confusion_matrix_dataframe(y_test, y_pred)
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        figures_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — {best_model_name}",
    )
    save_roc_curve_figure(
        y_test,
        y_proba,
        figures_dir / "roc_curve.png",
        title=f"ROC Curve — {best_model_name}",
    )
    save_comparison_chart(
        comparison_df,
        figures_dir / "model_comparison.png",
        title="Model Comparison (CV ROC-AUC)",
    )

    print("Computing feature importance and SHAP explainability...")
    explain_result = run_explainability(
        best_pipeline,
        X_test,
        figures_dir=figures_dir,
        sample_size=300,
        random_state=RANDOM_STATE,
    )

    report_path = save_evaluation_report(
        best_model_name=best_model_name,
        cv_comparison=comparison_df,
        test_metrics=test_metrics,
        test_comparison=test_comparison_df,
        output_path=root / "reports" / "evaluation_report.json",
    )

    excel_path = save_results_to_excel(
        output_path=root / "reports" / "model_results.xlsx",
        best_model_name=best_model_name,
        cv_comparison=comparison_df,
        test_comparison=test_comparison_df,
        test_metrics=test_metrics,
        confusion_df=confusion_df,
        feature_importance=explain_result["feature_importance"],
        shap_importance=explain_result["shap_importance"],
    )

    model_path = save_model(best_pipeline)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved evaluation report to: {report_path}")
    print(f"Saved Excel results to: {excel_path}")
    print(f"Saved feature importance chart: {explain_result['feature_importance_figure']}")
    print(f"Saved SHAP summary chart: {explain_result['shap_figure']}")

    return {
        "best_model": best_model_name,
        "cv_comparison": comparison_df,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "excel_path": str(excel_path),
        "feature_importance": explain_result["feature_importance"],
        "shap_importance": explain_result["shap_importance"],
    }


if __name__ == "__main__":
    run_training()
