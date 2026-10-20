"""Generate EDA figures for reports/figures (runnable without Jupyter)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import get_default_data_path, load_raw_data

FIGURES_DIR = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data(get_default_data_path())
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 1. Churn Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    order = df["Churn"].value_counts().index
    sns.countplot(
        data=df,
        x="Churn",
        order=order,
        hue="Churn",
        ax=ax,
        palette="Set2",
        legend=False,
    )
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Churn by Contract Type
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
    )
    contract_churn.plot(kind="bar", ax=ax, color="#e76f51")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("Contract")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=150)
    plt.close(fig)

    # 3. Churn by Internet Service
    fig, ax = plt.subplots(figsize=(8, 5))
    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
    )
    internet_churn.plot(kind="bar", ax=ax, color="#457b9d")
    ax.set_title("Churn Rate by Internet Service")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("Internet Service")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_by_internet.png", dpi=150)
    plt.close(fig)

    # 4. Monthly Charges Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True, ax=ax, palette="muted")
    ax.set_title("Monthly Charges Distribution by Churn")
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_charges_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Tenure vs Churn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure",
        hue="Churn",
        ax=ax,
        palette="pastel",
        legend=False,
    )
    ax.set_title("Tenure vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Tenure (months)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tenure_vs_churn.png", dpi=150)
    plt.close(fig)

    # 6. Correlation heatmap for numerical features
    numeric = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    numeric["Churn"] = (df["Churn"] == "Yes").astype(int)
    corr = numeric.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (Numerical Features + Churn)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Extra: payment method and tech support for notebook parity
    fig, ax = plt.subplots(figsize=(9, 5))
    pay = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
    )
    pay.plot(kind="bar", ax=ax, color="#2a9d8f")
    ax.set_title("Churn Rate by Payment Method")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_by_payment.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    tech = (
        df.groupby("TechSupport")["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .sort_values(ascending=False)
    )
    tech.plot(kind="bar", ax=ax, color="#9b2226")
    ax.set_title("Churn Rate by Tech Support")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_by_techsupport.png", dpi=150)
    plt.close(fig)

    print(f"Saved EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
