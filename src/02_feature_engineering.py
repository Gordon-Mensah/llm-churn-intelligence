"""
LLM-Powered Customer Churn Intelligence System
================================================
Step 2: Feature Engineering
Combines classical telecom features with LLM-derived signals.

Author: John Jerry Gordon-Mensah
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""

    df = df.copy()

    # ── 1. Tenure Buckets ──────────────────────────────────────────────────────
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        include_lowest=True
    )

    # ── 2. Revenue Risk Score ─────────────────────────────────────────────────
    # High monthly charges + short tenure = high revenue-at-risk
    df["revenue_risk"] = (df["monthly_charges"] / (df["tenure"] + 1)).round(3)

    # ── 3. Service Density ────────────────────────────────────────────────────
    # How many services does the customer use? More = more sticky.
    service_cols = ["phone_service", "multiple_lines", "online_security",
                    "tech_support", "paperless_billing"]
    df["service_count"] = sum(
        (df[col] == "Yes").astype(int) for col in service_cols
    )

    # ── 4. LLM Sentiment Tier ─────────────────────────────────────────────────
    df["sentiment_tier"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.01, -0.33, 0.33, 1.01],
        labels=["negative", "neutral", "positive"]
    )

    # ── 5. High-Risk Complaint Flag ───────────────────────────────────────────
    high_risk_complaints = {"competitor_offer", "billing_dispute", "service_outage"}
    df["high_risk_complaint"] = df["complaint_theme"].isin(high_risk_complaints).astype(int)

    # ── 6. Composite Churn Risk Score (business-readable) ────────────────────
    df["manual_risk_score"] = (
        1.5 * (df["contract"] == "Month-to-month").astype(float) +
        1.2 * (df["tenure"] < 12).astype(float) +
        1.0 * (df["sentiment_score"] < -0.3).astype(float) +
        0.8 * df["high_risk_complaint"] +
        0.6 * (df["internet_service"] == "Fiber optic").astype(float) +
        0.5 * (df["payment_method"] == "Electronic check").astype(float)
    ).round(3)

    # ── 7. Encode Categoricals for Modelling ─────────────────────────────────
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "partner", "dependents", "phone_service",
                "paperless_billing", "online_security", "tech_support"]:
        df[f"{col}_enc"] = df[col].map(binary_map).fillna(0).astype(int)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_enc"] = df["contract"].map(contract_map)

    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
    df["internet_service_enc"] = df["internet_service"].map(internet_map)

    payment_map = {
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    }
    df["payment_method_enc"] = df["payment_method"].map(payment_map)

    complaint_map = {
        "none": 0, "speed_issues": 1, "contract_confusion": 1,
        "billing_dispute": 2, "service_outage": 2, "competitor_offer": 3
    }
    df["complaint_severity"] = df["complaint_theme"].map(complaint_map)

    sentiment_tier_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment_tier_enc"] = df["sentiment_tier"].map(sentiment_tier_map).fillna(1).astype(int)

    return df


def get_model_features():
    """Returns the list of feature columns for model training."""
    return [
        "tenure", "monthly_charges", "total_charges",
        "service_count", "revenue_risk", "high_risk_complaint",
        "complaint_severity", "sentiment_score", "sentiment_tier_enc",
        "gender_enc", "senior_citizen", "partner_enc", "dependents_enc",
        "phone_service_enc", "paperless_billing_enc",
        "online_security_enc", "tech_support_enc",
        "contract_enc", "internet_service_enc", "payment_method_enc"
    ]


if __name__ == "__main__":
    df = pd.read_csv("data/telco_churn_enriched.csv")
    df_engineered = engineer_features(df)
    df_engineered.to_csv("data/telco_churn_features.csv", index=False)

    print(f"✅ Feature engineering complete: {df_engineered.shape[1]} columns")
    print(f"   New features: tenure_group, revenue_risk, service_count,")
    print(f"   sentiment_tier, high_risk_complaint, manual_risk_score")
    print(f"   Saved to: data/telco_churn_features.csv")