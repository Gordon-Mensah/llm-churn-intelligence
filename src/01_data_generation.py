"""
LLM-Powered Customer Churn Intelligence System
================================================
Step 1: Synthetic Data Generation (mirrors real telecom churn datasets)
Based on IBM Telco Customer Churn dataset schema + enriched with
LLM-generated behavioural signals.

Author: John Jerry Gordon-Mensah
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N = 7043  # Same size as IBM Telco dataset

def generate_telco_churn_dataset(n=N):
    """
    Generate a realistic telecom churn dataset with LLM-enrichable features.
    Mirrors the IBM Telco Customer Churn dataset schema but with added
    'sentiment_score' and 'complaint_theme' columns representing LLM-enriched signals.
    """

    # --- Demographics ---
    gender = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n)
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = np.random.randint(0, 72, n)

    # --- Services ---
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == "No", "No phone service",
                              np.random.choice(["Yes", "No"], n))
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_security = np.where(internet_service == "No", "No internet service",
                                np.random.choice(["Yes", "No"], n))
    tech_support = np.where(internet_service == "No", "No internet service",
                             np.random.choice(["Yes", "No"], n))
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], n)
    payment_method = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], n, p=[0.34, 0.23, 0.22, 0.21])

    # --- Financials ---
    monthly_charges = np.round(np.random.uniform(18, 118, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(0, 50, n), 2)

    # --- LLM-ENRICHED FEATURES (simulated NLP outputs from support ticket analysis) ---
    # Sentiment score: LLM-assigned sentiment from last customer service interaction (-1 to 1)
    # Negative sentiment correlates with churn
    base_sentiment = np.random.normal(0.2, 0.4, n)
    sentiment_score = np.clip(base_sentiment, -1, 1).round(3)

    # Complaint theme: LLM-extracted topic from last support ticket
    complaint_themes = [
        "billing_dispute", "service_outage", "speed_issues",
        "contract_confusion", "competitor_offer", "none", "none", "none"
    ]
    complaint_theme = np.random.choice(complaint_themes, n)

    # Churn label — driven by risk factors
    churn_prob = (
        0.08 +
        0.25 * (contract == "Month-to-month").astype(float) +
        0.10 * (internet_service == "Fiber optic").astype(float) +
        0.08 * (payment_method == "Electronic check").astype(float) +
        0.15 * (tenure < 12).astype(float) +
        0.10 * (online_security == "No").astype(float) +
        0.20 * (sentiment_score < -0.3).astype(float) +
        0.25 * (complaint_theme == "competitor_offer").astype(float) +
        0.15 * (complaint_theme == "billing_dispute").astype(float) -
        0.12 * (contract == "Two year").astype(float) -
        0.05 * (tenure > 48).astype(float)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{str(i).zfill(5)}" for i in range(1, n+1)],
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "sentiment_score": sentiment_score,
        "complaint_theme": complaint_theme,
        "churn": churn
    })

    return df


if __name__ == "__main__":
    df = generate_telco_churn_dataset()
    df.to_csv("data/telco_churn_enriched.csv", index=False)
    print(f"✅ Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Churn rate: {df['churn'].mean():.1%}")
    print(f"   Saved to: data/telco_churn_enriched.csv")