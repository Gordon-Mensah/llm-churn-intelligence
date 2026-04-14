"""
LLM-Powered Customer Churn Intelligence System
================================================
Step 4: LLM Explainer — the unique differentiator
Uses Groq (llama-3.3-70b) to generate human-readable churn explanations
for individual customers. Turns model outputs into executive-ready narratives.

Author: John Jerry Gordon-Mensah
"""

import os
import json
import time
from groq import Groq


GROQ_SYSTEM_PROMPT = """You are a senior customer retention analyst at a telecom company. 
You receive structured data about a customer and their churn risk score from a machine learning model.
Your job is to write a concise, human-readable retention briefing for the account management team.

Rules:
- Be specific — reference actual values from the data
- Write in 3 short sections: Risk Summary, Key Drivers, Recommended Action
- Keep it under 120 words total
- Use plain business language, no jargon
- Always end with one concrete, actionable recommendation
- Do not mention model names or scores directly — translate them to business language
"""


def build_customer_prompt(customer: dict) -> str:
    """Format a customer record into a structured prompt for the LLM."""
    return f"""
Customer profile:
- Tenure: {customer.get('tenure', 'N/A')} months
- Contract type: {customer.get('contract', 'N/A')}
- Monthly charges: ${customer.get('monthly_charges', 'N/A')}
- Internet service: {customer.get('internet_service', 'N/A')}
- Last support sentiment: {customer.get('sentiment_score', 'N/A')} (scale: -1 very negative to +1 very positive)
- Last complaint topic: {customer.get('complaint_theme', 'none')}
- Service count: {customer.get('service_count', 'N/A')} active services
- ML churn probability: {customer.get('churn_probability', 'N/A'):.0%}
- Risk tier: {customer.get('churn_risk_label', 'N/A')}

Write a retention briefing for this customer.
"""


def get_llm_explanation(customer: dict, client: Groq = None, mock: bool = False) -> str:
    """
    Get LLM-generated explanation for a customer's churn risk.
    
    Args:
        customer: dict of customer features + churn score
        client: Groq client instance (pass None to use mock)
        mock: if True, return a canned example (for demo/testing without API key)
    
    Returns:
        str: Human-readable retention briefing
    """
    if mock or client is None:
        return _mock_explanation(customer)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                {"role": "user", "content": build_customer_prompt(customer)}
            ],
            max_tokens=200,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ LLM explanation unavailable: {str(e)}"


def _mock_explanation(customer: dict) -> str:
    """
    Returns a realistic canned explanation for demo purposes.
    Used when no Groq API key is available.
    """
    risk = customer.get("churn_risk_label", "Medium")
    tenure = customer.get("tenure", 0)
    complaint = customer.get("complaint_theme", "none")
    contract = customer.get("contract", "Month-to-month")

    if risk == "High":
        return f"""**Risk Summary:** This customer is at high risk of leaving within the next 30 days. Short tenure ({tenure} months), a month-to-month contract, and a recent {complaint.replace('_', ' ')} complaint are compounding indicators.

**Key Drivers:** Month-to-month flexibility means zero switching cost. The negative recent interaction suggests unresolved dissatisfaction.

**Recommended Action:** Assign to a retention specialist immediately. Offer a 15% loyalty discount tied to an annual contract upgrade — targeting resolution of the {complaint.replace('_', ' ')} issue first."""

    elif risk == "Medium":
        return f"""**Risk Summary:** This customer shows moderate churn signals worth monitoring. They've been with us {tenure} months and are on a {contract} plan.

**Key Drivers:** Moderate service engagement and no long-term commitment create a window of vulnerability.

**Recommended Action:** Trigger an automated check-in email highlighting unused service benefits. Consider a prorated annual upgrade offer within the next 2 weeks."""

    else:
        return f"""**Risk Summary:** This customer is stable and likely to renew. {tenure} months tenure on a {contract} plan signals healthy loyalty.

**Key Drivers:** Strong tenure and contract security make switching unlikely in the near term.

**Recommended Action:** Include in the loyalty rewards program. This customer could be a candidate for an upsell on premium services."""


def batch_explain(df, client=None, mock=True, sample_n=None, delay=0.5):
    """
    Generate LLM explanations for a dataframe of customers.
    
    Args:
        df: DataFrame with customer features + churn scores
        client: Groq client (None = use mock)
        mock: use mock explanations (no API cost)
        sample_n: only process N rows (for testing)
        delay: seconds between API calls (rate limiting)
    
    Returns:
        df with added 'llm_explanation' column
    """
    if sample_n:
        df = df.head(sample_n).copy()
    else:
        df = df.copy()

    explanations = []
    for _, row in df.iterrows():
        explanation = get_llm_explanation(row.to_dict(), client=client, mock=mock)
        explanations.append(explanation)
        if not mock and delay:
            time.sleep(delay)

    df["llm_explanation"] = explanations
    return df


if __name__ == "__main__":
    import pandas as pd

    print("🤖 Testing LLM Explainer (mock mode)...")
    df = pd.read_csv("data/telco_churn_scored.csv")

    # Test on 3 high-risk customers
    high_risk = df[df["churn_risk_label"] == "High"].head(3)
    for _, customer in high_risk.iterrows():
        print(f"\n── Customer {customer['customer_id']} ──")
        print(f"   Churn probability: {customer['churn_probability']:.0%}")
        explanation = get_llm_explanation(customer.to_dict(), mock=True)
        print(explanation)
        print()

    print("\n✅ LLM Explainer working correctly.")
    print("   Set GROQ_API_KEY environment variable to use real LLM explanations.")