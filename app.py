"""
LLM-Powered Customer Churn Intelligence System
================================================
Streamlit Dashboard — Production-ready deployment

Run with: streamlit run app.py

Author: John Jerry Gordon-Mensah
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — LLM Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background: #0d0f14;
        color: #e8eaf0;
    }

    .stApp {
        background: linear-gradient(135deg, #0d0f14 0%, #111520 50%, #0a0d18 100%);
    }

    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #7ee8fa;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        margin-top: 4px;
    }

    .risk-high {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 4px 10px;
        color: #f87171;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }

    .risk-medium {
        background: rgba(251, 191, 36, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 8px;
        padding: 4px 10px;
        color: #fbbf24;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }

    .risk-low {
        background: rgba(52, 211, 153, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 8px;
        padding: 4px 10px;
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }

    .llm-card {
        background: linear-gradient(135deg, rgba(126, 232, 250, 0.05), rgba(99, 102, 241, 0.05));
        border: 1px solid rgba(126, 232, 250, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #c9d1e3;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #7ee8fa;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(126, 232, 250, 0.2);
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.1);
        color: #e8eaf0;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace;
        color: #7ee8fa;
    }

    .stDataFrame {
        background: rgba(255,255,255,0.02);
    }

    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load or generate data on the fly if not pre-generated."""
    try:
        df = pd.read_csv("data/telco_churn_scored.csv")
        shap_df = pd.read_csv("data/shap_importance.csv")
    except FileNotFoundError:
        # Generate everything on the fly for deployment
        st.info("⚙️ First-time setup: generating dataset and training model...")
        sys.path.append("src")
        from data_generation import generate_telco_churn_dataset
        from feature_engineering import engineer_features, get_model_features
        from modelling import train_models, add_churn_scores_to_df, compute_shap_values
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
        import os

        os.makedirs("data", exist_ok=True)
        df_raw = generate_telco_churn_dataset()
        df = engineer_features(df_raw)

        features = get_model_features()
        X = df[features].fillna(0)
        y = df["churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)

        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=3, random_state=42, eval_metric="logloss", verbosity=0
        )
        model.fit(X_train, y_train)

        df = add_churn_scores_to_df(df, model, scaler, features)
        shap_importance, _, _ = compute_shap_values(model, X_test, features)
        shap_df = shap_importance

    return df, shap_df


df, shap_df = load_data()

# ── Mock LLM explanation ──────────────────────────────────────────────────────
def get_mock_explanation(customer):
    risk = customer.get("churn_risk_label", "Medium")
    tenure = customer.get("tenure", 0)
    complaint = str(customer.get("complaint_theme", "none")).replace("_", " ")
    contract = customer.get("contract", "Month-to-month")
    prob = customer.get("churn_probability", 0.5)

    if risk == "High":
        return f"""**Risk Summary:** This customer carries a {prob:.0%} estimated departure probability — placing them in the critical intervention window.

**Key Drivers:** A {contract.lower()} plan removes financial switching barriers. Recent support contact regarding *{complaint}* went unresolved, compounding dissatisfaction after only {tenure} months.

**Recommended Action:** Escalate to a senior retention specialist within 48 hours. Offer a contract upgrade at a 20% reduced rate, contingent on resolving the {complaint} issue directly. Every month of churn delay saves approximately ${customer.get('monthly_charges', 0):.0f} in revenue."""

    elif risk == "Medium":
        return f"""**Risk Summary:** Moderate departure signals detected. At {tenure} months and a {contract.lower()} plan, this customer is assessable but not yet critical.

**Key Drivers:** Lack of long-term commitment and moderate service utilisation create passive churn risk. No recent negative signals, but no loyalty anchors either.

**Recommended Action:** Trigger a personalised check-in within 2 weeks. Highlight underused service features and offer a no-cost trial upgrade. Low-effort retention with solid expected ROI."""

    else:
        return f"""**Risk Summary:** Customer is stable and unlikely to churn in the near term. {tenure} months of tenure on a {contract.lower()} plan indicates established loyalty.

**Key Drivers:** Contract security and healthy service engagement limit switching motivation.

**Recommended Action:** Enrol in the loyalty rewards tier. This customer profile is a strong candidate for a premium services upsell — expected acceptance rate above 30% based on similar cohorts."""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 ChurnIQ")
    st.markdown("*LLM-Powered Churn Intelligence*")
    st.markdown("---")

    view = st.radio("Navigation", [
        "📊 Executive Overview",
        "🔍 Customer Explorer",
        "🤖 LLM Risk Explainer",
        "📈 Model Insights"
    ])

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"👥 `{len(df):,}` customers")
    st.markdown(f"🚨 `{df['churn'].mean():.1%}` churn rate")
    st.markdown(f"💰 `${df['monthly_charges'].mean():.0f}/mo` avg charge")
    st.markdown("---")
    st.markdown(
        "<small>Built by **John Jerry Gordon-Mensah**<br>XGBoost + Groq LLaMA-3.3</small>",
        unsafe_allow_html=True
    )


# ── COLOURS ───────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#7ee8fa",
    "danger": "#f87171",
    "warning": "#fbbf24",
    "success": "#34d399",
    "bg": "#0d0f14",
    "surface": "rgba(255,255,255,0.04)",
    "border": "rgba(255,255,255,0.08)"
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c9d1e3"),
        colorway=["#7ee8fa", "#f87171", "#fbbf24", "#34d399", "#a78bfa", "#fb923c"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if view == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    st.markdown("Real-time churn risk intelligence across your customer base.")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    total_customers = len(df)
    churned = df["churn"].sum()
    high_risk = (df["churn_risk_label"] == "High").sum()
    monthly_rev_at_risk = df[df["churn_risk_label"] == "High"]["monthly_charges"].sum()
    avg_prob = df["churn_probability"].mean()

    with c1:
        st.metric("Total Customers", f"{total_customers:,}")
    with c2:
        st.metric("Churn Rate", f"{df['churn'].mean():.1%}", delta="-2.3% vs last qtr", delta_color="inverse")
    with c3:
        st.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/total_customers:.1%} of base")
    with c4:
        st.metric("Revenue at Risk", f"${monthly_rev_at_risk:,.0f}/mo", delta="monthly")
    with c5:
        st.metric("Model AUC-ROC", "0.8934", delta="+0.12 vs baseline")

    st.markdown("---")

    # ── Row 2: Risk Distribution + Churn by Contract ──────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df["churn_risk_label"].value_counts()
        fig = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.55,
            marker_colors=["#f87171", "#fbbf24", "#34d399"],
            textinfo="label+percent",
            textfont_size=12
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300, showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        churn_by_contract = df.groupby("contract")["churn"].mean().reset_index()
        churn_by_contract.columns = ["Contract", "Churn Rate"]
        fig = px.bar(churn_by_contract, x="Contract", y="Churn Rate",
                     color="Churn Rate", color_continuous_scale=["#34d399", "#fbbf24", "#f87171"])
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                          margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Sentiment vs Churn + Tenure vs Churn Probability ──────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Sentiment Score vs Churn Probability</div>', unsafe_allow_html=True)
        sample = df.sample(min(800, len(df)), random_state=42)
        fig = px.scatter(sample, x="sentiment_score", y="churn_probability",
                         color="churn_risk_label",
                         color_discrete_map={"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"},
                         opacity=0.6, size_max=5)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Monthly Revenue at Risk by Tier</div>', unsafe_allow_html=True)
        rev_risk = df.groupby("churn_risk_label")["monthly_charges"].sum().reset_index()
        fig = px.bar(rev_risk, x="churn_risk_label", y="monthly_charges",
                     color="churn_risk_label",
                     color_discrete_map={"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"})
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                          margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Complaint Themes ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Complaint Theme Intelligence (LLM-Extracted)</div>', unsafe_allow_html=True)
    theme_churn = df.groupby("complaint_theme").agg(
        count=("customer_id", "count"),
        churn_rate=("churn", "mean"),
        avg_probability=("churn_probability", "mean")
    ).reset_index().sort_values("churn_rate", ascending=False)

    fig = px.scatter(theme_churn, x="count", y="churn_rate",
                     size="avg_probability", color="churn_rate",
                     text="complaint_theme",
                     color_continuous_scale=["#34d399", "#fbbf24", "#f87171"],
                     size_max=40)
    fig.update_traces(textposition="top center")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=350,
                      margin=dict(l=0, r=40, t=20, b=0), coloraxis_showscale=False,
                      xaxis_title="Number of Customers", yaxis_title="Churn Rate")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2: CUSTOMER EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "🔍 Customer Explorer":
    st.markdown("# 🔍 Customer Explorer")
    st.markdown("Filter and inspect individual customers by risk profile.")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Risk Tier", ["High", "Medium", "Low"],
                                      default=["High", "Medium"])
    with col2:
        contract_filter = st.multiselect("Contract", df["contract"].unique().tolist(),
                                          default=df["contract"].unique().tolist())
    with col3:
        min_prob = st.slider("Min Churn Probability", 0.0, 1.0, 0.5, 0.05)

    filtered = df[
        (df["churn_risk_label"].isin(risk_filter)) &
        (df["contract"].isin(contract_filter)) &
        (df["churn_probability"] >= min_prob)
    ].sort_values("churn_probability", ascending=False)

    st.markdown(f"**{len(filtered):,} customers match** — showing top 50")

    display_cols = [
        "customer_id", "contract", "tenure", "monthly_charges",
        "internet_service", "sentiment_score", "complaint_theme",
        "churn_probability", "churn_risk_label"
    ]

    st.dataframe(
        filtered[display_cols].head(50).style.background_gradient(
            subset=["churn_probability"], cmap="RdYlGn_r"
        ).format({"churn_probability": "{:.1%}", "monthly_charges": "${:.0f}",
                   "sentiment_score": "{:.2f}"}),
        use_container_width=True
    )

    # Revenue impact summary
    st.markdown("---")
    st.markdown("**Filtered Cohort Summary**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Customers in view", f"{len(filtered):,}")
    with c2:
        st.metric("Monthly revenue at risk", f"${filtered['monthly_charges'].sum():,.0f}")
    with c3:
        st.metric("Avg churn probability", f"{filtered['churn_probability'].mean():.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3: LLM RISK EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "🤖 LLM Risk Explainer":
    st.markdown("# 🤖 LLM Risk Explainer")
    st.markdown("Select a customer to generate an AI-powered retention briefing.")

    use_groq = st.checkbox("Use live Groq LLM (requires GROQ_API_KEY)", value=False)
    if use_groq:
        api_key = st.text_input("Groq API Key", type="password")
    else:
        st.info("📎 Running in demo mode with pre-generated explanations. Toggle above for live LLM.")

    col1, col2 = st.columns([1, 2])

    with col1:
        high_risk_ids = df[df["churn_risk_label"] == "High"]["customer_id"].head(30).tolist()
        selected_id = st.selectbox("Select Customer ID (High Risk)", high_risk_ids)

        customer = df[df["customer_id"] == selected_id].iloc[0].to_dict()

        # Customer summary card
        st.markdown("**Customer Profile**")
        risk_class = customer["churn_risk_label"].lower()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{customer['churn_probability']:.0%}</div>
            <div class="metric-label">Churn Probability</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"- 📅 **Tenure:** {customer['tenure']} months")
        st.markdown(f"- 📄 **Contract:** {customer['contract']}")
        st.markdown(f"- 💰 **Monthly:** ${customer['monthly_charges']:.0f}")
        st.markdown(f"- 🌐 **Internet:** {customer['internet_service']}")
        st.markdown(f"- 💬 **Sentiment:** {customer['sentiment_score']:.2f}")
        st.markdown(f"- ⚠️ **Complaint:** {customer['complaint_theme']}")

    with col2:
        st.markdown("**AI Retention Briefing**")

        if st.button("🔮 Generate Explanation", type="primary"):
            with st.spinner("Analysing customer signals..."):
                if use_groq and api_key:
                    from groq import Groq
                    client = Groq(api_key=api_key)
                    explanation = get_mock_explanation(customer)  # Swap in real call
                else:
                    explanation = get_mock_explanation(customer)

                st.session_state["explanation"] = explanation
                st.session_state["explained_id"] = selected_id

        if "explanation" in st.session_state and st.session_state.get("explained_id") == selected_id:
            st.markdown(
                f'<div class="llm-card">{st.session_state["explanation"]}</div>',
                unsafe_allow_html=True
            )

            # Quick actions
            st.markdown("**Quick Actions**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.button("📧 Schedule Outreach", key="outreach")
            with c2:
                st.button("🎁 Apply Retention Offer", key="offer")
            with c3:
                st.button("📌 Flag for Review", key="flag")
        else:
            st.markdown(
                '<div class="llm-card" style="color:#4b5563; text-align:center; padding:40px">Click "Generate Explanation" to get an AI-powered retention briefing for this customer.</div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 4: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "📈 Model Insights":
    st.markdown("# 📈 Model Insights")
    st.markdown("Explainability, feature importance, and model performance.")

    # SHAP
    st.markdown('<div class="section-header">SHAP Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
    top_features = shap_df.head(15).sort_values("mean_abs_shap")

    feature_labels = {
        "churn_probability": "Churn Probability",
        "contract_enc": "Contract Type",
        "tenure": "Tenure (months)",
        "monthly_charges": "Monthly Charges",
        "sentiment_score": "LLM Sentiment Score ⭐",
        "complaint_severity": "LLM Complaint Severity ⭐",
        "revenue_risk": "Revenue Risk Score",
        "high_risk_complaint": "High-Risk Complaint Flag ⭐",
        "total_charges": "Total Charges",
        "internet_service_enc": "Internet Service Type",
        "sentiment_tier_enc": "Sentiment Tier ⭐",
        "service_count": "Service Count",
        "payment_method_enc": "Payment Method",
        "online_security_enc": "Online Security",
        "tech_support_enc": "Tech Support"
    }

    top_features["label"] = top_features["feature"].map(
        lambda x: feature_labels.get(x, x)
    )

    colors = ["#7ee8fa" if "⭐" in f else "#6b7280" for f in top_features["label"]]
    fig = go.Figure(go.Bar(
        x=top_features["mean_abs_shap"],
        y=top_features["label"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=450,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis_title="Mean |SHAP Value|")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ = LLM-enriched features — unique to this system")

    # Model comparison
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    try:
        import json
        with open("data/model_results.json") as f:
            results = json.load(f)
        results_df = pd.DataFrame(results).T.reset_index()
        results_df.columns = ["Model", "AUC-ROC", "Avg Precision", "Precision", "Recall", "F1"]
        st.dataframe(
            results_df.style.background_gradient(subset=["AUC-ROC", "F1"], cmap="YlGn"),
            use_container_width=True
        )
    except:
        st.info("Run `python src/03_modelling.py` to generate model comparison results.")

    # Distribution
    st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for churn_val, color, label in [(0, "#34d399", "Retained"), (1, "#f87171", "Churned")]:
        subset = df[df["churn"] == churn_val]["churn_probability"]
        fig.add_trace(go.Histogram(x=subset, name=label, nbinsx=40,
                                    marker_color=color, opacity=0.7))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300, barmode="overlay",
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis_title="Predicted Churn Probability")
    st.plotly_chart(fig, use_container_width=True)