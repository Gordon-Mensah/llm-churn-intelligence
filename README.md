# 🔮 ChurnIQ — LLM-Powered Customer Churn Intelligence

> **A production-grade data science system that combines XGBoost predictive modelling with Groq LLaMA-3.3 to generate human-readable retention briefs for at-risk customers — turning model outputs into executive decisions.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Groq LLaMA](https://img.shields.io/badge/LLM-Groq%20LLaMA--3.3-purple.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.8934-brightgreen.svg)](#model-performance)

---

## 🎯 Business Problem

Telecom companies lose **15–25% of their customer base each year** to churn. The challenge isn't predicting *who* will leave — it's knowing *why* and *what to say* to keep them.

**ChurnIQ solves both problems:**
1. **XGBoost model** scores every customer with a calibrated churn probability
2. **Groq LLM layer** reads the model's signals and writes a human-readable retention brief — specific enough for a customer success manager to act on immediately

Most churn systems produce a score. ChurnIQ produces a conversation.

---

## ✨ What Makes This Different

| Standard Churn Model | ChurnIQ |
|---|---|
| Outputs a probability score | Outputs a probability + LLM retention brief |
| Feature importance as a bar chart | SHAP explainability per customer |
| Requires a data analyst to interpret | Executive-readable insights for any team |
| Static model | LLM enriched with complaint sentiment signals |

The **LLM enrichment layer** is the core innovation: support ticket themes and sentiment scores (simulating a real NLP pipeline on customer service data) are fed directly into both the ML model and the LLM explainer. This creates a feedback loop between model performance and narrative quality.

---

## 📊 System Architecture

```
Customer Data (7,043 records)
        │
        ▼
Feature Engineering ──── LLM-Enriched Signals
  │  tenure, charges        │  sentiment_score (NLP)
  │  contract, services     │  complaint_theme (LLM)
  │  payment method         │  sentiment_tier
        │
        ▼
XGBoost Classifier (AUC-ROC: 0.8934)
        │
        ▼
Churn Probability Score + Risk Tier
        │
        ▼
Groq LLaMA-3.3-70b ──── Retention Briefing
                          (personalised, actionable)
        │
        ▼
Streamlit Dashboard ──── Executive Overview
                          Customer Explorer
                          LLM Explainer
                          Model Insights
```

---

## 🧠 LLM Enrichment Layer

The most distinctive element of this project. Real-world customer service platforms (Zendesk, Salesforce) generate thousands of support tickets. This project simulates an NLP pipeline that would:

1. **Sentiment Analysis** → score each customer's last interaction (-1 to +1)
2. **Topic Extraction** → classify the complaint (`billing_dispute`, `competitor_offer`, `service_outage`, etc.)
3. **Risk Amplification** → feed these signals into XGBoost as features (improving AUC by ~4%)
4. **Narrative Generation** → pass combined signals to Groq LLaMA-3.3 for a retention brief

The LLM prompt is engineered to produce structured, actionable outputs — not generic summaries.

---

## 📁 Project Structure

```
llm-churn-intelligence/
├── app.py                        # Streamlit dashboard (deploy this)
├── requirements.txt
├── Makefile                      # make all → generates data, trains, launches
├── .env.example                  # GROQ_API_KEY placeholder
├── data/
│   ├── telco_churn_enriched.csv  # Raw generated dataset
│   ├── telco_churn_features.csv  # Feature-engineered dataset
│   ├── telco_churn_scored.csv    # Final dataset with churn scores
│   ├── shap_importance.csv       # SHAP feature importance
│   └── model_results.json        # Model comparison metrics
└── src/
    ├── 01_data_generation.py     # Synthetic dataset generation
    ├── 02_feature_engineering.py # Feature pipeline
    ├── 03_modelling.py           # Training, evaluation, SHAP
    └── 04_llm_explainer.py       # Groq LLM integration
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Gordon-Mensah/llm-churn-intelligence
cd llm-churn-intelligence
pip install -r requirements.txt

# 2. (Optional) Add Groq API key for live LLM explanations
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Generate data + train model + launch dashboard
make all

# Or step by step:
python src/01_data_generation.py
python src/02_feature_engineering.py
python src/03_modelling.py
streamlit run app.py
```

---

## 📈 Model Performance

| Model | AUC-ROC | Avg Precision | F1 (Churn) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7821 | 0.5634 | 0.5820 |
| Random Forest | 0.8612 | 0.6891 | 0.6340 |
| **XGBoost (final)** | **0.8934** | **0.7412** | **0.6891** |

**LLM features contribution:** Adding `sentiment_score`, `complaint_severity`, and `high_risk_complaint` to XGBoost improved AUC-ROC by **+4.1%** over the same model without them.

---

## 🛠 Feature Engineering Highlights

| Feature | Type | Description |
|---|---|---|
| `revenue_risk` | Engineered | `monthly_charges / (tenure + 1)` — high charges on new customers |
| `service_count` | Engineered | Number of active services — sticky indicator |
| `sentiment_score` | **LLM-enriched** | NLP score from last support interaction |
| `complaint_severity` | **LLM-enriched** | 0–3 scale from LLM-extracted complaint topic |
| `high_risk_complaint` | **LLM-enriched** | Binary flag: competitor_offer / billing_dispute / outage |
| `sentiment_tier` | **LLM-enriched** | Categorical: negative / neutral / positive |

---

## 🤖 LLM Explainer: Sample Output

**Customer: CUST-00234 | Contract: Month-to-month | Churn Risk: 87%**

> **Risk Summary:** This customer carries an 87% estimated departure probability — placing them in the critical intervention window.
>
> **Key Drivers:** A month-to-month plan removes financial switching barriers. Recent support contact regarding *competitor offer* went unresolved, compounding dissatisfaction after only 4 months.
>
> **Recommended Action:** Escalate to a senior retention specialist within 48 hours. Offer a contract upgrade at a 20% reduced rate, contingent on resolving the competitor offer concern directly. Every month of churn delay saves approximately $94 in revenue.

---

## 🌐 Deployment

**Streamlit Community Cloud:**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set `app.py` as main file
4. Add `GROQ_API_KEY` in Secrets

**Hugging Face Spaces:**
1. Create a new Space with Streamlit runtime
2. Push this repository
3. Add `GROQ_API_KEY` to Space secrets

---

## 🏆 Skills Demonstrated

- **End-to-end ML pipeline**: data generation → EDA → feature engineering → modelling → deployment
- **LLM integration**: Groq API, prompt engineering, structured output generation
- **Explainable AI**: SHAP values, per-customer explanations
- **Production dashboard**: Streamlit, Plotly, custom dark UI
- **Business framing**: Revenue-at-risk quantification, retention ROI logic
- **Class imbalance handling**: `scale_pos_weight`, precision-recall optimisation

---

## 👤 Author

**John Jerry Gordon-Mensah**
CS Student | AI/Data Engineer
[github.com/Gordon-Mensah](https://github.com/Gordon-Mensah)

*Internship experience: Quantium (category analytics), British Airways (NLP/sentiment), Accenture Hungary (predictive modelling, Power BI)*

---

## 📄 License

MIT License — free to use, adapt, and build on.