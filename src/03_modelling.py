"""
LLM-Powered Customer Churn Intelligence System
================================================
Step 3: Model Training, Evaluation & SHAP Explainability
Models: XGBoost (primary), Logistic Regression (baseline), Random Forest

Author: John Jerry Gordon-Mensah
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util, os as _os
_spec = importlib.util.spec_from_file_location("feature_engineering", _os.path.join(_os.path.dirname(__file__), "02_feature_engineering.py"))
_fe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fe)
engineer_features = _fe.engineer_features; get_model_features = _fe.get_model_features


def load_and_prepare(path="data/telco_churn_features.csv"):
    df = pd.read_csv(path)
    features = get_model_features()
    X = df[features].fillna(0)
    y = df["churn"]
    return X, y, df


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=3,  # Handle class imbalance
            random_state=42, eval_metric="logloss", verbosity=0
        )
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    trained = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model
        print(f"   ✓ Trained: {name}")

    return trained, scaler


def evaluate_models(trained_models, scaler, X_test, y_test):
    results = {}
    X_test_scaled = scaler.transform(X_test)

    for name, model in trained_models.items():
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "auc_roc": round(auc, 4),
            "avg_precision": round(ap, 4),
            "precision_churn": round(report["1"]["precision"], 4),
            "recall_churn": round(report["1"]["recall"], 4),
            "f1_churn": round(report["1"]["f1-score"], 4),
        }

        print(f"\n── {name} ──")
        print(f"   AUC-ROC:         {auc:.4f}")
        print(f"   Avg Precision:   {ap:.4f}")
        print(f"   Precision (churn): {report['1']['precision']:.4f}")
        print(f"   Recall (churn):    {report['1']['recall']:.4f}")
        print(f"   F1 (churn):        {report['1']['f1-score']:.4f}")

    return results


def compute_shap_values(model, X_test, feature_names):
    """Compute SHAP values for XGBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP values = feature importance
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    return importance, shap_values, explainer.expected_value


def add_churn_scores_to_df(df, model, scaler, features):
    """Add churn probability scores to full dataframe for the dashboard."""
    X = df[features].fillna(0)
    df["churn_probability"] = model.predict_proba(X)[:, 1].round(4)
    df["churn_risk_label"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return df


if __name__ == "__main__":
    print("🔧 Loading and preparing data...")
    X, y, df = load_and_prepare("data/telco_churn_features.csv")

    print(f"   Dataset: {X.shape[0]} customers, {X.shape[1]} features")
    print(f"   Churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n🔧 Training models...")
    trained_models, scaler = train_models(X_train, y_train)

    print("\n📊 Evaluation Results:")
    results = evaluate_models(trained_models, scaler, X_test, y_test)

    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # SHAP analysis on XGBoost
    print("\n🔍 Computing SHAP values for XGBoost...")
    xgb_model = trained_models["XGBoost"]
    shap_importance, shap_vals, base_val = compute_shap_values(
        xgb_model, X_test, get_model_features()
    )
    shap_importance.to_csv("data/shap_importance.csv", index=False)
    print("   Top 5 features by SHAP importance:")
    print(shap_importance.head(5).to_string(index=False))

    # Add scores to full dataset for dashboard
    print("\n💾 Adding churn scores to full dataset...")
    df_orig = pd.read_csv("data/telco_churn_features.csv")
    df_scored = add_churn_scores_to_df(df_orig, xgb_model, scaler, get_model_features())
    df_scored.to_csv("data/telco_churn_scored.csv", index=False)

    # Save model
    with open("data/xgb_model.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "scaler": scaler, "features": get_model_features()}, f)

    print("\n✅ All done!")
    print("   → data/model_results.json")
    print("   → data/shap_importance.csv")
    print("   → data/telco_churn_scored.csv")
    print("   → data/xgb_model.pkl")