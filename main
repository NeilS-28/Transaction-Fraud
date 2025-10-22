# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

st.set_page_config(page_title="Fraud Detection Model Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def prepare_data(df, target_col="Class", test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=random_state, stratify=y_res
    )
    return X_train, X_test, y_train, y_test, X.columns

def train_models(X_train, y_train, random_state=42):
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        criterion="entropy",
        class_weight="balanced",
        random_state=random_state,
    )
    dt.fit(X_train, y_train)
    models["Decision Tree"] = dt

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model

    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }

    try:
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
    except Exception:
        fpr, tpr, roc_auc = None, None, None

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return metrics, cm, (fpr, tpr, roc_auc), y_pred, y_prob, report

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def plot_roc_curves(roc_curves_dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    any_curve = False
    for name, v in roc_curves_dict.items():
        if v is None:
            continue
        fpr, tpr, roc_auc = v
        if fpr is not None and tpr is not None and roc_auc is not None:
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
            any_curve = True
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    if any_curve:
        ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

def plot_feature_importance(model, feature_names, title):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1])
        ax.set_title(title)
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    elif hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        order = np.argsort(np.abs(coefs))[::-1][:15]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(np.array(feature_names)[order][::-1], coefs[order][::-1])
        ax.set_title(title + " (Top Coefficients)")
        ax.set_xlabel("Coefficient")
        st.pyplot(fig)
    else:
        st.info("No feature importance or coefficients available for this model.")

def show_prediction_samples(X_test, y_test, preds, probs, n=10):
    df = pd.DataFrame(X_test).copy()
    df["Actual"] = np.array(y_test)
    df["Predicted"] = preds
    if probs is not None:
        df["Fraud_Prob"] = probs
    st.dataframe(df.head(n))

def main():
    st.title("Transaction Fraud Detection – Model Comparison Dashboard")
    st.write("Upload creditcard.csv or keep default path. Then train, visualize, and compare four models.")

    # Sidebar
    st.sidebar.header("Data & Settings")
    uploaded = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])
    default_path = st.sidebar.text_input("Or path to creditcard.csv", "creditcard.csv")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = load_data(default_path)

    st.subheader("Data preview")
    st.dataframe(df.head(), use_container_width=True)

    # Train once across all models with consistent split/SMOTE
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        df, target_col="Class", test_size=test_size, random_state=random_state
    )

    with st.spinner("Training models..."):
        models = train_models(X_train, y_train, random_state=random_state)

    # Evaluate all
    results = {}
    for name, model in models.items():
        metrics, cm, roc_data, y_pred, y_prob, report = evaluate_model(model, X_test, y_test)
        results[name] = {
            "metrics": metrics,
            "cm": cm,
            "roc": roc_data,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": report,
            "model": model
        }

    # Tabs
    tabs = st.tabs(["Overview", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"])

    # Overview
    with tabs[0]:
        st.subheader("Metrics comparison")
        cmp = []
        for name, r in results.items():
            m = r["metrics"]
            cmp.append({
                "Model": name,
                "Accuracy": round(m["Accuracy"], 4),
                "Precision": round(m["Precision"], 4),
                "Recall": round(m["Recall"], 4),
                "F1": round(m["F1"], 4)
            })
        cmp_df = pd.DataFrame(cmp).sort_values("F1", ascending
