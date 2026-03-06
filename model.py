"""
model.py  –  Predictive model for the Universal Bank Dashboard.
Trains a Random Forest classifier and returns evaluation artefacts.
Results are cached so the model trains only once per Streamlit session.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve,
)
import streamlit as st


FEATURES = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "Securities_Account",
    "CD_Account", "Online", "CreditCard",
]

FEATURE_LABELS = {
    "Income":              "Income",
    "CCAvg":               "CC Avg Spend",
    "Education":           "Education",
    "Family":              "Family Size",
    "CD_Account":          "CD Account",
    "Mortgage":            "Mortgage",
    "Experience":          "Experience",
    "Age":                 "Age",
    "Online":              "Online Banking",
    "CreditCard":          "Credit Card",
    "Securities_Account":  "Securities Acct",
}


@st.cache_data(show_spinner="Training predictive model…")
def train_model(df: pd.DataFrame):
    """
    Train Random Forest and return a results dict.
    Cached so it only runs once per session.
    """
    X = df[FEATURES].copy()
    y = df["Personal_Loan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred,
                                target_names=["Rejected", "Accepted"],
                                output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    importances = pd.Series(
        rf.feature_importances_,
        index=FEATURES
    ).rename(index=FEATURE_LABELS).sort_values(ascending=False)

    return dict(
        model=rf,
        accuracy=acc,
        auc=auc,
        cm=cm,
        report=cr,
        fpr=fpr,
        tpr=tpr,
        importances=importances,
        feature_names=FEATURES,
    )


def predict_single(model_results: dict, input_dict: dict) -> dict:
    """
    Run the trained model on a single customer's feature dict.
    Returns probability and binary prediction.
    """
    rf    = model_results["model"]
    names = model_results["feature_names"]
    X     = pd.DataFrame([{k: input_dict.get(k, 0) for k in names}])
    prob  = rf.predict_proba(X)[0, 1]
    pred  = int(prob >= 0.5)
    return {"probability": prob, "prediction": pred}
