"""
data_loader.py  –  Universal Bank Dashboard helper
Loads and preprocesses the UniversalBank dataset.
"""

import os
import pandas as pd
import numpy as np


# ── Column name map (handles both space & underscore variants) ──────────────
COL_MAP = {
    "ZIP Code": "ZIP_Code",
    "Personal Loan": "Personal_Loan",
    "Securities Account": "Securities_Account",
    "CD Account": "CD_Account",
}

FEATURES = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "Securities_Account",
    "CD_Account", "Online", "CreditCard",
]

EDU_LABELS  = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}
LOAN_LABELS = {0: "Rejected", 1: "Accepted"}


def load_data() -> pd.DataFrame:
    """Return cleaned Universal Bank dataframe."""
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "UniversalBank.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "UniversalBank.csv not found next to app.py. "
            "Please add the file to the project folder."
        )

    df = pd.read_csv(csv_path)
    df.rename(columns=COL_MAP, inplace=True)
    # Normalise any remaining spaces
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Fix negative Experience values (data quirk)
    df["Experience"] = df["Experience"].clip(lower=0)

    # Derived columns
    df["Education_Label"] = df["Education"].map(EDU_LABELS)
    df["Loan_Status"]     = df["Personal_Loan"].map(LOAN_LABELS)

    income_bins   = [0, 50, 100, 150, 200, 300]
    income_labels = ["<$50k", "$50–100k", "$100–150k", "$150–200k", "$200k+"]
    df["Income_Group"] = pd.cut(
        df["Income"], bins=income_bins, labels=income_labels
    )

    age_bins   = [20, 30, 40, 50, 60, 70]
    age_labels = ["20s", "30s", "40s", "50s", "60s"]
    df["Age_Group"] = pd.cut(
        df["Age"], bins=age_bins, labels=age_labels
    )

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Pre-compute key summary numbers for KPI cards."""
    total      = len(df)
    accepted   = df["Personal_Loan"].sum()
    rate       = accepted / total * 100
    avg_income = df["Income"].mean()
    avg_ccavg  = df["CCAvg"].mean()
    avg_mort   = df["Mortgage"].mean()
    cd_rate    = df[df["CD_Account"] == 1]["Personal_Loan"].mean() * 100
    no_cd_rate = df[df["CD_Account"] == 0]["Personal_Loan"].mean() * 100
    return dict(
        total=total,
        accepted=accepted,
        rate=rate,
        avg_income=avg_income,
        avg_ccavg=avg_ccavg,
        avg_mort=avg_mort,
        cd_rate=cd_rate,
        no_cd_rate=no_cd_rate,
    )
