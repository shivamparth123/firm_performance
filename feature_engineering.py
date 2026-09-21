import numpy as np
import pandas as pd

FEATURES = [
    "Other income",
    "Profit after tax",
    "Total capital",
    "Reserves and funds",
    "Deposits (accepted by commercial banks)",
    "Current liabilities & provisions",
]
TARGET = "Income from financial services"


def add_financial_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    eps = 1e-9

    df["Profit_to_Capital"] = df["Profit after tax"] / (df["Total capital"].abs() + eps)
    df["Reserves_to_Capital"] = df["Reserves and funds"] / (df["Total capital"].abs() + eps)
    df["Liabilities_to_Capital"] = df["Current liabilities & provisions"] / (df["Total capital"].abs() + eps)
    df["OtherIncome_to_Target"] = df["Other income"] / (df["Target"].abs() + eps)
    df["Deposits_to_Capital"] = df["Deposits (accepted by commercial banks)"] / (df["Total capital"].abs() + eps)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_year_over_year_growth(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    out = current.copy()
    eps = 1e-9
    for col in FEATURES:
        out[f"{col} YoY Growth"] = (
            (current[col] - previous[col]) / (previous[col].abs() + eps)
        )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
