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


def add_financial_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create predictor-only financial ratios.

    The target is intentionally not used in any engineered predictor,
    preventing target leakage.
    """
    df = panel.copy()
    eps = 1e-9

    capital = df["Total capital"].abs() + eps
    df["Profit_to_Capital"] = df["Profit after tax"] / capital
    df["Reserves_to_Capital"] = df["Reserves and funds"] / capital
    df["Liabilities_to_Capital"] = df["Current liabilities & provisions"] / capital
    df["Deposits_to_Capital"] = df["Deposits (accepted by commercial banks)"] / capital
    df["OtherIncome_to_Capital"] = df["Other income"] / capital

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_year_over_year_growth(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Add lag-based growth features using only information available at forecast time."""
    out = current.copy()
    eps = 1e-9

    for col in FEATURES:
        out[f"{col} YoY Growth"] = (
            (current[col] - previous[col]) /
            (previous[col].abs() + eps)
        )

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
