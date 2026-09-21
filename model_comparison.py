"""
Improved financial performance forecasting pipeline.

Uses expanding-window time-aware validation and compares:
Linear Regression, Ridge, Lasso, SVR-RBF, Random Forest,
Extra Trees, and Gradient Boosting.

The final Year 6 -> Year 7 target block is audited separately because
the supplied CSV contains an exact duplicate target block.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = [
    "Other income",
    "Profit after tax",
    "Total capital",
    "Reserves and funds",
    "Deposits (accepted by commercial banks)",
    "Current liabilities & provisions",
]
TARGET = "Income from financial services"
RANDOM_STATE = 42


def load_years(path="data/7_year.csv"):
    df = pd.read_csv(path)
    years = []
    for year in range(7):
        suffix = "" if year == 0 else f".{year}"
        cols = FEATURES + [TARGET]
        block = df[["Company Name"] + [c + suffix for c in cols]].copy()
        block.columns = ["Company Name"] + cols
        block["YearIndex"] = year + 1
        years.append(block)
    return df, years


def build_transitions(years):
    rows = []
    for t in range(6):
        current, nxt = years[t], years[t + 1]
        x = current[FEATURES].copy()
        x["Company Name"] = current["Company Name"].values
        x["Forecast From Year"] = t + 1
        x["Target Year"] = t + 2
        x["Target"] = nxt[TARGET].values
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def get_models():
    return {
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()), ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=20000))
        ]),
        "SVR (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.01))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=1.0,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.03, max_depth=2,
            loss="huber", random_state=RANDOM_STATE
        ),
    }


def walk_forward(panel, models):
    results = []
    # Year 6 -> Year 7 is excluded from the main score because its target
    # is an exact duplicate of Year 5 -> Year 6 in the supplied dataset.
    for test_target_year in [3, 4, 5, 6]:
        train = panel[panel["Target Year"] < test_target_year]
        test = panel[panel["Target Year"] == test_target_year]
        for name, model in models.items():
            fitted = clone(model)
            fitted.fit(train[FEATURES], train["Target"])
            pred = fitted.predict(test[FEATURES])
            results.append({
                "Test transition": f"Year {test_target_year-1} -> Year {test_target_year}",
                "Model": name,
                "RMSE": mean_squared_error(test["Target"], pred) ** 0.5,
                "MAE": mean_absolute_error(test["Target"], pred),
                "R2": r2_score(test["Target"], pred),
            })
    return pd.DataFrame(results)


def audit_final_target(years):
    a = years[5][TARGET].to_numpy()
    b = years[6][TARGET].to_numpy()
    return {
        "exact_duplicate": bool(np.array_equal(a, b)),
        "matching_share": float(np.mean(np.isclose(a, b))),
        "max_absolute_difference": float(np.max(np.abs(a - b))),
    }


if __name__ == "__main__":
    df, years = load_years()
    panel = build_transitions(years)
    models = get_models()

    print(f"Dataset: {df.shape[0]} firms, {df.shape[1]} columns")
    print("Final target audit:", audit_final_target(years))

    results = walk_forward(panel, models)
    summary = results.groupby("Model")[["RMSE", "MAE", "R2"]].mean().sort_values("RMSE")
    print("\nWalk-forward model comparison:")
    print(summary.to_string())
