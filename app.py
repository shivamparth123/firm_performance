import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from model_comparison import load_years, build_transitions, get_models, FEATURES, TARGET
from feature_engineering import add_financial_features

st.set_page_config(page_title="Firm Performance Analytics", layout="wide")
st.title("Firm Performance Analytics & Forecasting")
st.caption("Time-aware financial forecasting with model comparison and diagnostics.")

path = Path("data/7_year.csv")
if not path.exists():
    st.error("data/7_year.csv was not found.")
    st.stop()

df, years = load_years(path)
panel = build_transitions(years)
feature_panel = add_financial_features(panel)

col1, col2, col3 = st.columns(3)
col1.metric("Firms", len(df))
col2.metric("Financial indicators / year", len(FEATURES) + 1)
col3.metric("Forecast transitions", 6)

st.subheader("Dataset overview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Company analysis")
company = st.selectbox("Select a company", sorted(df["Company Name"].dropna().unique()))
row = df[df["Company Name"] == company].iloc[0]

history = []
for year in range(7):
    suffix = "" if year == 0 else f".{year}"
    history.append({
        "Year": year + 1,
        TARGET: row[TARGET + suffix],
        "Profit after tax": row["Profit after tax" + suffix],
        "Deposits": row["Deposits (accepted by commercial banks)" + suffix],
    })
history = pd.DataFrame(history)

st.line_chart(history.set_index("Year")[[TARGET, "Profit after tax", "Deposits"]])
st.dataframe(history, use_container_width=True)

st.subheader("Model comparison")
models = get_models()
model_name = st.selectbox("Model", list(models.keys()))
model = models[model_name]

# Train on all clean historical transitions and demonstrate the latest
# available non-duplicated transition.
train = feature_panel[feature_panel["Target Year"] <= 5]
test = feature_panel[feature_panel["Target Year"] == 6]

base_features = FEATURES + [
    "Profit_to_Capital", "Reserves_to_Capital",
    "Liabilities_to_Capital", "OtherIncome_to_Target",
    "Deposits_to_Capital"
]
model.fit(train[base_features], train["Target"])
prediction = model.predict(test[base_features])

company_mask = test["Company Name"] == company
if company_mask.any():
    actual = test.loc[company_mask, "Target"].iloc[0]
    pred = prediction[test.index.get_loc(test.index[company_mask][0])]
    c1, c2 = st.columns(2)
    c1.metric("Actual target", f"{actual:,.4f}")
    c2.metric("Model prediction", f"{pred:,.4f}")

st.info("The supplied dataset contains an exact duplicate final target block. The primary evaluation therefore excludes that duplicated transition from model scoring.")
