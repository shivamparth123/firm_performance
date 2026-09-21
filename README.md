# Firm Performance Analytics & Forecasting

An end-to-end machine-learning project for forecasting **next-year income from financial services** from historical firm/bank financial indicators.

## Why this project?

Instead of treating the dataset as ordinary tabular regression, the project preserves its **yearly structure** and evaluates models with time-aware validation. It also performs data-quality checks before reporting performance.

### Dataset

- 213 firms/banks
- 7 yearly blocks
- 7 financial indicators per year
- Target: **Income from financial services**

Core predictors include:

- Other income
- Profit after tax
- Total capital
- Reserves and funds
- Deposits accepted by commercial banks
- Current liabilities & provisions

## Pipeline

```
Multi-year financial data
        ↓
Data-quality audit
        ↓
Year t → Year t+1 transformations
        ↓
Financial feature engineering
        ↓
Expanding-window validation
        ↓
Model comparison
        ↓
RMSE / MAE / R²
        ↓
Diagnostics + dashboard
```

## Models

The improved pipeline compares seven complementary regressors:

1. Linear Regression — interpretable baseline
2. Ridge — regularized linear model
3. Lasso — regularization and feature selection
4. SVR (RBF) — nonlinear kernel model
5. Random Forest — bagged nonlinear trees
6. Extra Trees — randomized tree ensemble
7. Gradient Boosting — sequential nonlinear ensemble

The goal is **not** to maximize the number of algorithms. Models are included because they represent different assumptions and useful baselines.

## Validation

The primary evaluation uses **expanding-window walk-forward validation** rather than random train/test splitting.

For example:

```
Train early transitions → test next transition
Train more history     → test next transition
Train more history     → test next transition
...
```

This prevents future-year observations from being used to evaluate earlier periods.

Metrics:

- **RMSE** — penalizes larger errors
- **MAE** — average absolute error
- **R²** — variance explained

A simple "last observed value" baseline should also be compared before claiming that an ML model adds predictive value.

## Important data-quality finding

The supplied CSV contains an exact duplicate for the final target block:

```
Income from financial services.5
=
Income from financial services.6
```

for the supplied firms.

Because this can make the final transition misleading, the primary model-comparison score excludes that duplicated target transition. The project reports the data-quality finding instead of presenting artificially easy performance as genuine forecasting evidence.

## Feature engineering

The project derives predictor-only financial ratios such as:

- Profit / Capital
- Reserves / Capital
- Liabilities / Capital
- Deposits / Capital

Future target information is deliberately **not** used to construct predictors.

## Dashboard

A Streamlit dashboard is included for:

- Firm selection
- Historical financial trends
- Model selection
- Forecast inspection
- Dataset overview
- Data-quality disclosure

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
firm_performance/
├── README.md
├── requirements.txt
├── main.ipynb
├── model_comparison.py
├── feature_engineering.py
├── app.py
└── data/
    └── 7_year.csv
```

## Reproducibility

```bash
git clone https://github.com/shivamparth123/firm_performance.git
cd firm_performance
pip install -r requirements.txt
```

Then run the model pipeline:

```bash
python model_comparison.py
```

or launch the dashboard:

```bash
streamlit run app.py
```

## Limitations

The dataset is relatively small and contains a duplicated final target block. Therefore, results should be interpreted as an experimental study rather than evidence of deployable financial forecasting accuracy.

Further extensions could include:

- richer firm-level lag features
- nested time-series hyperparameter tuning
- uncertainty intervals
- SHAP-based explanations
- additional independent financial datasets
- external validation on a genuinely future period

## Author

**Shivam Kumar**  
B.Tech, Engineering Science  
Indian Institute of Technology Jodhpur

GitHub: https://github.com/shivamparth123
