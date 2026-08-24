# Financial Performance Forecasting using Machine Learning

A machine learning study for forecasting the next-year financial performance of firms and banks using historical financial indicators.

## Overview

Financial performance is influenced by multiple indicators such as profitability, capital, reserves, deposits, liabilities, and income from financial services. This project investigates whether information from previous years can be used to forecast a firm's performance in a subsequent year.

The dataset contains financial information for **213 firms/banks** across multiple years. The data is organized into repeated yearly groups of financial indicators, which are transformed into year-wise feature and target sets before modeling.

## Objectives

- Organize multi-year firm-level financial data into a supervised learning format.
- Use historical financial indicators to forecast subsequent-year performance.
- Compare different regression approaches using a time-ordered evaluation strategy.
- Measure forecasting error using **Root Mean Squared Error (RMSE)**.

## Dataset

The repository contains the dataset at:

```text
 data/7_year.csv
```

The dataset contains **213 rows and 50 columns**. Each firm has repeated yearly financial indicators including:

- Other income
- Profit after tax
- Total capital
- Reserves and funds
- Deposits
- Current liabilities and provisions
- Income from financial services

The notebook restructures these repeated yearly columns so that financial information from an earlier year can be used to predict the target performance in a later year.

## Methodology

The workflow implemented in `main.ipynb` is:

```text
Raw multi-year financial data
            ↓
Data loading and column organization
            ↓
Year-wise feature / target construction
            ↓
Time-ordered train/test evaluation
            ↓
Regression models
            ↓
RMSE comparison
```

### Models currently implemented

The current notebook compares:

1. **Linear Regression**
2. **Support Vector Regression (SVR) with a linear kernel**

The models are trained using historical yearly financial indicators and evaluated on subsequent-year observations.

## Results

The current notebook reports the following RMSE values for the final evaluation:

| Model | RMSE |
|---|---:|
| Linear Regression | **0.01655** |
| Linear SVR | **0.05370** |

Lower RMSE indicates lower prediction error on the evaluated target.

> These results correspond to the current implementation in `main.ipynb`. They should not be interpreted as a guarantee of performance on unseen datasets or future financial conditions.

## Project Structure

```text
firm_performance/
│
├── README.md
├── requirements.txt
├── main.ipynb
│
└── data/
    └── 7_year.csv
```

## Technologies

- **Python**
- **Pandas** – data loading and manipulation
- **NumPy** – numerical computation
- **Scikit-learn** – regression models and evaluation
- **Matplotlib** – visualization
- **Jupyter Notebook / Google Colab** – experimentation and analysis

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/shivamparth123/firm_performance.git
cd firm_performance
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open `main.ipynb` using Jupyter Notebook/JupyterLab or upload it to Google Colab.

If running the notebook locally, make sure the dataset path points to:

```text
data/7_year.csv
```

## Key Learning Outcomes

Through this project, I worked on:

- Structuring multi-year tabular financial data for machine learning.
- Converting historical observations into supervised learning features and targets.
- Designing time-aware train/test comparisons instead of treating all years as independent samples.
- Comparing regression models using an appropriate error metric.
- Interpreting model performance in the context of financial forecasting.

## Limitations and Future Improvements

The current implementation is an experimental forecasting study and has several areas for improvement:

- Add walk-forward / rolling-window validation for more robust time-series evaluation.
- Compare additional models such as Random Forest, Gradient Boosting, XGBoost, and regularized regression.
- Add MAE and R² alongside RMSE.
- Perform feature-importance and sensitivity analysis.
- Improve data preprocessing and handling of missing or unusual financial values.
- Evaluate performance across different forecast horizons and subsets of firms.

## Author

**Shivam Kumar**  
B.Tech, Engineering Science  
Indian Institute of Technology Jodhpur

- GitHub: https://github.com/shivamparth123
- LinkedIn: https://www.linkedin.com/in/shivam-kumar-338988294/
