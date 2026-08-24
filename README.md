# Financial Performance Forecasting using Machine Learning

A machine learning study for forecasting subsequent-year financial performance of firms and banks using historical financial indicators.

## Overview

Financial performance is influenced by multiple indicators such as profitability, capital, reserves, deposits, liabilities, and income from financial services. This project investigates whether information from previous years can be used to forecast performance in a subsequent year.

The dataset contains financial information for **213 firms/banks** across **seven yearly blocks** of financial indicators. The notebook converts the repeated yearly structure into a supervised forecasting problem and compares regression models using RMSE.

## Objectives

- Organize multi-year firm-level financial data into a supervised learning format.
- Use historical financial indicators to forecast a subsequent year's target.
- Compare regression approaches using a time-ordered evaluation strategy.
- Measure forecasting error using **Root Mean Squared Error (RMSE)**.

## Dataset

The dataset is available at `data/7_year.csv` and contains **213 rows and 50 columns**. Each yearly block contains:

- Other income
- Profit after tax
- Total capital
- Reserves and funds
- Deposits
- Current liabilities & provisions
- Income from financial services

The current implementation uses **Income from financial services** as the forecasting target. Historical yearly indicators are used to construct the feature/target transitions.

## Methodology

```text
Raw multi-year financial data
            ↓
Data loading and organization
            ↓
Year-wise feature / target construction
            ↓
Time-ordered train/test evaluation
            ↓
Regression models
            ↓
RMSE comparison
```

### Models

The current notebook implements:

1. **Linear Regression**
2. **Support Vector Regression (SVR) with a linear kernel**

The models learn from one year's financial indicators and evaluate prediction on a subsequent year's observations, preserving the chronological structure of the data.

## Results

The original project evaluation reported the following final RMSE values:

| Model | RMSE |
|---|---:|
| Linear Regression | **0.01655** |
| Linear SVR | **0.05370** |

Lower RMSE indicates lower prediction error on the evaluated target.

> These values are dataset-specific experimental results. They should not be interpreted as a guarantee of performance on unseen financial data or future market conditions.

## Project Structure

```text
firm_performance/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.ipynb
│
├── data/
│   └── 7_year.csv
│
└── notebooks/
    └── README.md
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

### 3. Run the notebook

Open `main.ipynb` with Jupyter Notebook/JupyterLab or upload it to Google Colab.

The notebook first looks for:

```text
data/7_year.csv
```

and also contains a `/content/7_year.csv` fallback for the original Colab workflow.

## Key Learning Outcomes

- Structured multi-year tabular financial data for machine learning.
- Converted historical observations into supervised learning features and targets.
- Preserved chronological ordering during year-to-year evaluation.
- Compared regression models using RMSE.
- Used Python data-analysis and machine-learning libraries to build a reproducible notebook workflow.

## Limitations and Future Improvements

The current implementation is an experimental forecasting study. Potential improvements include:

- Walk-forward / rolling-window validation for more robust forecasting evaluation.
- Additional models such as Random Forest, Gradient Boosting, XGBoost, and regularized regression.
- MAE and R² alongside RMSE.
- Feature-importance and sensitivity analysis.
- More systematic missing-value and outlier handling.
- Evaluation across multiple forecast horizons and subsets of firms.

## Author

**Shivam Kumar**  
B.Tech, Engineering Science  
Indian Institute of Technology Jodhpur

- GitHub: https://github.com/shivamparth123
- LinkedIn: https://www.linkedin.com/in/shivam-kumar-338988294/
