
---

# Credit Model Project

This project focuses on credit risk modeling using two main approaches:

- A **traditional scorecard model** (Part A) inspired by FICO and VantageScore methodologies, emphasizing interpretability and statistical rigor.
- A set of **advanced machine learning models** (Part C), implemented using object-oriented programming for modularity and scalability.

The objective is to predict the likelihood that a loan applicant will default based on financial and behavioral variables.

---

## Project Structure

```
src/
├── Part A/
│   ├── data/
│   │   ├── cs-training.csv
│   │   └── cs-test.csv
│   ├── exploratory_data_analysis.ipynb
│   ├── scorecard_model.py
│   └── test_multiple_thresholds.py
│
└── Part C/
    ├── data/
    │   ├── cs-training.csv
    │   ├── cs-test.csv
    │   └── cs-test-predicted.csv
    ├── credit_risk_model.py
    └── logistic_model.pkl
```


---

## Dataset Description

The data used in this project is sourced from the ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) competition on Kaggle. The dataset aims to improve credit scoring models by predicting the probability that an individual will experience financial distress in the next two years.

### Data Files:
- **cs-training.csv**: Training dataset containing labeled data.  
- **cs-test.csv**: Test dataset without labels, used for model evaluation.  
- **cs-test-predicted.csv**: Output file with predictions on the test set.

### Features:
The dataset includes the following variables:

- **SeriousDlqin2yrs**: Binary target variable; `1` indicates the person experienced 90 days past due delinquency or worse, `0` otherwise.
- **RevolvingUtilizationOfUnsecuredLines**: Total balance on credit cards and personal lines of credit divided by the sum of their credit limits.
- **age**: Age of the borrower in years.
- **NumberOfTime30-59DaysPastDueNotWorse**: Number of times the borrower has been 30–59 days past due but not worse in the last two years.
- **DebtRatio**: Monthly debt payments, alimony, living costs divided by monthly gross income.
- **MonthlyIncome**: Monthly income of the borrower.
- **NumberOfOpenCreditLinesAndLoans**: Number of open loans (installment loans like car loans or mortgages) and lines of credit (e.g., credit cards).
- **NumberOfTimes90DaysLate**: Number of times the borrower has been 90 days or more past due.
- **NumberRealEstateLoansOrLines**: Number of mortgage and real estate loans, including home equity lines of credit.
- **NumberOfTime60-89DaysPastDueNotWorse**: Number of times the borrower has been 60–89 days past due but not worse in the last two years.
- **NumberOfDependents**: Number of dependents in the family, excluding the borrower (e.g., spouse, children).

For more details, refer to the [competition's data page](https://www.kaggle.com/c/GiveMeSomeCredit/data).

## Part A: Traditional Scorecard Model

### Key Components:

#### `exploratory_data_analysis.ipynb`
- Performs comprehensive exploratory data analysis (EDA) on the training set.
- Highlights variable distributions, missing values, and key relationships with the target variable.
- Includes visualizations (histograms, boxplots, bar charts) and summary statistics.

#### `scorecard_model.py`
- Prepares the dataset by handling missing values, outliers, and categorical variables.
- Generates a **scorecard**, mapping coefficients to point scores on a scale similar to FICO (e.g., 300–900).
- Outputs model performance metrics and final score distributions.

#### `test_multiple_thresholds.py`
- Evaluates the model using different threshold cutoffs (e.g., 800, 850, 900).
- Reports performance metrics at each threshold: accuracy, precision, recall, F1-score.
- Useful for business scenario analysis (e.g., high-risk vs. low-risk applicants).

---

## Part C: Advanced ML Models (OOP-based)

This module introduces a more modular and scalable approach to credit risk modeling using **Object-Oriented Programming (OOP)**.

### Key Components:

#### `credit_risk_model.py`
- Implements an **OOP framework** with class definitions for:
  - Data loading and preprocessing
  - Model training
  - Evaluation
  - Exporting predictions
- Supports multiple model types:
  - **Logistic Regression**
  - **Random Forest**
  - **Multi-layer Perceptron (Neural Network)**
- Includes cross-validation and hyperparameter tuning.
- Saves predictions to `cs-test-predicted.csv` and serialized model to `logistic_model.pkl`.
- Evaluates models using ROC AUC, classification reports, and confusion matrices.

#### `logistic_model.pkl`
- Pre-trained logistic regression model stored with `joblib`.
- Can be reloaded for inference or benchmarking.

---

## Installation & Usage

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the scripts:
```bash
# Scorecard model
python src/Part\ A/scorecard_model.py

# Evaluate using multiple thresholds
python src/Part\ A/test_multiple_thresholds.py

# Advanced ML models (OOP)
python src/Part\ C/credit_risk_model.py
```

---

## Outputs

- **Scorecard scores** assigned to each customer (Part A)
- **Model predictions and metrics** (AUC, precision, recall, F1)
- **Predicted CSV file**: `cs-test-predicted.csv`
- **Trained model file**: `logistic_model.pkl`
- **EDA insights** via notebook
- **different types of plots**

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, seaborn, matplotlib, joblib

Optional (for EDA):
- Jupyter Notebook

---
