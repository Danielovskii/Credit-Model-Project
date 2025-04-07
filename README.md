# Credit Model Project

A credit risk modeling project that combines traditional **scorecard techniques** with more advanced machine learning models such as **neural networks**, **logistic regression**, and **random forests**.

---

## Project Structure

```
src/
├── Part A/                      # Traditional scorecard model
│   ├── data/
│   │   ├── cs-training.csv
│   │   └── cs-test.csv
│   ├── exploratory_data_analysis.ipynb
│   ├── scorecard_model.py
│   └── test_multiple_thresholds.py
│
└── Part C/                      # Advanced models
    ├── data/
    │   ├── cs-training.csv
    │   ├── cs-test.csv
    │   └── cs-test-predicted.csv
    ├── credit_risk_model.py
    └── logistic_model.pkl       # Trained logistic model
```

---

## Part A: Traditional Scorecard

Builds a scorecard model using statistical techniques inspired by FICO/VantageScore, aiming for interpretability and practical use in real-world banking environments.

### Key Files:
- `exploratory_data_analysis.ipynb`: Exploratory data analysis and visualizations.
- `scorecard_model.py`: Scorecard training using all the variables
- `test_multiple_thresholds.py`: Evaluates model performance across different decision thresholds (800, 850, 900), reporting precision, recall, and F1-score.

---

## Part C: Advanced Models

Implements and compares machine learning models for credit risk, including:
- **Logistic Regression**
- **Neural Networks**
- **Random Forests**

### Key Files:
- `credit_risk_model.py`: Trains and evaluates advanced models using the same dataset as Part A.
- `logistic_model.pkl`: Trained logistic regression model saved for prediction purposes.

---

## Installation & Usage

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run any script as needed:
```bash
# Run the scorecard model
python src/Part\ A/scorecard_model.py

# Evaluate using multiple thresholds
python src/Part\ A/test_multiple_thresholds.py

# Train advanced ML models
python src/Part\ C/credit_risk_model.py
```

---

## Expected Outputs

The scripts will generate:
- Model performance metrics (AUC, accuracy, precision, recall, F1-score)
- CSV output of predictions (`cs-test-predicted.csv`)
- Insights and visualizations from EDA (available in the notebook)

---

## Requirements

- Python 3.8+
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`, `joblib`
- (Optional) Jupyter Notebook

---

## Additional Notes

- The dataset (`cs-training.csv`, `cs-test.csv`) is synthetic and used for educational or prototyping purposes.
- All scripts can be adapted for real-world financial data in compliance with local regulations.

---
