# Traditional Credit Scoring Model for Personal Loans
# Authors: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Date: March 25, 2025
# Description: Hybrid scorecard with constraints,
# based on models like FICO and VantageScore.

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def check_eligibility(age, monthly_income, loan_amount):
    """Checks if the applicant meets the minimum requirements.
    
    Args:
        age (int): Applicant's age.
        monthly_income (float): Monthly income in USD.
        loan_amount (float): Requested loan amount in USD.
    
    Returns:
        bool: True if eligible, False otherwise.
    """
    min_age = 18
    min_income = 400  # $8,000 MXN ≈ $400 USD
    max_loan = 150000  # $3,000,000 MXN ≈ $150,000 USD
    
    if age < min_age or monthly_income < min_income or loan_amount > max_loan:
        return False
    return True

def repayment_history_score(late_payments):
    """Payment history score (maximum 350 points)."""
    if late_payments == 0:
        return 350
    elif late_payments == 1:
        return 200
    elif late_payments == 2:
        return 100
    else:
        return 0

def total_amount_owed_score(amount_owed):
    """Score for total amount owed (maximum 150 points)."""
    if amount_owed < 1000:
        return 150
    elif 1000 <= amount_owed < 5000:
        return 100
    elif 5000 <= amount_owed < 10000:
        return 75
    else:
        return 50

def credit_history_length_score(years):
    """Credit history length score (maximum 150 points)."""
    if years < 2:
        return 50
    elif 2 <= years < 5:
        return 75
    elif 5 <= years < 10:
        return 100
    else:
        return 150

def credit_types_score(num_types):
    """Credit mix score (maximum 100 points)."""
    if num_types == 1:
        return 60
    elif num_types == 2:
        return 80
    else:
        return 100

def new_credit_score(inquiries):
    """New credit inquiries score (maximum 100 points)."""
    if inquiries == 0:
        return 100
    elif inquiries == 1:
        return 80
    elif inquiries == 2:
        return 60
    else:
        return 40

def available_credit_score(credit_available):
    """Available credit score (maximum 100 points)."""
    if credit_available > 5000:
        return 100
    elif 2000 <= credit_available <= 5000:
        return 80
    elif 1000 <= credit_available < 2000:
        return 60
    else:
        return 40

def credit_utilization_score(usage_percent):
    """Credit utilization score (maximum 150 points)."""
    if usage_percent < 30:
        return 150
    elif 30 <= usage_percent < 50:
        return 100
    elif 50 <= usage_percent < 70:
        return 75
    else:
        return 50

def income_score(monthly_income):
    """Monthly income score (minimum $400 USD, maximum 100 points)."""
    if 400 <= monthly_income < 700:
        return 40
    elif 700 <= monthly_income < 1000:
        return 50
    elif 1000 <= monthly_income < 2000:
        return 60
    elif 2000 <= monthly_income < 3000:
        return 80
    else:
        return 100

def job_tenure_score(years):
    """Job tenure score (maximum 100 points)."""
    if years < 1:
        return 40
    elif 1 <= years < 3:
        return 60
    elif 3 <= years < 5:
        return 80
    else:
        return 100

def open_loans_score(num_loans):
    """Open loans score (maximum 100 points)."""
    if num_loans <= 1:
        return 100
    elif 2 <= num_loans <= 3:
        return 80
    elif 4 <= num_loans <= 5:
        return 60
    else:
        return 40

def prepare_applicant_data(row):
    """Transforms a dataset row into the format expected by the scorecard."""
    # Impute missing values
    monthly_income = row["MonthlyIncome"] if not pd.isna(row["MonthlyIncome"]) else row["MonthlyIncome"].median()
    
    # Variable mapping
    applicant_data = {
        "age": row["age"],
        "monthly_income": monthly_income,
        "loan_amount": 10000,  # Assume $10,000 for testing
        "late_payments": (
            row["NumberOfTime30-59DaysPastDueNotWorse"] +
            row["NumberOfTime60-89DaysPastDueNotWorse"] +
            row["NumberOfTimes90DaysLate"]
        ),
        "amount_owed": monthly_income * row["DebtRatio"] * 12,  # Estimate annual debt
        "credit_age": min(max(0, row["age"] - 18), 20),  # Limited to 20 years
        "credit_types": (
            2 if row["NumberRealEstateLoansOrLines"] > 0 and 
                row["NumberOfOpenCreditLinesAndLoans"] > row["NumberRealEstateLoansOrLines"] 
                else 1
        ),
        "inquiries": 0,  # Assume 0 inquiries
        "available_credit": 10000 * (1 - row["RevolvingUtilizationOfUnsecuredLines"]),  # Fixed limit of $10,000
        "credit_usage": row["RevolvingUtilizationOfUnsecuredLines"] * 100,  # Convert to percentage
        "job_tenure": 3,  # Assume 3 years
        "open_loans": row["NumberOfOpenCreditLinesAndLoans"]
    }
    return applicant_data

def evaluate_applicant(data):
    """Evaluates the applicant's eligibility and calculates total score."""
    # Check eligibility
    if not check_eligibility(data["age"], data["monthly_income"], data["loan_amount"]):
        return None, "Not eligible"
    
    # Calculate score
    total_score = (
        repayment_history_score(data["late_payments"]) +
        total_amount_owed_score(data["amount_owed"]) +
        credit_history_length_score(data["credit_age"]) +
        credit_types_score(data["credit_types"]) +
        new_credit_score(data["inquiries"]) +
        available_credit_score(data["available_credit"]) +
        credit_utilization_score(data["credit_usage"]) +
        income_score(data["monthly_income"]) +
        job_tenure_score(data["job_tenure"]) +
        open_loans_score(data["open_loans"])
    )
    
    threshold = 600  # Approval threshold
    decision = "Approved" if total_score >= threshold else "Rejected"
    return total_score, decision

def evaluate_dataset(data):
    """Evaluates entire dataset and compares with target variable."""
    scores = []
    decisions = []
    raw_scores = []  # For ROC curve we need numerical scores
    
    for _, row in data.iterrows():
        applicant_data = prepare_applicant_data(row)
        score, decision = evaluate_applicant(applicant_data)
        raw_score = score if score is not None else 0
        scores.append(raw_score)
        decisions.append(1 if decision == "Approved" else 0)  # 1 = Approved, 0 = Rejected/Not eligible
        raw_scores.append(raw_score)
    
    return scores, decisions, raw_scores

def plot_confusion_matrix(y_true, y_pred):
    """Visualizes confusion matrix."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    # Use seaborn for better visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'])
    
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    """Visualizes ROC curve."""
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

def main():
    """Main function to test the scorecard with dataset."""
    # Load the dataset
    data = pd.read_csv("data/cs-training.csv")

    # Clean the dataset (same preprocessing as in Part C)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    data = data[data["RevolvingUtilizationOfUnsecuredLines"] < 13]
    data = data[~data["NumberOfTimes90DaysLate"].isin([96, 98])]
    data = data[~data["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98])]
    data = data[~data["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])]
    debt_ratio_threshold = data["DebtRatio"].quantile(0.975)
    data = data[data["DebtRatio"] <= debt_ratio_threshold]

    # Impute missing values for MonthlyIncome (use median from full dataset)
    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(data["MonthlyIncome"].median())

    # Evaluate dataset
    scores, decisions, raw_scores = evaluate_dataset(data)

    # Compare with target variable (SeriousDlqin2yrs inverted: 1 = no default, 0 = default)
    y_true = 1 - data["SeriousDlqin2yrs"]  # Invert so that 1 = good client (should be Approved)
    y_pred = decisions

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, raw_scores)
    }

    # Print results
    print("=== Scorecard Evaluation Results ===")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")

    # Visualize confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)
    
    # Visualize ROC curve
    print("\nGenerating ROC curve...")
    plot_roc_curve(y_true, raw_scores)

    # Score statistics
    valid_scores = [s for s in scores if s > 0]
    print("\nScore Statistics:")
    print(f"Eligible Applicants: {len(valid_scores)}")
    print(f"Average Score: {np.mean(valid_scores):.2f}")
    print(f"Standard Deviation: {np.std(valid_scores):.2f}")
    print(f"Minimum Score: {np.min(valid_scores) if valid_scores else 0}")
    print(f"Maximum Score: {np.max(valid_scores) if valid_scores else 0}")

    # Generate score distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(valid_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=600, color='red', linestyle='--', label='Approval Threshold (600)')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.title('Credit Score Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    plt.show()

if __name__ == "__main__":
    main()