# Simplified and Adjusted Credit Scorecard Model Base
# Authors: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Date: March 25, 2025
# Description: Optimized scorecard implementation with enhanced validation,
#              performance metrics, and benchmark comparisons.

import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def check_eligibility(age, monthly_income):
    """Checks if the applicant meets minimum requirements."""
    min_age = 18
    min_income = 400  # $8,000 MXN ≈ $400 USD
    
    if age < min_age or monthly_income < min_income:
        return False
    return True

def repayment_history_score(late_payments):
    """Payment history score (max 350 points)."""
    if late_payments == 0:
        return 350
    elif late_payments == 1:
        return 150  # Smoothed
    elif late_payments == 2:
        return 50   # Smoothed
    else:
        return 0

def total_amount_owed_score(amount_owed):
    """Total debt amount score (max 150 points)."""
    if amount_owed < 1000:
        return 150
    elif 1000 <= amount_owed < 5000:
        return 100
    elif 5000 <= amount_owed < 10000:
        return 75
    else:
        return 50

def credit_history_length_score(years):
    """Credit history length score (max 150 points)."""
    if years < 2:
        return 50
    elif 2 <= years < 5:
        return 75
    elif 5 <= years < 10:
        return 100
    else:
        return 150

def credit_types_score(num_types):
    """Credit mix score (max 100 points)."""
    if num_types == 1:
        return 60
    elif num_types == 2:
        return 80
    else:
        return 100

def available_credit_score(credit_available):
    """Available credit score (max 100 points)."""
    if credit_available > 5000:
        return 100
    elif 2000 <= credit_available <= 5000:
        return 80
    elif 1000 <= credit_available < 2000:
        return 60
    else:
        return 40

def credit_utilization_score(usage_percent):
    """Credit utilization score (max 150 points)."""
    if usage_percent < 30:
        return 150
    elif 30 <= usage_percent < 50:
        return 50
    elif 50 <= usage_percent < 70:
        return 10
    else:
        return 0

def income_score(monthly_income):
    """Monthly income score (minimum $400 USD, max 50 points)."""
    if 400 <= monthly_income < 700:
        return 10
    elif 700 <= monthly_income < 1000:
        return 20
    elif 1000 <= monthly_income < 2000:
        return 30
    elif 2000 <= monthly_income < 3000:
        return 40
    else:
        return 50

def open_loans_score(num_loans):
    """Open loans score (max 100 points)."""
    if num_loans <= 1:
        return 100
    elif 2 <= num_loans <= 3:
        return 80
    elif 4 <= num_loans <= 5:
        return 40
    else:
        return 10

def prepare_applicant_data(row):
    """Transforms a dataset row into the scorecard's expected format."""
    monthly_income = row["MonthlyIncome"] if not pd.isna(row["MonthlyIncome"]) else row["MonthlyIncome"].median()
    
    applicant_data = {
        "age": row["age"],
        "monthly_income": monthly_income,
        "late_payments": (
            row["NumberOfTime30-59DaysPastDueNotWorse"] +
            row["NumberOfTime60-89DaysPastDueNotWorse"] +
            row["NumberOfTimes90DaysLate"]
        ),
        "amount_owed": monthly_income * row["DebtRatio"] * 12,
        "credit_age": min(max(0, row["age"] - 18), 20),
        "credit_types": (
            2 if row["NumberRealEstateLoansOrLines"] > 0 and 
                row["NumberOfOpenCreditLinesAndLoans"] > row["NumberRealEstateLoansOrLines"] 
                else 1
        ),
        "available_credit": 10000 * (1 - row["RevolvingUtilizationOfUnsecuredLines"]),
        "credit_usage": row["RevolvingUtilizationOfUnsecuredLines"] * 100,
        "open_loans": row["NumberOfOpenCreditLinesAndLoans"]
    }
    return applicant_data

def evaluate_applicant(data, threshold=800):
    """Evaluates applicant eligibility and calculates total credit score."""
    if not check_eligibility(data["age"], data["monthly_income"]):
        return None, "Not eligible"
    
    total_score = (
        repayment_history_score(data["late_payments"]) +
        total_amount_owed_score(data["amount_owed"]) +
        credit_history_length_score(data["credit_age"]) +
        credit_types_score(data["credit_types"]) +
        available_credit_score(data["available_credit"]) +
        credit_utilization_score(data["credit_usage"]) +
        income_score(data["monthly_income"]) +
        open_loans_score(data["open_loans"])
    )
    
    decision = "Approved" if total_score >= threshold else "Rejected"
    return total_score, decision

def evaluate_dataset(data, threshold=800):
    """Evaluates entire dataset and compares with target variable."""
    scores = []
    decisions = []
    
    for _, row in data.iterrows():
        applicant_data = prepare_applicant_data(row)
        score, decision = evaluate_applicant(applicant_data, threshold=threshold)
        scores.append(score if score is not None else 0)
        decisions.append(1 if decision == "Approved" else 0)
    
    return scores, decisions

def plot_score_histogram(data, scores, y_true, threshold=850):
    """Generates dual histogram of scores for non-defaulters vs defaulters."""
    plt.figure(figsize=(10, 6))
    sns.histplot(scores[y_true == 1], color="green", label="Non-defaulters (y_true = 1)", bins=50, alpha=0.5)
    sns.histplot(scores[y_true == 0], color="red", label="Defaulters (y_true = 0)", bins=50, alpha=0.5)
    plt.axvline(x=threshold, color="blue", linestyle="--", label=f"Threshold = {threshold}")
    plt.title(f"Scorecard Score Distribution (Threshold = {threshold})")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_evaluation(y_true, y_pred, scores, threshold):
    """Generates evaluation plots: confusion matrix and ROC curve."""
    plt.figure(figsize=(12, 4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False)
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title(f"ROC Curve (Threshold = {threshold})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def analyze_false_positives(data, y_true, y_pred, scores):
    """Analyzes false positives to identify patterns."""
    data["y_true"] = y_true
    data["y_pred"] = y_pred
    data["score"] = scores
    
    false_positives = data[(data["y_true"] == 0) & (data["y_pred"] == 1)]
    
    print("\n=== False Positive Analysis ===")
    print(f"Total False Positives: {len(false_positives)}")
    print("\nKey Variable Statistics in False Positives:")
    key_vars = ["late_payments", "RevolvingUtilizationOfUnsecuredLines", "NumberOfOpenCreditLinesAndLoans", "MonthlyIncome", "score"]
    for var in key_vars:
        print(f"\n{var}:")
        print(false_positives[var].describe())

def plot_score_distribution(scores):
    """Generates histogram of overall credit score distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, color="blue", bins=50, kde=True)
    plt.title("Credit Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.axvline(x=np.mean(scores), color="red", linestyle="--", 
                label=f"Mean: {np.mean(scores):.2f}")
    plt.axvline(x=np.median(scores), color="green", linestyle="-.", 
                label=f"Median: {np.median(scores):.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def evaluate_benchmark_comparison(data, threshold=800):
    """
    Evaluates scorecard model against real target variable (SeriousDlqin2yrs)
    and generates detailed comparison analysis.
    """
    # Calculate scores for all applicants
    scores, decisions = evaluate_dataset(data, threshold=threshold)
    
    # Verify array lengths
    print(f"Data length: {len(data)}")
    print(f"Scores length: {len(scores)}")
    print(f"Decisions length: {len(decisions)}")
    
    # Convert to numpy arrays for consistency
    scores_array = np.array(scores)
    decisions_array = np.array(decisions)
    y_true_array = np.array(1 - data["SeriousDlqin2yrs"])
    
    # Verify lengths after conversion
    print(f"Scores array length: {len(scores_array)}")
    print(f"Decisions array length: {len(decisions_array)}")
    print(f"y_true array length: {len(y_true_array)}")
    
    # Truncate to minimum common length
    min_length = min(len(scores_array), len(decisions_array), len(y_true_array), len(data))
    print(f"Minimum length: {min_length}")
    
    # Truncate all variables
    scores_array = scores_array[:min_length]
    decisions_array = decisions_array[:min_length]
    y_true_array = y_true_array[:min_length]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Actual': pd.Series(y_true_array).map({1: 'Good payer', 0: 'Defaulter'}),
        'Prediction': pd.Series(decisions_array).map({1: 'Approved', 0: 'Rejected'}),
        'Score': pd.Series(scores_array)
    })
    
    # Confusion matrix
    conf_matrix = pd.crosstab(results_df['Actual'], results_df['Prediction'], 
                              rownames=['Actual'], colnames=['Prediction'])
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true_array, decisions_array),
        "Precision": precision_score(y_true_array, decisions_array),
        "Recall": recall_score(y_true_array, decisions_array),
        "F1-Score": f1_score(y_true_array, decisions_array),
        "AUC-ROC": roc_auc_score(y_true_array, scores_array),
    }
    
    # Calculate error rates
    tn, fp, fn, tp = confusion_matrix(y_true_array, decisions_array).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    # Group analysis
    group_analysis = results_df.groupby(['Actual', 'Prediction']).agg(
        count=('Score', 'count'),
        avg_score=('Score', 'mean'),
        min_score=('Score', 'min'),
        max_score=('Score', 'max')
    ).reset_index()
    
    return {
        'results_df': results_df,
        'conf_matrix': conf_matrix,
        'metrics': metrics,
        'fpr': fpr,
        'fnr': fnr,
        'group_analysis': group_analysis
    }

def visualize_benchmark_comparison(results):
    """Generates visualizations for scorecard vs real data comparison."""
    # Confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['conf_matrix'], annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix (Threshold = 800)")
    plt.tight_layout()
    plt.show()
    
    # Score distribution by group
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Actual', y='Score', hue='Prediction', data=results['results_df'])
    plt.axhline(y=800, color='red', linestyle='--', label='Threshold (800)')
    plt.title('Score Distribution by Group')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Score histogram by actual class
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=results['results_df'], 
        x='Score', 
        hue='Actual', 
        multiple='stack',
        bins=50
    )
    plt.axvline(x=800, color='red', linestyle='--', label='Threshold (800)')
    plt.title('Score Distribution by Actual Class')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ROC curve
    y_true = results['results_df']['Actual'].map({'Good payer': 1, 'Defaulter': 0})
    scores = results['results_df']['Score']
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def run_benchmark_comparison(data, threshold=800):
    """Executes complete comparison and displays results."""
    print(f"\n===== SCORECARD EVALUATION VS REAL DATA (THRESHOLD = {threshold}) =====\n")
    
    results = evaluate_benchmark_comparison(data, threshold)
    
    print("Confusion Matrix:")
    print(results['conf_matrix'])
    print("\nInterpretation:")
    print("- True Positives (TP):", results['conf_matrix'].loc['Good payer', 'Approved'], 
          "- Correctly approved good payers")
    print("- False Positives (FP):", results['conf_matrix'].loc['Defaulter', 'Approved'], 
          "- Incorrectly approved defaulters")
    print("- True Negatives (TN):", results['conf_matrix'].loc['Defaulter', 'Rejected'], 
          "- Correctly rejected defaulters")
    print("- False Negatives (FN):", results['conf_matrix'].loc['Good payer', 'Rejected'], 
          "- Incorrectly rejected good payers")
    
    print("\nPerformance Metrics:")
    for metric, value in results['metrics'].items():
        print(f"- {metric}: {value:.4f}")
    
    print(f"- False Positive Rate (FPR): {results['fpr']:.4f}")
    print(f"- False Negative Rate (FNR): {results['fnr']:.4f}")
    
    print("\nGroup Analysis:")
    print(results['group_analysis'])
    
    print("\nGenerating visualizations...")
    visualize_benchmark_comparison(results)
    
    incorrect = results['results_df'][
        ((results['results_df']['Actual'] == 'Good payer') & (results['results_df']['Prediction'] == 'Rejected')) |
        ((results['results_df']['Actual'] == 'Defaulter') & (results['results_df']['Prediction'] == 'Approved'))
    ]
    
    print(f"\nTotal misclassified applicants: {len(incorrect)}")
    print(f"- False Positives (Approved defaulters): {len(incorrect[incorrect['Actual'] == 'Defaulter'])}")
    print(f"- False Negatives (Rejected good payers): {len(incorrect[incorrect['Actual'] == 'Good payer'])}")
    
    return results

def main():
    """Main function to test scorecard with dataset."""
    # Build file path
    data_path = os.path.join(os.path.dirname(__file__), "data", "cs-training.csv")
    
    print(f"Attempting to load file from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: File 'cs-training.csv' not found in {data_path}")
        print("Please verify 'cs-training.csv' is in 'src/Part A/data/'.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Clean dataset
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    
    # Remove delinquency outliers
    data = data[~data["NumberOfTimes90DaysLate"].isin([96, 98])]
    data = data[~data["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98])]
    data = data[~data["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])]
    data = data[data["NumberOfTimes90DaysLate"] <= 10]
    data = data[data["NumberOfTime60-89DaysPastDueNotWorse"] <= 10]
    data = data[data["NumberOfTime30-59DaysPastDueNotWorse"] <= 10]

    # Remove financial ratio outliers
    data = data[data["DebtRatio"] <= 2]
    data = data[data["RevolvingUtilizationOfUnsecuredLines"] <= 2]

    # Filter realistic ages
    data = data[(data["age"] >= 18) & (data["age"] <= 100)]

    # Impute missing values
    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(data["MonthlyIncome"].median())
    data["NumberOfDependents"] = data["NumberOfDependents"].fillna(0)

    # Calculate late payments
    data["late_payments"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"] +
        data["NumberOfTime60-89DaysPastDueNotWorse"] +
        data["NumberOfTimes90DaysLate"]
    )

    # Calculate scores for all applicants
    scores, _ = evaluate_dataset(data, threshold=0)  # Threshold 0 to get all scores

    # Filter valid scores
    valid_scores = [s for s in scores if s > 0]

    # Generate score distribution
    plot_score_distribution(valid_scores)

    # Compare with target variable
    y_true = 1 - data["SeriousDlqin2yrs"]

    # Generate dual histogram
    plot_score_histogram(data, np.array(scores), y_true, threshold=800)

    run_benchmark_comparison(data, threshold=800)

    # Test different thresholds
    thresholds = [800, 850, 900]
    for threshold in thresholds:
        print(f"\n=== Evaluation with Threshold = {threshold} ===")
        scores, decisions = evaluate_dataset(data, threshold=threshold)

        y_true = 1 - data["SeriousDlqin2yrs"]
        y_pred = decisions

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "AUC-ROC": roc_auc_score(y_true, scores),
            "FPR": confusion_matrix(y_true, y_pred)[0, 1] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1]),
            "FNR": confusion_matrix(y_true, y_pred)[1, 0] / (confusion_matrix(y_true, y_pred)[1, 0] + confusion_matrix(y_true, y_pred)[1, 1]),
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }

        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
        print(f"FPR (False Positive Rate): {metrics['FPR']:.4f}")
        print(f"FNR (False Negative Rate): {metrics['FNR']:.4f}")
        print("Confusion Matrix:")
        print(metrics['Confusion Matrix'])

        plot_evaluation(y_true, y_pred, scores, threshold)
        analyze_false_positives(data, y_true, y_pred, scores)

    # Score statistics
    valid_scores = [s for s in scores if s > 0]
    print("\nScore Statistics:")
    print(f"Eligible Applicants: {len(valid_scores)}")
    print(f"Average Score: {np.mean(valid_scores):.2f}")
    print(f"Standard Deviation: {np.std(valid_scores):.2f}")
    print(f"Minimum Score: {np.min(valid_scores) if valid_scores else 0}")
    print(f"Maximum Score: {np.max(valid_scores) if valid_scores else 0}")

if __name__ == "__main__":
    main()