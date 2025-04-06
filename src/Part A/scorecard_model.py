# Modelo Tradicional de Credit Scoring para Préstamos Personales
# Autores: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Fecha: 25 de marzo de 2025
# Descripción: Scorecard híbrido con restricciones de entrada basado en bancos mexicanos,
# basado en modelos como FICO y VantageScore.


import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def check_eligibility(age, monthly_income, loan_amount):
    """Verifica si el solicitante cumple con los requisitos mínimos.
    
    Args:
        age (int): Edad del solicitante.
        monthly_income (float): Ingresos mensuales en USD.
        loan_amount (float): Monto solicitado en USD.
    
    Returns:
        bool: True si es elegible, False si no.
    """
    min_age = 18
    min_income = 400  # $8,000 MXN ≈ $400 USD
    max_loan = 150000  # $3,000,000 MXN ≈ $150,000 USD
    
    if age < min_age or monthly_income < min_income or loan_amount > max_loan:
        return False
    return True

def repayment_history_score(late_payments):
    """Puntaje por historial de pagos (máximo 350 puntos)."""
    if late_payments == 0:
        return 350
    elif late_payments == 1:
        return 200
    elif late_payments == 2:
        return 100
    else:
        return 0

def total_amount_owed_score(amount_owed):
    """Puntaje por monto total adeudado (máximo 150 puntos)."""
    if amount_owed < 1000:
        return 150
    elif 1000 <= amount_owed < 5000:
        return 100
    elif 5000 <= amount_owed < 10000:
        return 75
    else:
        return 50

def credit_history_length_score(years):
    """Puntaje por antigüedad crediticia (máximo 150 puntos)."""
    if years < 2:
        return 50
    elif 2 <= years < 5:
        return 75
    elif 5 <= years < 10:
        return 100
    else:
        return 150

def credit_types_score(num_types):
    """Puntaje por mezcla de créditos (máximo 100 puntos)."""
    if num_types == 1:
        return 60
    elif num_types == 2:
        return 80
    else:
        return 100

def new_credit_score(inquiries):
    """Puntaje por nuevos créditos/consultas (máximo 100 puntos)."""
    if inquiries == 0:
        return 100
    elif inquiries == 1:
        return 80
    elif inquiries == 2:
        return 60
    else:
        return 40

def available_credit_score(credit_available):
    """Puntaje por crédito disponible (máximo 100 puntos)."""
    if credit_available > 5000:
        return 100
    elif 2000 <= credit_available <= 5000:
        return 80
    elif 1000 <= credit_available < 2000:
        return 60
    else:
        return 40

def credit_utilization_score(usage_percent):
    """Puntaje por utilización de crédito (máximo 150 puntos)."""
    if usage_percent < 30:
        return 150
    elif 30 <= usage_percent < 50:
        return 100
    elif 50 <= usage_percent < 70:
        return 75
    else:
        return 50

def income_score(monthly_income):
    """Puntaje por ingresos mensuales, mínimo $400 USD (máximo 100 puntos)."""
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
    """Puntaje por antigüedad laboral (máximo 100 puntos)."""
    if years < 1:
        return 40
    elif 1 <= years < 3:
        return 60
    elif 3 <= years < 5:
        return 80
    else:
        return 100

def open_loans_score(num_loans):
    """Puntaje por cantidad de préstamos abiertos (máximo 100 puntos)."""
    if num_loans <= 1:
        return 100
    elif 2 <= num_loans <= 3:
        return 80
    elif 4 <= num_loans <= 5:
        return 60
    else:
        return 40

def prepare_applicant_data(row):
    """Transforma una fila del dataset en el formato esperado por el scorecard."""
    # Imputar valores faltantes
    monthly_income = row["MonthlyIncome"] if not pd.isna(row["MonthlyIncome"]) else row["MonthlyIncome"].median()
    
    # Mapeo de variables
    applicant_data = {
        "age": row["age"],
        "monthly_income": monthly_income,
        "loan_amount": 10000,  # Asumimos $10,000 para pruebas
        "late_payments": (
            row["NumberOfTime30-59DaysPastDueNotWorse"] +
            row["NumberOfTime60-89DaysPastDueNotWorse"] +
            row["NumberOfTimes90DaysLate"]
        ),
        "amount_owed": monthly_income * row["DebtRatio"] * 12,  # Estimamos deuda anual
        "credit_age": min(max(0, row["age"] - 18), 20),  # Limitamos a 20 años
        "credit_types": (
            2 if row["NumberRealEstateLoansOrLines"] > 0 and 
                row["NumberOfOpenCreditLinesAndLoans"] > row["NumberRealEstateLoansOrLines"] 
                else 1
        ),
        "inquiries": 0,  # Asumimos 0 consultas
        "available_credit": 10000 * (1 - row["RevolvingUtilizationOfUnsecuredLines"]),  # Límite fijo de $10,000
        "credit_usage": row["RevolvingUtilizationOfUnsecuredLines"] * 100,  # Convertimos a porcentaje
        "job_tenure": 3,  # Asumimos 3 años
        "open_loans": row["NumberOfOpenCreditLinesAndLoans"]
    }
    return applicant_data

def evaluate_applicant(data):
    """Evalúa al solicitante si es elegible y calcula el puntaje total."""
    # Verificar elegibilidad
    if not check_eligibility(data["age"], data["monthly_income"], data["loan_amount"]):
        return None, "No elegible"
    
    # Calcular puntaje
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
    
    threshold = 600  # Umbral de aprobación
    decision = "Aprobado" if total_score >= threshold else "Rechazado"
    return total_score, decision

def evaluate_dataset(data):
    """Evalúa todo el dataset y compara con la variable objetivo."""
    scores = []
    decisions = []
    
    for _, row in data.iterrows():
        applicant_data = prepare_applicant_data(row)
        score, decision = evaluate_applicant(applicant_data)
        scores.append(score if score is not None else 0)
        decisions.append(1 if decision == "Aprobado" else 0)  # 1 = Aprobado, 0 = Rechazado/No elegible
    
    return scores, decisions

def main():
    """Función principal para probar el scorecard con el dataset."""
    # Cargar el dataset
    
    data = pd.read_csv("src/Part A/data/cs-training.csv")

    # Limpiar el dataset (mismo preprocesamiento que en la Parte C)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    data = data[data["RevolvingUtilizationOfUnsecuredLines"] < 13]
    data = data[~data["NumberOfTimes90DaysLate"].isin([96, 98])]
    data = data[~data["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98])]
    data = data[~data["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])]
    debt_ratio_threshold = data["DebtRatio"].quantile(0.975)
    data = data[data["DebtRatio"] <= debt_ratio_threshold]

    # Imputar valores faltantes para MonthlyIncome (usamos la mediana del dataset completo)
    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(data["MonthlyIncome"].median())

    # Evaluar el dataset
    scores, decisions = evaluate_dataset(data)

    # Comparar con la variable objetivo (SeriousDlqin2yrs invertida: 1 = no incumplimiento, 0 = incumplimiento)
    y_true = 1 - data["SeriousDlqin2yrs"]  # Invertimos para que 1 = buen cliente (debería ser Aprobado)
    y_pred = decisions

    # Calcular métricas
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

    # Imprimir resultados
    print("=== Resultados de la Evaluación del Scorecard ===")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print("Matriz de Confusión:")
    print(metrics['Confusion Matrix'])

    # Estadísticas de los puntajes
    valid_scores = [s for s in scores if s > 0]
    print("\nEstadísticas de los Puntajes:")
    print(f"Solicitantes Elegibles: {len(valid_scores)}")
    print(f"Promedio de Puntaje: {np.mean(valid_scores):.2f}")
    print(f"Desviación Estándar: {np.std(valid_scores):.2f}")
    print(f"Puntaje Mínimo: {np.min(valid_scores) if valid_scores else 0}")
    print(f"Puntaje Máximo: {np.max(valid_scores) if valid_scores else 0}")

    cm = metrics["Confusion Matrix"]
    labels = ["Rechazado (Real)", "Aprobado (Real)"]
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusión del Scorecard")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()