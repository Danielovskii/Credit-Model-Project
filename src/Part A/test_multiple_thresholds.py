# scorecard_model_base_simplified_adjusted.py
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def check_eligibility(age, monthly_income):
    """Verifica si el solicitante cumple con los requisitos mínimos."""
    min_age = 18
    min_income = 400  # $8,000 MXN ≈ $400 USD
    
    if age < min_age or monthly_income < min_income:
        return False
    return True

def repayment_history_score(late_payments):
    """Puntaje por historial de pagos (máximo 350 puntos)."""
    if late_payments == 0:
        return 350
    elif late_payments == 1:
        return 150  # Suavizado
    elif late_payments == 2:
        return 50   # Suavizado
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
        return 50
    elif 50 <= usage_percent < 70:
        return 10
    else:
        return 0

def income_score(monthly_income):
    """Puntaje por ingresos mensuales, mínimo $400 USD (máximo 50 puntos)."""
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
    """Puntaje por cantidad de préstamos abiertos (máximo 100 puntos)."""
    if num_loans <= 1:
        return 100
    elif 2 <= num_loans <= 3:
        return 80
    elif 4 <= num_loans <= 5:
        return 40
    else:
        return 10

def prepare_applicant_data(row):
    """Transforma una fila del dataset en el formato esperado por el scorecard."""
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
    """Evalúa al solicitante si es elegible y calcula el puntaje total."""
    if not check_eligibility(data["age"], data["monthly_income"]):
        return None, "No elegible"
    
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
    
    decision = "Aprobado" if total_score >= threshold else "Rechazado"
    return total_score, decision

def evaluate_dataset(data, threshold=800):
    """Evalúa todo el dataset y compara con la variable objetivo."""
    scores = []
    decisions = []
    
    for _, row in data.iterrows():
        applicant_data = prepare_applicant_data(row)
        score, decision = evaluate_applicant(applicant_data, threshold=threshold)
        scores.append(score if score is not None else 0)
        decisions.append(1 if decision == "Aprobado" else 0)
    
    return scores, decisions

def plot_score_histogram(data, scores, y_true, threshold=850):
    """Genera un multi-histograma de puntajes para no incumplidores y incumplidores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(scores[y_true == 1], color="green", label="No Incumplidores (y_true = 1)", bins=50, alpha=0.5)
    sns.histplot(scores[y_true == 0], color="red", label="Incumplidores (y_true = 0)", bins=50, alpha=0.5)
    plt.axvline(x=threshold, color="blue", linestyle="--", label=f"Umbral = {threshold}")
    plt.title(f"Distribución de Puntajes del Scorecard (Umbral = {threshold})")
    plt.xlabel("Puntaje")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()

def plot_evaluation(y_true, y_pred, scores, threshold):
    """Genera gráficos de evaluación: matriz de confusión y curva ROC."""
    plt.figure(figsize=(12, 4))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False)
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Curva ROC
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
    """Analiza los falsos positivos para identificar patrones."""
    data["y_true"] = y_true
    data["y_pred"] = y_pred
    data["score"] = scores
    
    false_positives = data[(data["y_true"] == 0) & (data["y_pred"] == 1)]
    
    print("\n=== Análisis de Falsos Positivos ===")
    print(f"Total de Falsos Positivos: {len(false_positives)}")
    print("\nEstadísticas de Variables Clave en Falsos Positivos:")
    key_vars = ["late_payments", "RevolvingUtilizationOfUnsecuredLines", "NumberOfOpenCreditLinesAndLoans", "MonthlyIncome", "score"]
    for var in key_vars:
        print(f"\n{var}:")
        print(false_positives[var].describe())

def plot_score_distribution(scores):
    """Genera un histograma de la distribución general de los puntajes de crédito."""
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, color="blue", bins=50, kde=True)
    plt.title("Distribución de Puntajes de Crédito")
    plt.xlabel("Puntaje")
    plt.ylabel("Frecuencia")
    plt.axvline(x=np.mean(scores), color="red", linestyle="--", 
                label=f"Media: {np.mean(scores):.2f}")
    plt.axvline(x=np.median(scores), color="green", linestyle="-.", 
                label=f"Mediana: {np.median(scores):.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def evaluate_benchmark_comparison(data, threshold=800):
    """
    Evalúa el scorecard model contra la variable objetivo real (SeriousDlqin2yrs)
    y genera un análisis detallado de la comparación.
    """
    # Calcular scores para todos los solicitantes
    scores, decisions = evaluate_dataset(data, threshold=threshold)
    
    # Verificar que todos los arrays tengan la misma longitud
    print(f"Longitud de data: {len(data)}")
    print(f"Longitud de scores: {len(scores)}")
    print(f"Longitud de decisions: {len(decisions)}")
    
    # Asegurarnos de trabajar con arrays del mismo tamaño
    # Convertir a arrays de numpy para uniformidad
    scores_array = np.array(scores)
    decisions_array = np.array(decisions)
    y_true_array = np.array(1 - data["SeriousDlqin2yrs"])
    
    # Verificar longitudes después de conversión
    print(f"Longitud de scores_array: {len(scores_array)}")
    print(f"Longitud de decisions_array: {len(decisions_array)}")
    print(f"Longitud de y_true_array: {len(y_true_array)}")
    
    # Si hay alguna diferencia en las longitudes, truncar al mínimo común
    min_length = min(len(scores_array), len(decisions_array), len(y_true_array), len(data))
    print(f"Longitud mínima: {min_length}")
    
    # Truncar todas las variables a la misma longitud
    scores_array = scores_array[:min_length]
    decisions_array = decisions_array[:min_length]
    y_true_array = y_true_array[:min_length]
    
    # Crear DataFrame de resultados asegurando que todos los arrays tienen la misma longitud
    results_df = pd.DataFrame({
        'Real': pd.Series(y_true_array).map({1: 'Buen pagador', 0: 'Incumplidor'}),
        'Predicción': pd.Series(decisions_array).map({1: 'Aprobado', 0: 'Rechazado'}),
        'Score': pd.Series(scores_array)
    })
    
    # Resto del código
    # Matriz de confusión en formato más legible
    conf_matrix = pd.crosstab(results_df['Real'], results_df['Predicción'], 
                              rownames=['Real'], colnames=['Predicción'])
    
    # Calcular métricas
    metrics = {
        "Accuracy": accuracy_score(y_true_array, decisions_array),
        "Precision": precision_score(y_true_array, decisions_array),
        "Recall": recall_score(y_true_array, decisions_array),
        "F1-Score": f1_score(y_true_array, decisions_array),
        "AUC-ROC": roc_auc_score(y_true_array, scores_array),
    }
    
    # Calcular tasas de falsos positivos y negativos
    tn, fp, fn, tp = confusion_matrix(y_true_array, decisions_array).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    # Análisis por grupo
    group_analysis = results_df.groupby(['Real', 'Predicción']).agg(
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
    """
    Genera visualizaciones para la comparación del scorecard contra datos reales.
    """
    # Matriz de confusión como heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['conf_matrix'], annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusión (Umbral = 800)")
    plt.tight_layout()
    plt.show()
    
    # Distribución de scores por grupo (Real vs Predicción)
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Real', y='Score', hue='Predicción', data=results['results_df'])
    plt.axhline(y=800, color='red', linestyle='--', label='Umbral (800)')
    plt.title('Distribución de Scores por Grupo')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Histograma de scores separado por clase real
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=results['results_df'], 
        x='Score', 
        hue='Real', 
        multiple='stack',
        bins=50
    )
    plt.axvline(x=800, color='red', linestyle='--', label='Umbral (800)')
    plt.title('Distribución de Scores por Clase Real')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calcular y graficar curva ROC
    y_true = results['results_df']['Real'].map({'Buen pagador': 1, 'Incumplidor': 0})
    scores = results['results_df']['Score']
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def run_benchmark_comparison(data, threshold=800):
    """
    Ejecuta la comparación completa y muestra resultados.
    """
    print(f"\n===== EVALUACIÓN DEL SCORECARD VS DATOS REALES (UMBRAL = {threshold}) =====\n")
    
    # Obtener resultados de la evaluación
    results = evaluate_benchmark_comparison(data, threshold)
    
    # Mostrar matriz de confusión
    print("Matriz de Confusión:")
    print(results['conf_matrix'])
    print("\nNota interpretativa:")
    print("- Verdaderos Positivos (VP):", results['conf_matrix'].loc['Buen pagador', 'Aprobado'], 
          "- Buenos pagadores correctamente aprobados")
    print("- Falsos Positivos (FP):", results['conf_matrix'].loc['Incumplidor', 'Aprobado'], 
          "- Incumplidores incorrectamente aprobados")
    print("- Verdaderos Negativos (VN):", results['conf_matrix'].loc['Incumplidor', 'Rechazado'], 
          "- Incumplidores correctamente rechazados")
    print("- Falsos Negativos (FN):", results['conf_matrix'].loc['Buen pagador', 'Rechazado'], 
          "- Buenos pagadores incorrectamente rechazados")
    
    # Mostrar métricas
    print("\nMétricas de Rendimiento:")
    for metric, value in results['metrics'].items():
        print(f"- {metric}: {value:.4f}")
    
    print(f"- Tasa de Falsos Positivos (FPR): {results['fpr']:.4f}")
    print(f"- Tasa de Falsos Negativos (FNR): {results['fnr']:.4f}")
    
    # Análisis por grupo
    print("\nAnálisis por Grupo:")
    print(results['group_analysis'])
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    visualize_benchmark_comparison(results)
    
    # Análisis de aplicantes incorrectamente clasificados
    incorrect = results['results_df'][
        ((results['results_df']['Real'] == 'Buen pagador') & (results['results_df']['Predicción'] == 'Rechazado')) |
        ((results['results_df']['Real'] == 'Incumplidor') & (results['results_df']['Predicción'] == 'Aprobado'))
    ]
    
    print(f"\nTotal de aplicantes incorrectamente clasificados: {len(incorrect)}")
    print(f"- Falsos Positivos (Incumplidores aprobados): {len(incorrect[incorrect['Real'] == 'Incumplidor'])}")
    print(f"- Falsos Negativos (Buenos pagadores rechazados): {len(incorrect[incorrect['Real'] == 'Buen pagador'])}")
    
    return results

def main():
    """Función principal para probar el scorecard con el dataset."""
    # Construir la ruta al archivo
    data_path = os.path.join(os.path.dirname(__file__), "data", "cs-training.csv")
    
    print(f"Intentando cargar el archivo desde: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo 'cs-training.csv' en {data_path}")
        print("Por favor, verifica que 'cs-training.csv' esté en 'src/Part A/data/'.")
        return

    # Cargar el dataset
    data = pd.read_csv(data_path)

    # Limpiar el dataset
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    
    # Eliminar outliers en variables de morosidad
    data = data[~data["NumberOfTimes90DaysLate"].isin([96, 98])]
    data = data[~data["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98])]
    data = data[~data["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])]
    data = data[data["NumberOfTimes90DaysLate"] <= 10]
    data = data[data["NumberOfTime60-89DaysPastDueNotWorse"] <= 10]
    data = data[data["NumberOfTime30-59DaysPastDueNotWorse"] <= 10]

    # Eliminar outliers en DebtRatio y RevolvingUtilizationOfUnsecuredLines
    data = data[data["DebtRatio"] <= 2]
    data = data[data["RevolvingUtilizationOfUnsecuredLines"] <= 2]

    # Filtrar edad realista
    data = data[(data["age"] >= 18) & (data["age"] <= 100)]

    # Imputar valores faltantes
    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(data["MonthlyIncome"].median())
    data["NumberOfDependents"] = data["NumberOfDependents"].fillna(0)

    # Calcular late_payments
    data["late_payments"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"] +
        data["NumberOfTime60-89DaysPastDueNotWorse"] +
        data["NumberOfTimes90DaysLate"]
    )

    # Calcular puntajes para todos los solicitantes
    scores, _ = evaluate_dataset(data, threshold=0)  # Umbral 0 para obtener todos los puntajes

    # Filtrar puntajes válidos (mayores que 0)
    valid_scores = [s for s in scores if s > 0]

    # Generar la distribución general de puntajes
    plot_score_distribution(valid_scores)

    # Comparar con la variable objetivo
    y_true = 1 - data["SeriousDlqin2yrs"]

    # Generar el multi-histograma
    plot_score_histogram(data, np.array(scores), y_true, threshold=800)

    run_benchmark_comparison(data, threshold=800)

    # Probar diferentes umbrales
    thresholds = [800, 850, 900]
    for threshold in thresholds:
        print(f"\n=== Evaluación con Umbral = {threshold} ===")
        scores, decisions = evaluate_dataset(data, threshold=threshold)

        # Comparar con la variable objetivo
        y_true = 1 - data["SeriousDlqin2yrs"]
        y_pred = decisions

        # Calcular métricas
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

        # Imprimir resultados
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
        print(f"FPR (False Positive Rate): {metrics['FPR']:.4f}")
        print(f"FNR (False Negative Rate): {metrics['FNR']:.4f}")
        print("Matriz de Confusión:")
        print(metrics['Confusion Matrix'])

        # Generar gráficos
        plot_evaluation(y_true, y_pred, scores, threshold)

        # Analizar falsos positivos
        analyze_false_positives(data, y_true, y_pred, scores)

    # Estadísticas de los puntajes
    valid_scores = [s for s in scores if s > 0]
    print("\nEstadísticas de los Puntajes:")
    print(f"Solicitantes Elegibles: {len(valid_scores)}")
    print(f"Promedio de Puntaje: {np.mean(valid_scores):.2f}")
    print(f"Desviación Estándar: {np.std(valid_scores):.2f}")
    print(f"Puntaje Mínimo: {np.min(valid_scores) if valid_scores else 0}")
    print(f"Puntaje Máximo: {np.max(valid_scores) if valid_scores else 0}")

if __name__ == "__main__":
    main()