# Modelo Tradicional de Credit Scoring para Préstamos Personales
# Autores: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Fecha: 25 de marzo de 2025
# Descripción: Scorecard híbrido con restricciones de entrada basado en bancos mexicanos,
# basado en modelos como FICO y VantageScore.

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
    
    threshold = 600 # Umbral de aprobación
    decision = "Aprobado" if total_score >= threshold else "Rechazado"
    return total_score, decision

# Ejemplo de uso
applicant_data = {
    "age": 25,              # Edad
    "monthly_income": 2500, # Ingresos mensuales en USD
    "loan_amount": 10000,   # Monto solicitado en USD
    "late_payments": 0,     # Sin atrasos
    "amount_owed": 3000,    # $3000 adeudados
    "credit_age": 7,        # 7 años de historial
    "credit_types": 3,      # 3 tipos de cuentas
    "inquiries": 1,         # 1 consulta
    "available_credit": 4000, # $4000 disponibles
    "credit_usage": 25,     # 25% utilización
    "job_tenure": 4,        # 4 años en empleo
    "open_loans": 2         # 2 préstamos activos
}

score, decision = evaluate_applicant(applicant_data)
if score is not None:
    print(f"Puntaje total: {score} / 1350")
print(f"Decisión: {decision}")

# Ejemplo no elegible
applicant_data2 = {
    "age": 17,              # Menor de edad
    "monthly_income": 300,  # Ingresos insuficientes
    "loan_amount": 200000,  # Excede el máximo
    "late_payments": 0,
    "amount_owed": 3000,
    "credit_age": 7,
    "credit_types": 3,
    "inquiries": 1,
    "available_credit": 4000,
    "credit_usage": 25,
    "job_tenure": 4,
    "open_loans": 2
}

score2, decision2 = evaluate_applicant(applicant_data2)
if score2 is not None:
    print(f"Puntaje total: {score2} / 1350")
print(f"Decisión: {decision2}")