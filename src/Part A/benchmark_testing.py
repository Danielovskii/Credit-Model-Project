import random
#comentario prueba


# Funciones del modelo (copiadas de la versión anterior)
def check_eligibility(age: int, monthly_income: float, loan_amount: float) -> bool:
    return age >= 18 and monthly_income >= 400 and loan_amount <= 150000

def repayment_history_score(late_payments): return 350 if late_payments == 0 else (200 if late_payments == 1 else (100 if late_payments == 2 else 0))
def total_amount_owed_score(amount_owed): return 150 if amount_owed < 1000 else (100 if amount_owed < 5000 else (75 if amount_owed < 10000 else 50))
def credit_history_length_score(years): return 50 if years < 2 else (75 if years < 5 else (100 if years < 10 else 150))
def credit_types_score(num_types): return 60 if num_types == 1 else (80 if num_types == 2 else 100)
def new_credit_score(inquiries): return 100 if inquiries == 0 else (80 if inquiries == 1 else (60 if inquiries == 2 else 40))
def available_credit_score(credit_available): return 100 if credit_available > 5000 else (80 if credit_available >= 2000 else (60 if credit_available >= 1000 else 40))
def credit_utilization_score(usage_percent): return 150 if usage_percent < 30 else (100 if usage_percent < 50 else (75 if usage_percent < 70 else 50))
def income_score(monthly_income): return 40 if monthly_income < 1000 else (60 if monthly_income < 2000 else (80 if monthly_income < 3000 else 100))
def job_tenure_score(years): return 40 if years < 1 else (60 if years < 3 else (80 if years < 5 else 100))
def open_loans_score(num_loans): return 100 if num_loans <= 1 else (80 if num_loans <= 3 else (60 if num_loans <= 5 else 40))

def evaluate_applicant(data):
    if not check_eligibility(data["age"], data["monthly_income"], data["loan_amount"]):
        return None, "No elegible"
    
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
    
    threshold = 550
    decision = "Aprobado" if total_score >= threshold else "Rechazado"
    return total_score, decision

# Generar conjunto de prueba simulado
def generate_test_data(n: int):
    data = []
    for _ in range(n):
        applicant = {
            "age": random.randint(16, 60),
            "monthly_income": random.uniform(300, 5000),
            "loan_amount": random.uniform(1000, 200000),
            "late_payments": random.randint(0, 4),
            "amount_owed": random.uniform(0, 15000),
            "credit_age": random.randint(0, 15),
            "credit_types": random.randint(1, 4),
            "inquiries": random.randint(0, 5),
            "available_credit": random.uniform(500, 10000),
            "credit_usage": random.uniform(0, 100),
            "job_tenure": random.randint(0, 10),
            "open_loans": random.randint(0, 6)
        }
        # Criterio de benchmark: aprobado si tiene buen perfil de riesgo
        expected = "Aprobado" if (
            applicant["late_payments"] <= 1 and 
            applicant["credit_usage"] < 50 and 
            applicant["monthly_income"] >= 1000 and 
            applicant["amount_owed"] < 10000 and 
            applicant["open_loans"] <= 3
        ) else "Rechazado"
        if not check_eligibility(applicant["age"], applicant["monthly_income"], applicant["loan_amount"]):
            expected = "No elegible"
        applicant["expected"] = expected
        data.append(applicant)
    return data

# Evaluar el modelo contra el benchmark
def evaluate_model(test_data):
    correct = 0
    total = len(test_data)
    eligible_count = 0
    false_positives = 0
    false_negatives = 0
    
    print("=== Resultados de la Evaluación ===")
    for i, applicant in enumerate(test_data, 1):
        score, decision = evaluate_applicant(applicant)
        expected = applicant["expected"]
        
        print(f"Solicitante {i}: Puntaje={score if score is not None else 'N/A'}, Decisión={decision}, Esperado={expected}")
        if decision == expected:
            correct += 1
        if expected != "No elegible":
            eligible_count += 1
            if decision == "Aprobado" and expected == "Rechazado":
                false_positives += 1
            elif decision == "Rechazado" and expected == "Aprobado":
                false_negatives += 1
    
    accuracy = correct / total * 100
    print(f"\nPrecisión total: {accuracy:.2f}% ({correct}/{total})")
    if eligible_count > 0:
        eligible_accuracy = (correct - (total - eligible_count)) / eligible_count * 100
        print(f"Precisión entre elegibles: {eligible_accuracy:.2f}%")
        print(f"Falsos positivos: {false_positives}")
        print(f"Falsos negativos: {false_negatives}")

# Ejecutar evaluación
test_data = generate_test_data(20)
evaluate_model(test_data)