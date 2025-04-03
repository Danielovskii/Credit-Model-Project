# Modelo de Riesgo Crediticio
# Autores: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Fecha: 31 de marzo de 2025
# Descripción: Modelo de riesgo que expande el scorecard tradicional de la Parte A,
# Mediante una Red Neuronal obtiene la probabilidad de incumplimiento q

                                                                # el modelo calcula métricas de riesgo como:
                                                                # - Probability of Default (PD)
                                                                # - Loss Given Default (LGD)
                                                                # - Exposure at Default (EAD)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

class CreditRiskMLModel:
    """Clase que implementa un modelo de riesgo crediticio usando Machine Learning."""
    
    def __init__(self, random_state=42):
        """Inicializa el modelo de riesgo crediticio.

        Args:
            random_state (int, optional): Semilla para reproducibilidad. Defaults to 42.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer_median = SimpleImputer(strategy="median")
        self.imputer_mode = SimpleImputer(strategy="most_frequent")
        self.logistic_model = LogisticRegression(random_state=self.random_state, class_weight="balanced", max_iter=1000)
        self.nn_model = None  # Red neuronal se inicializará en el método train_neural_network
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_weights = None

    def preprocess_data(self, data: pd.DataFrame, target_col: str) -> None:
        """Preprocesa los datos: limpia outliers, imputa valores faltantes, estandariza y divide en entrenamiento/prueba.

        Args:
            data (pd.DataFrame): Dataset completo.
            target_col (str): Nombre de la columna objetivo.
        """
        # Eliminar columna 'Unnamed: 0'
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        # Eliminar outliers en RevolvingUtilizationOfUnsecuredLines (>= 13)
        data = data[data["RevolvingUtilizationOfUnsecuredLines"] < 13]

        # Eliminar muestras donde las columnas de morosidad tienen valores 96 o 98
        data = data[~data["NumberOfTimes90DaysLate"].isin([96, 98])]
        data = data[~data["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98])]
        data = data[~data["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])]

        # Limitar DebtRatio (percentil 97.5 para evitar valores extremos)
        debt_ratio_threshold = data["DebtRatio"].quantile(0.975)
        data = data[data["DebtRatio"] <= debt_ratio_threshold]

        # Separar características y objetivo
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Imputar valores faltantes
        # MonthlyIncome con mediana
        X["MonthlyIncome"] = self.imputer_median.fit_transform(X[["MonthlyIncome"]])
        # NumberOfDependents con modo
        X["NumberOfDependents"] = self.imputer_mode.fit_transform(X[["NumberOfDependents"]])

        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        # Estandarizar las características
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Calcular pesos de clase para manejar el desbalanceo
        classes = np.unique(self.y_train)
        self.class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
        self.class_weights = dict(zip(classes, self.class_weights))

    def train_logistic_regression(self) -> None:
        """Entrena el modelo de regresión logística."""
        self.logistic_model.fit(self.X_train, self.y_train)

    def train_neural_network(self) -> None:
        """Entrena una red neuronal con dos capas ocultas."""
        self.nn_model = Sequential([
            Dense(64, activation="relu", input_shape=(self.X_train.shape[1],), kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])

        self.nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.nn_model.fit(
            self.X_train, self.y_train,
            epochs=50, batch_size=32, verbose=1,
            validation_split=0.2, class_weight=self.class_weights
        )

    def evaluate_models(self) -> dict:
        """Evalúa ambos modelos y retorna métricas de desempeño.

        Returns:
            dict: Diccionario con métricas para ambos modelos.
        """
        # Predicciones de la regresión logística
        lr_pred = self.logistic_model.predict(self.X_test)
        lr_prob = self.logistic_model.predict_proba(self.X_test)[:, 1]

        # Predicciones de la red neuronal
        nn_prob = self.nn_model.predict(self.X_test, verbose=0).flatten()
        nn_pred = (nn_prob >= 0.5).astype(int)

        # Calcular métricas
        metrics = {
            "Logistic Regression": {
                "Accuracy": accuracy_score(self.y_test, lr_pred),
                "Precision": precision_score(self.y_test, lr_pred),
                "Recall": recall_score(self.y_test, lr_pred),
                "F1-Score": f1_score(self.y_test, lr_pred),
                "AUC-ROC": roc_auc_score(self.y_test, lr_prob),
                "Confusion Matrix": confusion_matrix(self.y_test, lr_pred)
            },
            "Neural Network": {
                "Accuracy": accuracy_score(self.y_test, nn_pred),
                "Precision": precision_score(self.y_test, nn_pred),
                "Recall": recall_score(self.y_test, nn_pred),
                "F1-Score": f1_score(self.y_test, nn_pred),
                "AUC-ROC": roc_auc_score(self.y_test, nn_prob),
                "Confusion Matrix": confusion_matrix(self.y_test, nn_pred)
            }
        }
        return metrics

def main():
    """Función principal para ejecutar el modelo."""
    # Cargar el dataset
    data = pd.read_csv("cs-training.csv")  # Ajusta la ruta según donde tengas el archivo

    # Inicializar el modelo
    model = CreditRiskMLModel()

    # Preprocesar los datos
    model.preprocess_data(data, target_col="SeriousDlqin2yrs")

    # Entrenar los modelos
    model.train_logistic_regression()
    model.train_neural_network()

    # Evaluar los modelos
    metrics = model.evaluate_models()

    # Imprimir resultados
    print("=== Resultados de la Evaluación ===")
    for model_name, model_metrics in metrics.items():
        print(f"\nModelo: {model_name}")
        print(f"Accuracy: {model_metrics['Accuracy']:.4f}")
        print(f"Precision: {model_metrics['Precision']:.4f}")
        print(f"Recall: {model_metrics['Recall']:.4f}")
        print(f"F1-Score: {model_metrics['F1-Score']:.4f}")
        print(f"AUC-ROC: {model_metrics['AUC-ROC']:.4f}")
        print("Matriz de Confusión:")
        print(model_metrics['Confusion Matrix'])

if __name__ == "__main__":
    main()
