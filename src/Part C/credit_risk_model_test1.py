# credit_risk_model.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

class CreditRiskMLModel:
    """Clase que implementa un modelo de riesgo crediticio usando Machine Learning."""
    
    def __init__(self, random_state=42):
        """Inicializa el modelo de riesgo crediticio."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer_median = SimpleImputer(strategy="median")
        self.imputer_mode = SimpleImputer(strategy="most_frequent")
        self.logistic_model = LogisticRegression(random_state=self.random_state, class_weight="balanced", max_iter=1000)
        self.nn_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_weights = None

    def preprocess_data(self, data: pd.DataFrame, target_col: str) -> None:
        """Preprocesa los datos: limpia outliers, imputa valores faltantes, estandariza y divide en entrenamiento/prueba."""
        # Eliminar columna 'Unnamed: 0'
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        # Eliminar outliers
        data = data[data["RevolvingUtilizationOfUnsecuredLines"] < 13]
        data = data[~data["NumberOfTimes90DaysLate"].isin([96, 98])]
        data = data[~data["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98])]
        data = data[~data["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])]
        debt_ratio_threshold = data["DebtRatio"].quantile(0.975)
        data = data[data["DebtRatio"] <= debt_ratio_threshold]

        # Separar características y objetivo
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Imputar valores faltantes
        X["MonthlyIncome"] = self.imputer_median.fit_transform(X[["MonthlyIncome"]])
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
        """Evalúa ambos modelos y retorna métricas de desempeño."""
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

    def plot_evaluation(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        """Genera gráficos de evaluación: matriz de confusión y curva ROC."""
        plt.figure(figsize=(12, 4))

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

def main():
    """Función principal para ejecutar el modelo."""
    # Cargar el dataset
    data = pd.read_csv('src/Part C/data/cs-training.csv')

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
    print("\n=== Resultados de la Evaluación ===")
    for model_name, model_metrics in metrics.items():
        print(f"\nModelo: {model_name}")
        print(f"Accuracy: {model_metrics['Accuracy']:.4f}")
        print(f"Precision: {model_metrics['Precision']:.4f}")
        print(f"Recall: {model_metrics['Recall']:.4f}")
        print(f"F1-Score: {model_metrics['F1-Score']:.4f}")
        print(f"AUC-ROC: {model_metrics['AUC-ROC']:.4f}")
        print("Matriz de Confusión:")
        print(model_metrics['Confusion Matrix'])

    # Generar gráficos de evaluación
    lr_pred = model.logistic_model.predict(model.X_test)
    lr_prob = model.logistic_model.predict_proba(model.X_test)[:, 1]
    nn_prob = model.nn_model.predict(model.X_test, verbose=0).flatten()
    nn_pred = (nn_prob >= 0.5).astype(int)
    
    model.plot_evaluation("Logistic Regression", model.y_test, lr_pred, lr_prob)
    model.plot_evaluation("Neural Network", model.y_test, nn_pred, nn_prob)

if __name__ == "__main__":
    main()