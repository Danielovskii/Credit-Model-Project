# Modelo de Riesgo Crediticio
# Autores: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Fecha: 31 de marzo de 2025
# Descripción: Modelo de riesgo que expande el scorecard tradicional de la Parte A,
# Mediante una Red Neuronal obtiene la probabilidad de incumplimiento

                                                                # el modelo calcula métricas de riesgo como:
                                                                # - Probability of Default (PD)
                                                                # - Loss Given Default (LGD)
                                                                # - Exposure at Default (EAD)

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight


class CreditRiskMLModel:
    """
    Class implementing a credit risk model using machine learning.
    Now includes a more complex neural network architecture.
    """

    def __init__(self, random_state=42):
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
        self.features = None

    def preprocess_data(self, data: pd.DataFrame, target_col: str) -> None:
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        data = data[data["RevolvingUtilizationOfUnsecuredLines"] < 13]
        data = data[~data["NumberOfTimes90DaysLate"].isin([96, 98])]
        data = data[~data["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98])]
        data = data[~data["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])]

        debt_ratio_threshold = data["DebtRatio"].quantile(0.975)
        data = data[data["DebtRatio"] <= debt_ratio_threshold]

        X = data.drop(columns=[target_col])
        y = data[target_col]

        X["MonthlyIncome"] = self.imputer_median.fit_transform(X[["MonthlyIncome"]])
        X["NumberOfDependents"] = self.imputer_mode.fit_transform(X[["NumberOfDependents"]])

        self.features = X.columns  # Save feature names for consistency

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        classes = np.unique(self.y_train)
        self.class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
        self.class_weights = dict(zip(classes, self.class_weights))

    def train_logistic_regression(self) -> None:
        self.logistic_model.fit(self.X_train, self.y_train)

    def train_complex_neural_network(self) -> None:
        self.nn_model = Sequential([
            Dense(128, activation="relu", input_shape=(self.X_train.shape[1],), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])

        self.nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.nn_model.fit(
            self.X_train, self.y_train,
            epochs=5, batch_size=32, verbose=1,
            validation_split=0.2, class_weight=self.class_weights
        )

    def evaluate_models(self) -> dict:
        lr_pred = self.logistic_model.predict(self.X_test)
        lr_prob = self.logistic_model.predict_proba(self.X_test)[:, 1]

        nn_prob = self.nn_model.predict(self.X_test, verbose=0).flatten()
        nn_pred = (nn_prob >= 0.5).astype(int)

        metrics = {
            "Logistic Regression": {
                "Accuracy": accuracy_score(self.y_test, lr_pred),
                "Precision": precision_score(self.y_test, lr_pred),
                "Recall": recall_score(self.y_test, lr_pred),
                "F1-Score": f1_score(self.y_test, lr_pred),
                "AUC-ROC": roc_auc_score(self.y_test, lr_prob),
                "Confusion Matrix": confusion_matrix(self.y_test, lr_pred),
                "Actual": self.y_test,
                "Predicted Proba": lr_prob
            },
            "Neural Network": {
                "Accuracy": accuracy_score(self.y_test, nn_pred),
                "Precision": precision_score(self.y_test, nn_pred),
                "Recall": recall_score(self.y_test, nn_pred),
                "F1-Score": f1_score(self.y_test, nn_pred),
                "AUC-ROC": roc_auc_score(self.y_test, nn_prob),
                "Confusion Matrix": confusion_matrix(self.y_test, nn_pred),
                "Actual": self.y_test,
                "Predicted Proba": nn_prob
            }
        }
        self.plot_confusion_matrix(metrics)
        self.plot_roc_curves(metrics)

        return metrics

    def plot_confusion_matrix(self, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, (model_name, values) in enumerate(metrics.items()):
            cm = values["Confusion Matrix"]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Default", "Default"],
                        yticklabels=["No Default", "Default"], ax=axes[i])
            axes[i].set_title(f"{model_name} - Confusion Matrix")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, metrics):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for (ax, (model_name, values), color) in zip([ax1, ax2], metrics.items(), ['darkgreen', 'darkblue']):
            fpr, tpr, _ = roc_curve(values["Actual"], values["Predicted Proba"])
            auc = roc_auc_score(values["Actual"], values["Predicted Proba"])
            
            ax.plot(fpr, tpr, color=color, label=f'{model_name} ROC curve (AUC = {auc:.2f})', linewidth=2)
            ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.legend(loc="lower right")
            ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
        
        plt.tight_layout()
        plt.show()

    def save_model(self, filename="logistic_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.logistic_model, f)

    def test_on_external_data(self, filepath: str) -> pd.DataFrame:
        print("\nExpected input features:", list(self.features))
        test_data = pd.read_csv(filepath)
        if "Unnamed: 0" in test_data.columns:
            test_data = test_data.drop(columns=["Unnamed: 0"])

        has_target = "SeriousDlqin2yrs" in test_data.columns and test_data["SeriousDlqin2yrs"].notna().sum() > 0
        if has_target:
            test_data = test_data.dropna(subset=["SeriousDlqin2yrs"])
            y_true = test_data["SeriousDlqin2yrs"].astype(int)
            test_data = test_data.drop(columns=["SeriousDlqin2yrs"])
        else:
            y_true = None

        missing_cols = [col for col in self.features if col not in test_data.columns]
        if missing_cols:
            print(f"\nError: Missing columns in test data: {missing_cols}")
            return pd.DataFrame()

        test_data = test_data[self.features]
        test_data["MonthlyIncome"] = self.imputer_median.transform(test_data[["MonthlyIncome"]])
        test_data["NumberOfDependents"] = self.imputer_mode.transform(test_data[["NumberOfDependents"]])

        if test_data.empty:
            print("\nWarning: Test dataset is empty after preprocessing.")
            return pd.DataFrame()

        test_scaled = self.scaler.transform(test_data)

        # Logistic Regression
        lr_preds = self.logistic_model.predict(test_scaled)
        lr_probs = self.logistic_model.predict_proba(test_scaled)[:, 1]

        # Neural Network
        nn_probs = self.nn_model.predict(test_scaled, verbose=0).flatten()
        nn_preds = (nn_probs >= 0.5).astype(int)

        if y_true is not None:
            from pandas import DataFrame
            print("\n=== Test Set Evaluation ===")

            results = []

            results.append({
                "Model": "Logistic Regression",
                "Accuracy": accuracy_score(y_true, lr_preds),
                "Precision": precision_score(y_true, lr_preds),
                "Recall": recall_score(y_true, lr_preds),
                "F1-Score": f1_score(y_true, lr_preds),
                "AUC-ROC": roc_auc_score(y_true, lr_probs)
            })

            results.append({
                "Model": "Neural Network",
                "Accuracy": accuracy_score(y_true, nn_preds),
                "Precision": precision_score(y_true, nn_preds),
                "Recall": recall_score(y_true, nn_preds),
                "F1-Score": f1_score(y_true, nn_preds),
                "AUC-ROC": roc_auc_score(y_true, nn_probs)
            })

            df_results = DataFrame(results)
            print(df_results.to_string(index=False))

            print("\nConfusion Matrix: Logistic Regression")
            print(confusion_matrix(y_true, lr_preds))

            print("\nConfusion Matrix: Neural Network")
            print(confusion_matrix(y_true, nn_preds))

        return pd.DataFrame({"Prediction": nn_preds})

def main():
    data = pd.read_csv("src/Part C/data/cs-training.csv")
    model = CreditRiskMLModel()
    model.preprocess_data(data, target_col="SeriousDlqin2yrs")
    model.train_logistic_regression()
    model.train_complex_neural_network()
    metrics = model.evaluate_models()

    print("=== Model Evaluation Results ===")
    for model_name, model_metrics in metrics.items():
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {model_metrics['Accuracy']:.4f}")
        print(f"Precision: {model_metrics['Precision']:.4f}")
        print(f"Recall: {model_metrics['Recall']:.4f}")
        print(f"F1-Score: {model_metrics['F1-Score']:.4f}")
        print(f"AUC-ROC: {model_metrics['AUC-ROC']:.4f}")
        print("Confusion Matrix:")
        print(model_metrics['Confusion Matrix'])

    model.save_model("logistic_model.pkl")

    test_predictions = model.test_on_external_data("src/Part C/data/cs-test.csv")
    if not test_predictions.empty:
        print("\nPredictions on test data:")
        print(test_predictions.head())

if __name__ == "__main__":
    main()