# credit_risk_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

class CreditRiskMLModel:
    """
    A credit risk modeling class using multiple machine learning models:
    - Logistic Regression
    - Random Forest
    - Neural Network

    Includes data preprocessing, model training, evaluation, and ROC plotting.
    """

    def __init__(self, random_state=42):
        """
        Initializes preprocessing tools and ML model instances.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer_median = SimpleImputer(strategy="median")
        self.imputer_mode = SimpleImputer(strategy="most_frequent")
        self.logistic_model = LogisticRegression(random_state=self.random_state, class_weight="balanced", max_iter=1000)
        self.random_forest_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight="balanced")
        self.nn_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_weights = None

    def preprocess_data(self, data: pd.DataFrame, target_col: str) -> None:
        """
        Preprocesses the input dataset:
        - Removes extreme outliers
        - Imputes missing values
        - Splits data into train/test sets
        - Standardizes features
        - Computes class weights for imbalanced classification
        """
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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        classes = np.unique(self.y_train)
        self.class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
        self.class_weights = dict(zip(classes, self.class_weights))

    def train_logistic_regression(self):
        """Trains a Logistic Regression model."""
        self.logistic_model.fit(self.X_train, self.y_train)

    def train_random_forest(self):
        """Trains a Random Forest Classifier."""
        self.random_forest_model.fit(self.X_train, self.y_train)

    def train_neural_network(self):
        """
        Trains a simple feedforward Neural Network using Keras.
        Network has two hidden layers and uses dropout for regularization.
        """
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
            epochs=5, batch_size=32, verbose=1,
            validation_split=0.2, class_weight=self.class_weights
        )

    def evaluate_models(self) -> dict:
        """
        Evaluates model performance using the test set.
        Returns a dictionary of metrics per model: Accuracy, Precision,
        Recall, F1-Score, AUC-ROC, and Confusion Matrix.
        """
        lr_pred = self.logistic_model.predict(self.X_test)
        lr_prob = self.logistic_model.predict_proba(self.X_test)[:, 1]

        rf_pred = self.random_forest_model.predict(self.X_test)
        rf_prob = self.random_forest_model.predict_proba(self.X_test)[:, 1]

        nn_prob = self.nn_model.predict(self.X_test, verbose=0).flatten()
        nn_pred = (nn_prob >= 0.5).astype(int)

        return {
            "Logistic Regression": {
                "Accuracy": accuracy_score(self.y_test, lr_pred),
                "Precision": precision_score(self.y_test, lr_pred),
                "Recall": recall_score(self.y_test, lr_pred),
                "F1-Score": f1_score(self.y_test, lr_pred),
                "AUC-ROC": roc_auc_score(self.y_test, lr_prob),
                "Confusion Matrix": confusion_matrix(self.y_test, lr_pred)
            },
            "Random Forest": {
                "Accuracy": accuracy_score(self.y_test, rf_pred),
                "Precision": precision_score(self.y_test, rf_pred),
                "Recall": recall_score(self.y_test, rf_pred),
                "F1-Score": f1_score(self.y_test, rf_pred),
                "AUC-ROC": roc_auc_score(self.y_test, rf_prob),
                "Confusion Matrix": confusion_matrix(self.y_test, rf_pred)
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

    def plot_evaluation(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        """
        Generates confusion matrix and ROC curve for a given model.
        """
        plt.figure(figsize=(12, 4))
        cm = confusion_matrix(y_true, y_pred)
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def plot_combined_roc(self):
        """
        Plots all ROC curves together for comparison:
        - Logistic Regression
        - Random Forest
        - Neural Network
        """
        lr_prob = self.logistic_model.predict_proba(self.X_test)[:, 1]
        rf_prob = self.random_forest_model.predict_proba(self.X_test)[:, 1]
        nn_prob = self.nn_model.predict(self.X_test, verbose=0).flatten()

        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, lr_prob)
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, rf_prob)
        fpr_nn, tpr_nn, _ = roc_curve(self.y_test, nn_prob)

        auc_lr = roc_auc_score(self.y_test, lr_prob)
        auc_rf = roc_auc_score(self.y_test, rf_prob)
        auc_nn = roc_auc_score(self.y_test, nn_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
        plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
        plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {auc_nn:.2f})")
        plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def predict_on_new_data(model: CreditRiskMLModel, filepath: str, output_path: str = "data/cs-test-predicted.csv"):
    """
    Loads a new dataset, applies all trained models, and outputs predictions.
    Also prints executive insights and plots risk distribution across models.
    """
    test_df = pd.read_csv(filepath)
    test_df = test_df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")

    X_test = test_df.drop(columns=["Unnamed: 0"], errors="ignore")
    X_test["MonthlyIncome"] = X_test["MonthlyIncome"].fillna(X_test["MonthlyIncome"].median())
    X_test["NumberOfDependents"] = X_test["NumberOfDependents"].fillna(X_test["NumberOfDependents"].mode()[0])
    X_test_scaled = model.scaler.transform(X_test)

    test_df["Pred_Logistic"] = model.logistic_model.predict(X_test_scaled)
    test_df["Pred_RandomForest"] = model.random_forest_model.predict(X_test_scaled)
    test_df["Pred_NeuralNet"] = (model.nn_model.predict(X_test_scaled, verbose=0).flatten() >= 0.5).astype(int)

    print("\n===== Executive Summary from Test Predictions =====")
    for name in ["Pred_Logistic", "Pred_RandomForest", "Pred_NeuralNet"]:
        print(f"→ % risky clients predicted by {name.replace('Pred_', '')}: {test_df[name].mean() * 100:.2f}%")

    # Agreement
    agree_all = ((test_df["Pred_Logistic"] == test_df["Pred_RandomForest"]) &
                 (test_df["Pred_Logistic"] == test_df["Pred_NeuralNet"])).mean() * 100
    print(f"→ Agreement between all models: {agree_all:.2f}%")

    # Feature averages for risky clients
    print("\n→ Average features for risky clients (Logistic):")
    print(test_df[test_df["Pred_Logistic"] == 1][[
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome",
        "NumberOfDependents", "NumberOfOpenCreditLinesAndLoans"
    ]].mean().round(2).to_string())

    # Risk distribution chart with proper legend
    plt.figure(figsize=(8, 5))
    risk_counts = pd.DataFrame({
        "Logistic Regression": test_df["Pred_Logistic"].value_counts(normalize=True),
        "Random Forest": test_df["Pred_RandomForest"].value_counts(normalize=True),
        "Neural Network": test_df["Pred_NeuralNet"].value_counts(normalize=True)
    }).T * 100

    melted = risk_counts.reset_index().melt(id_vars="index")
    bar = sns.barplot(data=melted, x="index", y="value", hue="variable", palette=sns.color_palette("Blues", 2))
    plt.ylabel("Percentage")
    plt.xlabel("Model")
    plt.title("Risk Prediction Distribution per Model")
    handles, labels = bar.get_legend_handles_labels()
    labels = ["No Risk (0)" if l == "0" else "Risk (1)" for l in labels]
    bar.legend(handles=handles, labels=labels, title="Prediction")
    plt.tight_layout()
    plt.show()

    test_df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")

def main():
    """
    Main script to:
    - Load and preprocess training data
    - Train 3 ML models
    - Evaluate model performance
    - Predict on external test dataset
    - Generate visual and textual summaries
    """
    data = pd.read_csv("data/cs-training.csv")
    model = CreditRiskMLModel()
    model.preprocess_data(data, target_col="SeriousDlqin2yrs")
    model.train_logistic_regression()
    model.train_random_forest()
    model.train_neural_network()

    metrics = model.evaluate_models()
    for name, result in metrics.items():
        print(f"\nModel: {name}")
        for k, v in result.items():
            print(f"{k}: {v if k != 'Confusion Matrix' else ''}")
            if k == "Confusion Matrix":
                print(v)

    model.plot_evaluation("Logistic Regression", model.y_test,
                          model.logistic_model.predict(model.X_test),
                          model.logistic_model.predict_proba(model.X_test)[:, 1])
    model.plot_evaluation("Random Forest", model.y_test,
                          model.random_forest_model.predict(model.X_test),
                          model.random_forest_model.predict_proba(model.X_test)[:, 1])
    model.plot_evaluation("Neural Network", model.y_test,
                          (model.nn_model.predict(model.X_test) >= 0.5).astype(int),
                          model.nn_model.predict(model.X_test).flatten())
    model.plot_combined_roc()

    predict_on_new_data(model, "data/cs-test.csv")

if __name__ == "__main__":
    main()
