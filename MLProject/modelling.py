import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================
# KONFIGURASI
# =========================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Nas-virat/Telco-Customer-Churn/main/"
    "Telco-Customer-Churn.csv"
)

EXPERIMENT_NAME = "Telco Customer Churn - Workflow CI"


# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_URL)

print(f"Dataset shape: {df.shape}")


# =========================
# PREPROCESSING
# =========================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# COLUMN TYPES
# =========================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns


# =========================
# PREPROCESSING PIPELINE
# =========================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# =========================
# MODEL
# =========================

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================
# MLFLOW RUN
# =========================

mlflow_run_id = os.getenv("MLFLOW_RUN_ID")

if mlflow_run_id:
    # =========================
    # MLflow Project
    # =========================

    print(
        f"MLflow Project Run ID: {mlflow_run_id}"
    )

    mlflow.start_run(
        run_id=mlflow_run_id
    )

    run = mlflow.active_run()

else:
    # =========================
    # Direct Python Execution
    # =========================

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    run = mlflow.start_run()


try:

    print(
        f"MLflow Run ID: {run.info.run_id}"
    )

    with open("mlflow_run_id.txt", "w", encoding="utf-8") as file:
        file.write(run.info.run_id)

    # =========================
    # TRAINING
    # =========================

    pipeline.fit(
        X_train,
        y_train
    )

    # =========================
    # PREDICTION
    # =========================

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    # =========================
    # EVALUATION
    # =========================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # =========================
    # LOG PARAMETERS
    # =========================

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "C",
        1.0
    )

    mlflow.log_param(
        "max_iter",
        1000
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    # =========================
    # LOG METRICS
    # =========================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    # =========================
    # LOG MODEL
    # =========================

    model_info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        serialization_format="cloudpickle"
    )

    with open("mlflow_model_uri.txt", "w", encoding="utf-8") as file:
        file.write(model_info.model_uri)

    # =========================
    # OUTPUT
    # =========================

    print("\nTraining completed.")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"Run ID   : {run.info.run_id}"
    )

finally:

    # Tutup run yang sedang aktif.
    if mlflow.active_run():

        mlflow.end_run()