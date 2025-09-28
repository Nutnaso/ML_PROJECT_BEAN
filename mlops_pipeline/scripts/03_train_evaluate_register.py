# 03_train_evaluate_register.py (แก้ไขสมบูรณ์)

import sys
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def train_and_evaluate_models(preprocessing_run_id, random_state=42):
    """
    Train multiple models, evaluate them, and register the best one in MLflow Model Registry.
    """

    mlflow.set_experiment("Dry Bean Classification - Model Training")

    # 1. โหลดข้อมูลจาก Preprocessing Artifacts
    try:
        local_artifact_path = download_artifacts(
            run_id=preprocessing_run_id,
            artifact_path="processed_data"
        )
        print(f"Artifacts downloaded to: {local_artifact_path}")

        train_path = os.path.join(local_artifact_path, "train.csv")
        val_path = os.path.join(local_artifact_path, "val.csv")
        test_path = os.path.join(local_artifact_path, "test.csv")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Successfully loaded train/val/test datasets.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        sys.exit(1)

    # Split features/labels
    X_train, y_train = train_df.drop("Class", axis=1), train_df["Class"]
    X_val, y_val = val_df.drop("Class", axis=1), val_df["Class"]
    X_test, y_test = test_df.drop("Class", axis=1), test_df["Class"]

    # LabelEncoder สำหรับ XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # แปลงเป็น pandas Series เพื่อ concat
    y_train_enc_series = pd.Series(y_train_enc, index=X_train.index)
    y_val_enc_series = pd.Series(y_val_enc, index=X_val.index)

    # 2. กำหนด candidate models
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0, random_state=random_state, max_iter=10000, multi_class="multinomial"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state
        ),
        "SVC": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="mlogloss", random_state=random_state
        )
    }

    best_model_name = None
    best_model = None
    best_score = 0.0

    # 3. เทรนและ log โมเดลแต่ละตัว
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_training", nested=True):
            print(f"Training model: {name}")
            mlflow.set_tag("ml.step", "model_training_evaluation")
            mlflow.log_param("model_name", name)

            # ใช้ labels encoded สำหรับ XGBoost, ตัวอื่นใช้ string labels
            if name == "XGBoost":
                model.fit(pd.concat([X_train, X_val]), pd.concat([y_train_enc_series, y_val_enc_series]))
                y_pred_enc = model.predict(X_test)
                y_pred = le.inverse_transform(y_pred_enc)
            else:
                model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
                y_pred = model.predict(X_test)

            # คำนวณ metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")

            print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = os.path.join(local_artifact_path, f"{name}_classification_report.csv")
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path, artifact_path="reports")

            # Log model
            mlflow.sklearn.log_model(model, artifact_path=f"{name}_model")

            # Track best model
            if acc > best_score:
                best_score = acc
                best_model_name = name
                best_model = model

    # 4. ลงทะเบียนโมเดลที่ดีที่สุด
    if best_model is not None:
        print(f"Best model is {best_model_name} with accuracy {best_score:.4f}")
        with mlflow.start_run(run_name="register_best_model") as run:
            model_uri = f"runs:/{run.info.run_id}/{best_model_name}_model"
            mlflow.register_model(
                model_uri=model_uri,
                name="dry-bean-classifier-prod"
            )
    else:
        print("No model met the performance requirement.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    train_and_evaluate_models(preprocessing_run_id=run_id)
