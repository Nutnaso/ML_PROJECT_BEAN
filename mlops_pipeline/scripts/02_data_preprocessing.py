import os
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(random_state=42):
    """
    Preprocess Dry Bean dataset:
    - Remove identifier columns (e.g., 'id' or 'Bean ID')
    - Split data into stratified Train/Validation/Test (70/15/15)
    - Scale features (StandardScaler)
    - Save processed datasets and scaler object as MLflow artifacts
    """
    mlflow.set_experiment("Dry Bean Classification - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # ----------------------------
        # 1. Load data
        # ----------------------------
        base_dir = os.path.dirname(os.path.dirname(__file__))  # ../ from scripts
        excel_path = os.path.join(base_dir, "Dry_Bean_Dataset.xlsx")

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Dataset not found at {excel_path}")

        df = pd.read_excel(excel_path)
        print(f"Data loaded successfully from: {excel_path}")

        # ----------------------------
        # 2. Drop identifier column if exists
        # ----------------------------
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        if "Bean ID" in df.columns:
            df = df.drop(columns=["Bean ID"])

        # ----------------------------
        # 3. Split features & target
        # ----------------------------
        X = df.drop("Class", axis=1)   # ✅ Features
        y = df["Class"]                # ✅ Target

        # First split Train+Val and Test (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.15,
            stratify=y,
            random_state=random_state
        )

        # Then split Train and Validation (15/85 ≈ 0.176 ≈ 15% of total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.176,   # 0.176 * 85% ≈ 15%
            stratify=y_temp,
            random_state=random_state
        )

        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

        # ----------------------------
        # 4. Feature Scaling
        # ----------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame (for saving)
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_val = pd.DataFrame(X_val_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

        # ----------------------------
        # 5. Save processed data
        # ----------------------------
        processed_data_dir = os.path.join(base_dir, "processed_data")
        os.makedirs(processed_data_dir, exist_ok=True)

        pd.concat([X_train, y_train.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(processed_data_dir, "train.csv"), index=False)
        pd.concat([X_val, y_val.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(processed_data_dir, "val.csv"), index=False)
        pd.concat([X_test, y_test.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(processed_data_dir, "test.csv"), index=False)

        print(f"Processed data saved to '{processed_data_dir}'")

        # ----------------------------
        # 6. Save scaler
        # ----------------------------
        scaler_path = os.path.join(processed_data_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler object saved at {scaler_path}")

        # ----------------------------
        # 7. Log MLflow artifacts
        # ----------------------------
        mlflow.log_param("split_ratio", "70/15/15")
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("val_rows", len(X_val))
        mlflow.log_metric("test_rows", len(X_test))

        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")

        print("Preprocessing artifacts logged in MLflow.")
        print("-" * 50)
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)

        # ✅ GitHub Actions output
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)


if __name__ == "__main__":
    preprocess_data()
