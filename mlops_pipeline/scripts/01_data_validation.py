import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
from pathlib import Path

def validate_data():
    mlflow.set_experiment("Dry Bean Classification - Data Validation")

    # base_dir ของ repo (root ของ mlops_pipeline)
    base_dir = Path(__file__).resolve().parent.parent  # ../ จาก scripts/
    data_path = base_dir / "Dry_Bean_Dataset.xlsx"

    # folder สำหรับเก็บ report
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "validation_report.json"

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # โหลด dataset
        df = pd.read_excel(data_path)
        print("Data loaded successfully.")

        # ลบ Bean ID (identifier)
        if "Bean ID" in df.columns:
            df = df.drop(columns=["Bean ID"])

        # ข้อมูลพื้นฐาน
        num_rows, num_cols = df.shape
        target_col = "Class"
        num_classes = df[target_col].nunique()
        missing_values = df.isnull().sum().sum()

        # ตรวจสอบ class distribution
        class_dist = df[target_col].value_counts(normalize=True).to_dict()

        # ตรวจสอบ outliers (Z-score > 3)
        numeric_df = df.drop(columns=[target_col])
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        outlier_counts = (z_scores > 3).sum().to_dict()
        total_outliers = int((z_scores > 3).sum().sum())

        # Scaling (StandardScaler)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled, columns=numeric_df.columns)

        # สร้าง validation report
        validation_report = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "num_classes": num_classes,
            "missing_values": int(missing_values),
            "class_distribution": class_dist,
            "outlier_counts": outlier_counts,
            "total_outliers": total_outliers,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "validation_status": "Success"
            if missing_values == 0 and num_classes >= 2
            else "Failed",
        }

        # log metrics
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("num_classes", num_classes)
        mlflow.log_metric("total_outliers", total_outliers)
        mlflow.log_param("validation_status", validation_report["validation_status"])

        # บันทึก report เป็น JSON แล้ว log เข้า MLflow
        with open(report_path, "w") as f:
            json.dump(validation_report, f, indent=4)
        mlflow.log_artifact(str(report_path))

        print(json.dumps(validation_report, indent=4))
        print("Data validation run finished.")

if __name__ == "__main__":
    validate_data()
