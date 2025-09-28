import sys
import mlflow
from mlflow.tracking import MlflowClient


def transition_model_alias(model_name, alias="champion", metric="accuracy"):
    """
    Promote the best model version to a given alias, based on metric value.
    Default alias = "champion", metric = "accuracy".
    """

    client = MlflowClient()
    try:
        # ค้นหา versions ของ model
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"❌ No versions found for model '{model_name}'.")
            sys.exit(1)

        # ดึง run_id จากแต่ละ version เพื่อดู metric
        best_version = None
        best_metric = -float("inf")

        for v in versions:
            run_id = v.run_id
            try:
                run = mlflow.get_run(run_id)
                metric_val = run.data.metrics.get(metric)
                if metric_val is not None and metric_val > best_metric:
                    best_metric = metric_val
                    best_version = v
            except Exception as e:
                print(f"⚠️ Could not fetch run {run_id}: {e}")

        if best_version is None:
            print(f"❌ No valid metric '{metric}' found for model '{model_name}'.")
            sys.exit(1)

        version_number = best_version.version
        print(f"✅ Best model version: {version_number} (metric {metric}={best_metric:.4f})")

        # ตั้ง alias
        print(f"Promoting model '{model_name}' v{version_number} → alias '{alias}' ...")
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version_number
        )
        print(f"🎉 Successfully set alias '{alias}' for model '{model_name}' version {version_number}.")

    except Exception as e:
        print(f"❌ Error during alias transition: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/04_transition_model.py <model_name> [alias] [metric]")
        sys.exit(1)

    model_name_arg = sys.argv[1]
    target_alias_arg = sys.argv[2] if len(sys.argv) > 2 else "champion"
    target_metric_arg = sys.argv[3] if len(sys.argv) > 3 else "accuracy"

    transition_model_alias(model_name_arg, alias=target_alias_arg, metric=target_metric_arg)
