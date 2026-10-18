from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import warnings


def log_mlflow_run(
    *,
    run_name: str,
    parameters: dict[str, object],
    metrics: dict[str, float],
    metadata: dict[str, object],
    artifact_paths: list[str | Path] | None = None,
) -> bool:
    """Log a run when MLflow tracking is configured; otherwise remain a no-op."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return False
    try:
        import mlflow
    except ImportError as error:
        warnings.warn("MLflow tracking is configured but MLflow is not installed; run was not logged.", RuntimeWarning)
        raise RuntimeError("Install the optional MLflow dependency with `pip install -e '.[mlops]'`.") from error

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "retailpulse"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path = Path(temporary_directory) / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(metadata_path))
        for artifact_path in artifact_paths or []:
            mlflow.log_artifact(str(artifact_path))
    return True