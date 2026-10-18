from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from retailpulse.config import get_settings
from retailpulse.data import load_online_retail_ii
from retailpulse.monitoring import build_numeric_reference, feature_drift_report
from retailpulse.tracking import log_mlflow_run


CHURN_FEATURES = ("recency_days", "frequency", "monetary", "observation_month")


def _valid_transactions(data: pd.DataFrame) -> pd.DataFrame:
    required = {"invoice_date", "invoice", "customer_id", "sales"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Churn training data is missing columns: {', '.join(sorted(missing))}")
    frame = data.dropna(subset=["invoice_date", "invoice", "customer_id", "sales"]).copy()
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.dropna(subset=["invoice_date", "sales"])
    return frame.loc[frame["sales"].gt(0)].copy()


def churn_data_fingerprint(data: pd.DataFrame) -> str:
    transactions = _valid_transactions(data)
    values = transactions[["invoice_date", "invoice", "customer_id", "sales"]].copy()
    values["customer_id"] = values["customer_id"].astype(str)
    values = values.sort_values(["invoice_date", "invoice", "customer_id", "sales"])
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def build_churn_snapshots(data: pd.DataFrame, horizon_days: int = 90, step_days: int = 30) -> pd.DataFrame:
    """Build customer snapshots labelled by purchase behavior in the following horizon."""
    if horizon_days < 30 or step_days < 1:
        raise ValueError("Churn horizon must be at least 30 days and snapshot step must be positive.")
    transactions = _valid_transactions(data)
    if transactions.empty:
        raise ValueError("No valid customer transactions are available for churn training.")

    transactions["customer_id"] = transactions["customer_id"].astype(str)
    transactions["invoice_day"] = transactions["invoice_date"].dt.normalize()
    first_day = transactions["invoice_day"].min()
    last_day = transactions["invoice_day"].max()
    snapshots = pd.date_range(first_day + pd.Timedelta(days=180), last_day - pd.Timedelta(days=horizon_days), freq=f"{step_days}D")
    if len(snapshots) < 4:
        raise ValueError("At least four valid monthly snapshots are required to train and evaluate churn.")

    rows: list[pd.DataFrame] = []
    for snapshot_date in snapshots:
        observed = transactions.loc[transactions["invoice_day"].le(snapshot_date)]
        future = transactions.loc[
            transactions["invoice_day"].gt(snapshot_date)
            & transactions["invoice_day"].le(snapshot_date + pd.Timedelta(days=horizon_days))
        ]
        grouped = observed.groupby("customer_id").agg(
            last_purchase=("invoice_day", "max"),
            frequency=("invoice", "nunique"),
            monetary=("sales", "sum"),
        )
        grouped["recency_days"] = (snapshot_date - grouped.pop("last_purchase")).dt.days
        future_customers = set(future["customer_id"].unique())
        grouped["churned"] = (~grouped.index.isin(future_customers)).astype("int8")
        grouped["observation_month"] = snapshot_date.month
        grouped["snapshot_date"] = snapshot_date
        rows.append(grouped.reset_index())

    snapshots_frame = pd.concat(rows, ignore_index=True)
    if snapshots_frame["churned"].nunique() < 2:
        raise ValueError("The observed snapshots do not contain both churn and retained customers.")
    return snapshots_frame


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(labels, probabilities)),
        "prevalence": float(labels.mean()),
    }
    if np.unique(labels).size > 1:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(average_precision_score(labels, probabilities))
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")
    return result


def train_churn_model(
    data: pd.DataFrame,
    model_path: str | Path,
    horizon_days: int = 90,
) -> dict[str, object]:
    snapshots = build_churn_snapshots(data, horizon_days=horizon_days)
    dates = sorted(snapshots["snapshot_date"].unique())
    split_index = max(2, int(len(dates) * 0.75))
    if split_index >= len(dates):
        raise ValueError("Not enough chronological snapshots for a held-out churn evaluation.")
    split_date = pd.Timestamp(dates[split_index])
    train = snapshots.loc[
        snapshots["snapshot_date"].add(pd.Timedelta(days=horizon_days)).le(split_date)
    ]
    test = snapshots.loc[snapshots["snapshot_date"].ge(split_date)]
    if train["churned"].nunique() < 2 or test["churned"].nunique() < 2:
        raise ValueError("Both churn outcomes must appear in train and chronological test periods.")

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=140,
        max_leaf_nodes=12,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=42,
    )
    model.fit(train.loc[:, CHURN_FEATURES], train["churned"])
    probabilities = model.predict_proba(test.loc[:, CHURN_FEATURES])[:, 1]
    metrics = _metrics(test["churned"].to_numpy(), probabilities)
    fingerprint_values = pd.util.hash_pandas_object(
        snapshots[["customer_id", "snapshot_date", *CHURN_FEATURES, "churned"]], index=False
    ).to_numpy().tobytes()

    final_model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=140,
        max_leaf_nodes=12,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=42,
    )
    final_model.fit(snapshots.loc[:, CHURN_FEATURES], snapshots["churned"])
    artifact = {
        "model": final_model,
        "feature_columns": CHURN_FEATURES,
        "horizon_days": horizon_days,
        "target_definition": f"No purchase during the {horizon_days} days after the monthly observation date",
        "train_start": train["snapshot_date"].min().date().isoformat(),
        "train_end": train["snapshot_date"].max().date().isoformat(),
        "label_horizon_end": (train["snapshot_date"].max() + pd.Timedelta(days=horizon_days)).date().isoformat(),
        "validation_start": test["snapshot_date"].min().date().isoformat(),
        "validation_end": test["snapshot_date"].max().date().isoformat(),
        "snapshot_count": len(snapshots),
        "customer_count": int(snapshots["customer_id"].nunique()),
        "feature_reference": {
            feature: build_numeric_reference(train[feature]) for feature in CHURN_FEATURES
        },
        "metrics": metrics,
        "snapshot_fingerprint": hashlib.sha256(fingerprint_values).hexdigest(),
        "data_fingerprint": churn_data_fingerprint(data),
        "trained_on": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    destination = Path(model_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as temporary:
            temp_name = temporary.name
        joblib.dump(artifact, temp_name)
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    log_mlflow_run(
        run_name="customer-churn-classifier",
        parameters={"model": "HistGradientBoostingClassifier", "horizon_days": horizon_days},
        metrics=metrics,
        metadata={key: value for key, value in artifact.items() if key != "model"},
        artifact_paths=[destination],
    )
    return {key: value for key, value in artifact.items() if key != "model"}


def score_current_customers(data: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    transactions = _valid_transactions(data)
    if transactions.empty:
        return pd.DataFrame(columns=["customer_id", "recency_days", "frequency", "monetary", "churn_probability"])
    snapshot_date = transactions["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    transactions["customer_id"] = transactions["customer_id"].astype(str)
    current = transactions.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("sales", "sum"),
    )
    current["recency_days"] = (snapshot_date - current.pop("last_purchase").dt.normalize()).dt.days
    current["observation_month"] = snapshot_date.month
    probabilities = artifact["model"].predict_proba(current.loc[:, CHURN_FEATURES])[:, 1]
    current["churn_probability"] = probabilities
    current["churn_risk"] = np.where(probabilities >= 0.5, "High risk", "Lower risk")
    return current.reset_index().sort_values("churn_probability", ascending=False).reset_index(drop=True)


def monitor_current_churn_features(data: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    scored = score_current_customers(data, artifact)
    if scored.empty:
        return pd.DataFrame(columns=["feature", "psi", "status"])
    return feature_drift_report(artifact["feature_reference"], scored)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train and chronologically evaluate the RetailPulse churn classifier.")
    parser.add_argument("--source", type=Path, default=settings.online_retail_ii_path)
    parser.add_argument("--model-path", type=Path, default=settings.cache_dir / "churn_online_retail_ii.joblib")
    parser.add_argument("--horizon-days", type=int, default=90)
    arguments = parser.parse_args()
    result = train_churn_model(
        load_online_retail_ii(arguments.source),
        arguments.model_path,
        horizon_days=arguments.horizon_days,
    )
    print(json.dumps(result, indent=2))