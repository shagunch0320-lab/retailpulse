from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from retailpulse.analytics import forecast_daily_sales
from retailpulse.config import get_settings
from retailpulse.tracking import log_mlflow_run


LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)
FEATURE_COLUMNS = (
    "trend_days",
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "mean_7",
    "mean_28",
    "std_28",
)
MIN_RELATIVE_WAPE_IMPROVEMENT = 0.05


def _daily_series(history: pd.DataFrame) -> pd.Series:
    if not {"date", "sales"}.issubset(history.columns):
        raise ValueError("Daily sales history must contain 'date' and 'sales' columns.")
    daily = history[["date", "sales"]].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["sales"] = pd.to_numeric(daily["sales"], errors="coerce")
    daily = daily.dropna(subset=["date", "sales"])
    if daily.empty:
        raise ValueError("No valid daily sales observations were supplied.")
    daily = daily.groupby("date")["sales"].sum().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(calendar, fill_value=0.0).astype(float)


def daily_series_fingerprint(history: pd.DataFrame) -> str:
    daily = _daily_series(history)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(daily, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _features_for_date(history: pd.Series, target_date: pd.Timestamp, origin: pd.Timestamp) -> dict[str, float]:
    features: dict[str, float] = {
        "trend_days": float((target_date - origin).days),
        "day_of_week": float(target_date.dayofweek),
        "day_of_month": float(target_date.day),
        "month": float(target_date.month),
        "day_of_year_sin": float(np.sin(2 * np.pi * target_date.dayofyear / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * target_date.dayofyear / 365.25)),
    }
    for lag in LAGS:
        features[f"lag_{lag}"] = float(history.get(target_date - pd.Timedelta(days=lag), 0.0))
    for window in WINDOWS:
        values = history.loc[target_date - pd.Timedelta(days=window): target_date - pd.Timedelta(days=1)]
        features[f"mean_{window}"] = float(values.mean()) if not values.empty else 0.0
        if window == 28:
            features["std_28"] = float(values.std(ddof=0)) if not values.empty else 0.0
    return features


def _training_frame(daily: pd.Series) -> pd.DataFrame:
    origin = daily.index.min()
    rows = []
    for target_date in daily.index[28:]:
        rows.append(
            {
                **_features_for_date(daily.loc[:target_date - pd.Timedelta(days=1)], target_date, origin),
                "target": float(daily.loc[target_date]),
                "target_date": target_date,
            }
        )
    return pd.DataFrame(rows)


def _recursive_model_forecast(
    model: HistGradientBoostingRegressor,
    history: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    origin: pd.Timestamp,
) -> np.ndarray:
    context = history.copy()
    predictions: list[float] = []
    for target_date in forecast_dates:
        feature_row = _features_for_date(context, target_date, origin)
        value = float(model.predict(pd.DataFrame([feature_row], columns=FEATURE_COLUMNS))[0])
        value = max(0.0, value)
        predictions.append(value)
        context.loc[target_date] = value
    return np.asarray(predictions)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    denominator = max(float(np.abs(actual).sum()), 1e-9)
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "wape_pct": float(100 * np.abs(errors).sum() / denominator),
    }


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=42,
    )


def train_demand_model(
    history: pd.DataFrame,
    model_path: str | Path,
    holdout_days: int = 90,
) -> dict[str, object]:
    """Train on past observations and recursively score the final holdout period."""
    daily = _daily_series(history)
    if holdout_days < 14:
        raise ValueError("The temporal holdout must be at least 14 days.")
    folds = min(3, (len(daily) - 28 - 60) // holdout_days)
    if folds < 1:
        raise ValueError("At least 60 training days beyond the 28-day feature history and holdout are required.")

    supervised = _training_frame(daily)
    model_actuals: list[np.ndarray] = []
    model_predictions_by_fold: list[np.ndarray] = []
    baseline_predictions_by_fold: list[np.ndarray] = []
    fold_results: list[dict[str, object]] = []

    for fold_number in range(folds, 0, -1):
        validation_start_index = len(daily) - fold_number * holdout_days
        split_date = daily.index[validation_start_index]
        train = supervised.loc[supervised["target_date"].lt(split_date)]
        if len(train) < 60:
            raise ValueError("Not enough training observations before a rolling validation fold.")
        model = _new_model()
        model.fit(train.loc[:, FEATURE_COLUMNS], train["target"])
        test_dates = daily.index[validation_start_index:validation_start_index + holdout_days]
        train_history = daily.loc[:split_date - pd.Timedelta(days=1)]
        model_predictions = _recursive_model_forecast(model, train_history, test_dates, daily.index.min())
        weekday_means = train_history.groupby(train_history.index.dayofweek).mean()
        baseline_predictions = np.asarray(
            [float(weekday_means.get(timestamp.dayofweek, train_history.mean())) for timestamp in test_dates]
        )
        actual = daily.loc[test_dates].to_numpy(dtype=float)
        model_actuals.append(actual)
        model_predictions_by_fold.append(model_predictions)
        baseline_predictions_by_fold.append(baseline_predictions)
        fold_results.append(
            {
                "validation_start": test_dates[0].date().isoformat(),
                "validation_end": test_dates[-1].date().isoformat(),
                "model": _metrics(actual, model_predictions),
                "weekday_baseline": _metrics(actual, baseline_predictions),
            }
        )

    actual_all = np.concatenate(model_actuals)
    model_predictions_all = np.concatenate(model_predictions_by_fold)
    baseline_predictions_all = np.concatenate(baseline_predictions_by_fold)
    rolling_model_metrics = _metrics(actual_all, model_predictions_all)
    rolling_baseline_metrics = _metrics(actual_all, baseline_predictions_all)
    model_selected = rolling_model_metrics["wape_pct"] <= (
        rolling_baseline_metrics["wape_pct"] * (1.0 - MIN_RELATIVE_WAPE_IMPROVEMENT)
    )
    final_fold = fold_results[-1]
    model_metrics = final_fold["model"]
    baseline_metrics = final_fold["weekday_baseline"]
    residual_quantile = float(np.quantile(np.abs(model_predictions_all - actual_all), 0.9))
    if model_selected:
        final_model = _new_model()
        final_model.fit(supervised.loc[:, FEATURE_COLUMNS], supervised["target"])
    else:
        final_model = model

    artifact = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "model_selected": model_selected,
        "data_fingerprint": daily_series_fingerprint(history),
        "observations": len(daily),
        "holdout_days": holdout_days,
        "rolling_folds": fold_results,
        "rolling_metrics": {"model": rolling_model_metrics, "weekday_baseline": rolling_baseline_metrics},
        "train_start": daily.index.min().date().isoformat(),
        "train_end": (pd.Timestamp(fold_results[-1]["validation_start"]) - pd.Timedelta(days=1)).date().isoformat(),
        "validation_start": fold_results[-1]["validation_start"],
        "validation_end": fold_results[-1]["validation_end"],
        "metrics": {"model": model_metrics, "weekday_baseline": baseline_metrics},
        "residual_q90": residual_quantile,
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
        run_name="daily-demand-forecast",
        parameters={"model": "HistGradientBoostingRegressor", "holdout_days": holdout_days, "folds": folds},
        metrics={
            "rolling_model_wape_pct": rolling_model_metrics["wape_pct"],
            "rolling_baseline_wape_pct": rolling_baseline_metrics["wape_pct"],
            "selected_model": float(model_selected),
        },
        metadata={key: value for key, value in artifact.items() if key != "model"},
        artifact_paths=[destination],
    )

    return {key: value for key, value in artifact.items() if key != "model"}


def load_demand_model(model_path: str | Path) -> dict[str, object] | None:
    path = Path(model_path).expanduser()
    if not path.is_file():
        return None
    return joblib.load(path)


def forecast_with_demand_model(
    history: pd.DataFrame,
    artifact: dict[str, object],
    horizon: int = 30,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")
    if not artifact["model_selected"]:
        return forecast_daily_sales(history, horizon)

    daily = _daily_series(history)
    dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    predictions = _recursive_model_forecast(artifact["model"], daily, dates, daily.index.min())
    residual_q90 = float(artifact["residual_q90"])
    return pd.DataFrame(
        {
            "date": dates,
            "forecast": predictions,
            "lower": np.maximum(0.0, predictions - residual_q90),
            "upper": predictions + residual_q90,
        }
    )


def daily_sales_from_database(database_path: str | Path) -> pd.DataFrame:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "transactions" not in tables:
            raise ValueError(f"No transactions table found in {database_path}. Load the dataset bundle first.")
        return connection.execute(
            "SELECT CAST(date AS DATE) AS date, SUM(TRY_CAST(total_value AS DOUBLE)) AS sales "
            "FROM transactions GROUP BY 1 ORDER BY 1"
        ).df()
    finally:
        connection.close()


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train and backtest the RetailPulse daily demand model.")
    parser.add_argument(
        "--database",
        type=Path,
        default=settings.cache_dir / "retailpulse_bundle.duckdb",
        help="DuckDB bundle cache created by loading the dataset in RetailPulse",
    )
    parser.add_argument("--model-path", type=Path, default=settings.cache_dir / "demand_forecast.joblib")
    parser.add_argument("--holdout-days", type=int, default=90)
    arguments = parser.parse_args()
    result = train_demand_model(
        daily_sales_from_database(arguments.database),
        arguments.model_path,
        holdout_days=arguments.holdout_days,
    )
    print(json.dumps(result, indent=2))
    if not result["model_selected"]:
        print("The model did not beat the weekday baseline; the dashboard will continue using that baseline.")


if __name__ == "__main__":
    main()