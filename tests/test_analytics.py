import math

import joblib
import pandas as pd

from retailpulse.analytics import build_customer_rfm, build_inventory_view, forecast_daily_sales
from retailpulse.demand_model import forecast_with_demand_model, train_demand_model
from retailpulse.churn_model import build_churn_snapshots, score_current_customers, train_churn_model
from retailpulse.monitoring import build_numeric_reference, population_stability_index
from retailpulse.etl import load_prepared_online_retail_ii, prepare_online_retail_ii
from retailpulse.tracking import log_mlflow_run
from retailpulse.data import load_online_retail_ii, normalize_retail_data
from retailpulse.pipeline import process_file
from retailpulse.config import get_settings


def sample_data() -> pd.DataFrame:
    return normalize_retail_data(
        pd.DataFrame(
            {
                "InvoiceNo": ["A", "A", "B", "C"],
                "InvoiceDate": ["2024-01-01", "2024-01-01", "2024-01-15", "2024-01-16"],
                "CustomerID": [1, 1, 1, 2],
                "Description": ["Mug", "Plate", "Mug", "Plate"],
                "Quantity": [2, 1, 3, 1],
                "UnitPrice": [5, 10, 5, 10],
                "Country": ["UK"] * 4,
            }
        )
    )


def test_normalizes_online_retail_fields_and_calculates_sales() -> None:
    data = sample_data()
    assert list(data["sales"]) == [10, 10, 15, 10]
    assert data["invoice_date"].notna().all()


def test_rfm_uses_invoice_frequency_and_recent_purchase() -> None:
    rfm = build_customer_rfm(sample_data())
    first_customer = rfm.loc[rfm["customer_id"].eq("1")].iloc[0]
    assert first_customer["frequency"] == 2
    assert first_customer["recency_days"] == 2


def test_inventory_view_uses_recent_units() -> None:
    inventory = build_inventory_view(sample_data())
    assert inventory.loc[inventory["description"].eq("Mug"), "units_sold_30d"].iloc[0] == 5


def test_forecast_requires_two_weeks_of_history() -> None:
    history = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=13), "sales": range(13)})
    try:
        forecast_daily_sales(history)
    except ValueError as error:
        assert "14 distinct days" in str(error)
    else:
        raise AssertionError("A short history should not produce a forecast.")


def test_normalizes_aggregated_currency_dataset() -> None:
    data = normalize_retail_data(
        pd.DataFrame(
            {
                "Year": [2010],
                "Month": ["December"],
                "Day": [1],
                "Year Month": ["2010-12"],
                "Sum of Quantity": [6],
                "Sum of UnitPrice": ["₹ 8.5"],
                "Sum of Total Sales": ["₹ 51"],
                "Sum of Estimated Cost": ["₹ 38.25"],
                "Sum of Estimated Profit": ["₹ 12.75"],
                "Description": ["CHRISTMAS LIGHTS 10 REINDEER"],
            }
        )
    )
    assert data.loc[0, "invoice_date"] == pd.Timestamp("2010-12-01")
    assert data.loc[0, "sales"] == 51
    assert data.loc[0, "profit"] == 12.75


def test_daily_date_components_take_precedence_over_year_month() -> None:
    data = normalize_retail_data(
        pd.DataFrame(
            {
                "Year": [2024, 2024],
                "Month": ["January", "January"],
                "Day": [3, 4],
                "Year Month": ["2024-01", "2024-01"],
                "Sum of Total Sales": [10, 20],
            }
        )
    )
    assert list(data["invoice_date"]) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]


def test_forecast_builds_thirty_day_weekday_baseline() -> None:
    history = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=28), "sales": range(28)})
    forecast = forecast_daily_sales(history, horizon=30)
    assert len(forecast) == 30
    assert forecast["date"].min() == pd.Timestamp("2024-01-29")
    assert forecast["lower"].le(forecast["forecast"]).all()


def test_etl_writes_normalized_csv(tmp_path) -> None:
    source = tmp_path / "raw.csv"
    destination = tmp_path / "processed" / "clean.csv"
    pd.DataFrame(
        {"InvoiceDate": ["2024-01-01"], "Quantity": [2], "UnitPrice": [5]}
    ).to_csv(source, index=False)

    row_count = process_file(source, destination)

    assert row_count == 1
    assert pd.read_csv(destination).loc[0, "sales"] == 10


def test_runtime_settings_use_environment_paths_and_allowlist(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RETAILPULSE_TRANSACTION_PATH", str(tmp_path / "facts.csv"))
    monkeypatch.setenv("RETAILPULSE_REFERENCE_DIR", str(tmp_path / "reference"))
    monkeypatch.setenv("RETAILPULSE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RETAILPULSE_ONLINE_RETAIL_II_PATH", str(tmp_path / "online_retail_II.xlsx"))
    monkeypatch.setenv("RETAILPULSE_REQUIRE_LOGIN", "true")
    monkeypatch.setenv("RETAILPULSE_ALLOWED_EMAILS", "ops@example.com, Analyst@example.com ")

    settings = get_settings()

    assert settings.transaction_path == tmp_path / "facts.csv"
    assert settings.reference_dir == tmp_path / "reference"
    assert settings.cache_dir == tmp_path / "cache"
    assert settings.online_retail_ii_path == tmp_path / "online_retail_II.xlsx"
    assert settings.require_login is True
    assert settings.allowed_emails == frozenset({"ops@example.com", "analyst@example.com"})


def test_online_retail_ii_combines_both_year_sheets_and_filters_returns(tmp_path) -> None:
    workbook_path = tmp_path / "online_retail_II.xlsx"
    columns = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    first_year = pd.DataFrame(
        [
            ["489434", "A1", "First year item", 2, "2009-12-01 07:45:00", 6.95, 13085, "United Kingdom"],
            ["C489435", "A1", "Cancelled item", -1, "2010-01-02 09:00:00", 6.95, 13085, "United Kingdom"],
        ],
        columns=columns,
    )
    second_year = pd.DataFrame(
        [["536365", "B2", "Second year item", 3, "2010-12-01 08:26:00", 2.55, 17850, "United Kingdom"]],
        columns=columns,
    )
    with pd.ExcelWriter(workbook_path) as writer:
        first_year.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        second_year.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    data = load_online_retail_ii(workbook_path)

    assert len(data) == 2
    assert data["invoice_date"].min() == pd.Timestamp("2009-12-01 07:45:00")
    assert data["invoice_date"].max() == pd.Timestamp("2010-12-01 08:26:00")
    assert data["sales"].round(2).tolist() == [13.9, 7.65]
    assert data["customer_id"].notna().all()
    assert data["invoice"].dtype.name == "string"
    assert data["sku_id"].dtype.name == "string"


def test_online_retail_ii_config_defaults_to_primary_workbook(monkeypatch) -> None:
    monkeypatch.delenv("RETAILPULSE_ONLINE_RETAIL_II_PATH", raising=False)
    settings = get_settings()
    assert settings.online_retail_ii_path.name == "online_retail_II.xlsx"
    assert settings.online_retail_ii_path.is_file()


def test_online_retail_ii_etl_writes_parquet_and_manifest(tmp_path) -> None:
    workbook_path = tmp_path / "source.xlsx"
    columns = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    rows = pd.DataFrame(
        [["I1", "A1", "Test item", 2, "2010-01-01 12:00:00", 5.0, 1, "UK"]],
        columns=columns,
    )
    with pd.ExcelWriter(workbook_path) as writer:
        rows.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        rows.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    output_path = tmp_path / "processed" / "transactions.parquet"

    manifest = prepare_online_retail_ii(workbook_path, output_path)

    parquet_data = pd.read_parquet(output_path)
    assert manifest["rows"] == 2
    assert manifest["invoice_count"] == 1
    assert len(parquet_data) == 2
    assert output_path.with_suffix(".parquet.manifest.json").is_file()
    cached_data = load_prepared_online_retail_ii(workbook_path, output_path)
    assert len(cached_data) == 2


def test_demand_model_trains_with_temporal_holdout_and_forecasts(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=420, freq="D")
    day_index = range(len(dates))
    sales = [
        100_000 + index * 450 + (20_000 if timestamp.dayofweek in (4, 5) else 0)
        + 8_000 * math.sin(2 * math.pi * timestamp.dayofyear / 365.25)
        for index, timestamp in zip(day_index, dates, strict=True)
    ]
    history = pd.DataFrame({"date": dates, "sales": sales})
    model_path = tmp_path / "demand_forecast.joblib"

    metadata = train_demand_model(history, model_path, holdout_days=60)

    assert model_path.is_file()
    assert metadata["validation_start"] == dates[-60].date().isoformat()
    assert metadata["train_end"] < metadata["validation_start"]
    assert metadata["metrics"]["model"]["wape_pct"] >= 0
    assert len(metadata["rolling_folds"]) == 3
    assert metadata["rolling_metrics"]["model"]["wape_pct"] >= 0
    forecast = forecast_with_demand_model(history, joblib.load(model_path), horizon=30)
    assert len(forecast) == 30
    assert forecast["date"].min() == dates[-1] + pd.Timedelta(days=1)


def test_churn_model_uses_future_purchase_window_and_scores_customers(tmp_path) -> None:
    rows = []
    start = pd.Timestamp("2020-01-01")
    invoice_number = 0
    for customer_number in range(24):
        customer_id = f"C{customer_number:03d}"
        last_day = 490 if customer_number < 12 else 210
        for day in range(customer_number % 7, last_day + 1, 20):
            invoice_number += 1
            rows.append(
                {
                    "InvoiceNo": f"I{invoice_number:05d}",
                    "InvoiceDate": start + pd.Timedelta(days=day),
                    "CustomerID": customer_id,
                    "Description": "Test product",
                    "Quantity": 2,
                    "UnitPrice": 10,
                }
            )
    data = normalize_retail_data(pd.DataFrame(rows))
    snapshots = build_churn_snapshots(data, horizon_days=90, step_days=30)
    model_path = tmp_path / "churn_model.joblib"

    metadata = train_churn_model(data, model_path, horizon_days=90)

    assert snapshots["churned"].nunique() == 2
    assert metadata["validation_start"] > metadata["train_end"]
    assert metadata["label_horizon_end"] <= metadata["validation_start"]
    assert metadata["target_definition"].startswith("No purchase during the 90 days")
    assert metadata["metrics"]["roc_auc"] >= 0
    assert model_path.is_file()
    scored = score_current_customers(data, joblib.load(model_path))
    assert len(scored) == 24
    assert scored["churn_probability"].between(0, 1).all()


def test_feature_drift_monitor_detects_shifted_values() -> None:
    training = pd.Series(range(100), dtype=float)
    reference = build_numeric_reference(training)
    stable_psi = population_stability_index(reference, training)
    shifted_psi = population_stability_index(reference, pd.Series([1000.0] * 100))

    assert stable_psi < 0.01
    assert shifted_psi >= 0.25

    constant_reference = build_numeric_reference(pd.Series([5.0] * 50))
    shifted_constant_psi = population_stability_index(constant_reference, pd.Series([50.0] * 50))
    assert shifted_constant_psi >= 0.25


def test_mlflow_tracking_is_optional_without_tracking_uri(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert log_mlflow_run(run_name="unit-test", parameters={}, metrics={}, metadata={}) is False