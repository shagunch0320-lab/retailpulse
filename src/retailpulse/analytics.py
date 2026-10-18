from __future__ import annotations

import pandas as pd


def build_sales_summary(data: pd.DataFrame) -> dict[str, int | float | None]:
    invoice_count = data["invoice"].nunique(dropna=True)
    orders = int(invoice_count if invoice_count else len(data))
    customer_count = int(data["customer_id"].nunique(dropna=True))
    product_count = int(data["description"].nunique(dropna=True))
    profit = float(data["profit"].sum()) if data["profit"].notna().any() else None
    return {
        "sales": float(data["sales"].sum()),
        "orders": orders,
        "customers": customer_count,
        "products": product_count,
        "profit": profit,
        "average_order_value": float(data["sales"].sum() / orders) if orders else 0.0,
    }


def get_sales_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.assign(date=data["invoice_date"].dt.floor("D"))
        .groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )


def build_customer_rfm(data: pd.DataFrame, churn_after_days: int = 90) -> pd.DataFrame:
    customers = data.dropna(subset=["customer_id"]).copy()
    if customers.empty:
        return pd.DataFrame(columns=["customer_id", "recency_days", "frequency", "monetary", "segment", "churn_risk"])

    snapshot = customers["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    grouped = customers.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("sales", "sum"),
    )
    grouped["frequency"] = grouped["frequency"].where(grouped["frequency"].gt(0), customers.groupby("customer_id").size())
    grouped["recency_days"] = (snapshot - grouped.pop("last_purchase").dt.normalize()).dt.days
    high_value_cut = grouped["monetary"].quantile(0.8)
    grouped["segment"] = "Regular"
    grouped.loc[grouped["monetary"].ge(high_value_cut), "segment"] = "High value"
    grouped.loc[grouped["frequency"].eq(1) & grouped["monetary"].lt(high_value_cut), "segment"] = "One-time"
    grouped["churn_risk"] = grouped["recency_days"].ge(churn_after_days).map({True: "At risk", False: "Active"})
    return grouped.reset_index()


def forecast_daily_sales(history: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Forecast from historical weekday means; requires at least two weeks of history."""
    if history["date"].nunique() < 14:
        raise ValueError("At least 14 distinct days of sales history are needed for the baseline forecast.")
    daily = history.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    weekday_means = daily.assign(weekday=daily["date"].dt.dayofweek).groupby("weekday")["sales"].mean()
    residual_scale = float((daily["sales"] - daily["date"].dt.dayofweek.map(weekday_means)).std())
    if pd.isna(residual_scale):
        residual_scale = 0.0
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    predicted = pd.Series(future_dates.dayofweek.map(weekday_means), index=future_dates)
    predicted = predicted.fillna(float(daily["sales"].mean())).clip(lower=0)
    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast": predicted.to_numpy(),
            "lower": (predicted - 1.96 * residual_scale).clip(lower=0).to_numpy(),
            "upper": (predicted + 1.96 * residual_scale).to_numpy(),
        }
    )


def build_inventory_view(data: pd.DataFrame, recent_days: int = 30) -> pd.DataFrame:
    end_date = data["invoice_date"].max().normalize()
    start_date = end_date - pd.Timedelta(days=recent_days - 1)
    recent = data.loc[data["invoice_date"].ge(start_date)].copy()
    quantity = recent.dropna(subset=["quantity"])
    if quantity.empty:
        return pd.DataFrame(columns=["description", "units_sold_30d", "projected_30d_units", "signal"])

    view = quantity.groupby("description", as_index=False).agg(units_sold_30d=("quantity", "sum"), sales_30d=("sales", "sum"))
    view["projected_30d_units"] = view["units_sold_30d"].clip(lower=0)
    threshold = view["units_sold_30d"].quantile(0.75)
    view["signal"] = view["units_sold_30d"].ge(threshold).map({True: "High demand", False: "Standard demand"})
    return view.sort_values("projected_30d_units", ascending=False).reset_index(drop=True)