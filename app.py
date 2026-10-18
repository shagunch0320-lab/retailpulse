from __future__ import annotations


import pandas as pd
import duckdb
import streamlit as st
from pathlib import Path as _Path

from retailpulse.analytics import (
    build_customer_rfm,
    build_inventory_view,
    build_sales_summary,
    forecast_daily_sales,
    get_sales_timeseries,
)
from retailpulse.data import load_retail_data
from retailpulse.bundle import load_retail_bundle
from retailpulse.config import get_settings
from retailpulse.demand_model import (
    daily_series_fingerprint,
    forecast_with_demand_model,
    load_demand_model,
    train_demand_model,
)
from retailpulse.churn_model import (
    churn_data_fingerprint,
    monitor_current_churn_features,
    score_current_customers,
    train_churn_model,
)
from retailpulse.etl import load_prepared_online_retail_ii
import joblib


st.set_page_config(
    page_title="RetailPulse | Retail intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #edf4ef; --muted: #a8b8ae; --line: #34443c; --surface: #1b2922; --surface-raised: #22332a; --accent: #72e0a2; }
    .stApp { background: #101713; color: var(--ink); }
    [data-testid="stAppViewContainer"] { background: #101713; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(16, 23, 19, 0.96); }
    [data-testid="stSidebar"] { background: #17221c; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: var(--ink); }
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); padding: 16px 18px; border-radius: 6px; }
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: var(--ink); }
    h1, h2, h3, p, label, span { color: var(--ink); letter-spacing: 0; }
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: var(--ink); }
    .eyebrow { color: var(--accent); text-transform: uppercase; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] { background: var(--surface); border-color: var(--line); color: var(--ink); }
    [data-testid="stDataFrame"], [data-testid="stTable"] { background: var(--surface); border: 1px solid var(--line); }
    [data-testid="stAlert"] { background: var(--surface); color: var(--ink); border-color: var(--line); }
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
if settings.require_login:
    if not settings.allowed_emails:
        st.error("Sign-in is enabled, but no authorized accounts are configured. Set RETAILPULSE_ALLOWED_EMAILS.")
        st.stop()
    if not st.user.is_logged_in:
        st.title("RetailPulse sign in")
        st.button("Sign in", on_click=st.login, type="primary")
        st.stop()
    if str(st.user.email).lower() not in settings.allowed_emails:
        st.error("This account is not authorized to access RetailPulse.")
        st.button("Sign out", on_click=st.logout)
        st.stop()


def money(value: float, currency: str = "₹") -> str:
    return f"{currency} {value:,.0f}"


def render_empty_state() -> None:
    st.info("Upload your cleaned or raw retail CSV/XLSX in the sidebar to begin. RetailPulse will detect common Online Retail column names automatically.")
    st.markdown("**Accepted fields:** invoice date, sales or quantity and unit price. Customer, product, country, and invoice identifiers add deeper analysis.")


def render_demand_forecast(history: pd.DataFrame, source_key: str) -> None:
    st.subheader("Daily sales forecast")
    artifact_path = (
        settings.demand_model_path
        if source_key == "relational_bundle"
        else settings.cache_dir / "demand_forecast_online_retail_ii.joblib"
    )
    artifact = load_demand_model(artifact_path)
    fingerprint = daily_series_fingerprint(history)
    if artifact is not None and artifact.get("data_fingerprint") != fingerprint:
        artifact = None
    train_clicked = st.button("Train and validate 90-day demand model", key=f"train_demand_{source_key}")
    if train_clicked:
        try:
            with st.spinner("Training the demand model and testing the final 90 days against a weekday baseline..."):
                train_demand_model(history, artifact_path, holdout_days=90)
            artifact = load_demand_model(artifact_path)
            st.rerun()
        except (ValueError, OSError) as error:
            st.warning(str(error))

    if artifact is not None:
        metrics = artifact["rolling_metrics"]
        model_wape = metrics["model"]["wape_pct"]
        baseline_wape = metrics["weekday_baseline"]["wape_pct"]
        if artifact["model_selected"]:
            st.success(f"Trained model selected · rolling holdout WAPE {model_wape:.2f}% vs weekday baseline {baseline_wape:.2f}%.")
        else:
            st.info(f"Trained model did not beat the weekday baseline by the required 5% margin (rolling WAPE {model_wape:.2f}% vs {baseline_wape:.2f}%); forecasts use the baseline.")
        validation_periods = ", ".join(
            f"{fold['validation_start']} to {fold['validation_end']}" for fold in artifact["rolling_folds"]
        )
        st.caption(f"Training starts {artifact['train_start']} · rolling validation folds: {validation_periods}")
    else:
        st.caption("No matching trained model is available. Train it below; it will only be selected if it beats the chronological weekday baseline.")

    try:
        forecast = (
            forecast_with_demand_model(history, artifact, horizon=30)
            if artifact is not None
            else forecast_daily_sales(history, horizon=30)
        )
    except ValueError as error:
        st.warning(str(error))
        return

    selected_label = "Trained model" if artifact is not None and artifact["model_selected"] else "Weekday baseline"
    history_and_forecast = pd.concat(
        [
            history.rename(columns={"sales": "forecast"}).assign(series="Actual"),
            forecast[["date", "forecast"]].assign(series=selected_label),
        ],
        ignore_index=True,
    )
    st.line_chart(
        history_and_forecast.pivot(index="date", columns="series", values="forecast"),
        color=["#167a58", "#d28147"],
    )
    st.dataframe(forecast, use_container_width=True, hide_index=True)


st.sidebar.markdown("<div class='eyebrow'>Retail intelligence</div>", unsafe_allow_html=True)
st.sidebar.title("RetailPulse")
st.sidebar.caption("Customer analytics · demand · inventory")
_SOURCE_OPTIONS = ["Online Retail II (primary)", "Upload one file"]
if (_Path(__file__).parent / "data" / "raw" / "sales_transactions.csv").exists():
    _SOURCE_OPTIONS.insert(1, "Relational RetailPulse bundle")

data_source = st.sidebar.radio(
    "Data source",
    _SOURCE_OPTIONS,
)
bundle = st.session_state.get("retail_bundle")
if data_source == "Online Retail II (primary)":
    workbook_path = settings.online_retail_ii_path
    st.sidebar.caption(f"Two-sheet transaction workbook: {workbook_path.name}")
    if st.sidebar.button("Load Online Retail II", type="primary", use_container_width=True):
        try:
            with st.spinner("Combining both Online Retail II year sheets and cleaning transaction rows..."):
                st.session_state["online_retail_ii_data"] = load_prepared_online_retail_ii(
                    workbook_path,
                    settings.cache_dir / "online_retail_ii.parquet",
                )
                st.session_state["online_retail_ii_filename"] = workbook_path.name
        except (ValueError, OSError, ImportError) as error:
            st.sidebar.error(str(error))
    uploaded = None
elif data_source == "Relational RetailPulse bundle":
    transaction_path = settings.transaction_path
    reference_path = settings.reference_dir
    st.sidebar.caption(f"Bundle: {transaction_path.name} · references: {reference_path.name}/")
    if st.sidebar.button("Load / refresh dataset", type="primary", use_container_width=True):
        try:
            with st.spinner("Scanning transactions and joining reference tables. First load may take a few minutes..."):
                st.session_state["retail_bundle"] = load_retail_bundle(
                    transaction_path,
                    reference_path,
                    settings.cache_dir,
                )
                st.session_state["retail_bundle_path"] = transaction_path.name
            bundle = st.session_state["retail_bundle"]
        except (ValueError, OSError, RuntimeError, duckdb.Error) as error:
            st.sidebar.error(str(error))
    uploaded = None
else:
    uploaded = st.sidebar.file_uploader("Retail transactions", type=["csv", "xlsx", "xls"])
page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Sales", "Customers", "Forecast", "Inventory"],
    label_visibility="collapsed",
)

if uploaded is not None:
    try:
        st.session_state["retail_data"] = load_retail_data(uploaded)
        st.session_state["retail_filename"] = uploaded.name
    except (ValueError, OSError, ImportError) as error:
        st.sidebar.error(str(error))

if data_source == "Online Retail II (primary)":
    df = st.session_state.get("online_retail_ii_data")
    standalone_filename = st.session_state.get("online_retail_ii_filename", settings.online_retail_ii_path.name)
elif data_source == "Upload one file":
    df = st.session_state.get("retail_data")
    standalone_filename = st.session_state.get("retail_filename", "Uploaded dataset")
else:
    df = None
    standalone_filename = ""
st.title(page)
st.caption("Retail performance, customer value, and demand signals in one place.")

if data_source == "Relational RetailPulse bundle" and bundle is not None:
    summary = bundle["summary"].iloc[0]
    st.sidebar.success(f"Bundle ready · {int(summary['transaction_lines']):,} transaction lines")
    st.sidebar.caption(st.session_state.get("retail_bundle_path", "Local dataset bundle"))

    if page == "Overview":
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Net sales", money(summary["sales"], "PKR"))
        col2.metric("Orders", f"{int(summary['orders']):,}")
        col3.metric("Customers", f"{int(summary['customers']):,}")
        col4.metric("SKUs", f"{int(summary['products']):,}")
        st.subheader("Sales over time")
        st.line_chart(bundle["daily_sales"].set_index("date")["sales"], color="#167a58")
        left, right = st.columns(2)
        with left:
            st.subheader("Top products")
            st.bar_chart(bundle["top_products"].set_index("description")["sales"].sort_values().tail(12), color="#d28147")
        with right:
            st.subheader("Sales by city")
            st.bar_chart(bundle["sales_by_city"].set_index("city")["sales"].sort_values().tail(12), color="#167a58")
        critical_count = int(bundle["inventory"]["recommendation"].isin(["Critical stock", "Reorder", "Flagged stockout risk"]).sum())
        st.metric("Store/SKU inventory lines requiring attention", f"{critical_count:,}")

    elif page == "Sales":
        col1, col2, col3 = st.columns(3)
        col1.metric("Net sales", money(summary["sales"], "PKR"))
        col2.metric("Estimated profit", money(summary["estimated_profit"], "PKR"))
        col3.metric("Estimated cost", money(summary["estimated_cost"], "PKR"))
        st.subheader("Daily sales")
        st.line_chart(bundle["daily_sales"].set_index("date")["sales"], color="#167a58")
        left, right = st.columns(2)
        with left:
            st.subheader("Sales by city")
            st.bar_chart(bundle["sales_by_city"].set_index("city")["sales"].sort_values(), color="#d28147")
        with right:
            st.subheader("Promotion performance")
            promotions = bundle["promotions"]
            if promotions.empty:
                st.info("No transactions matched promotion IDs.")
            else:
                st.dataframe(promotions, use_container_width=True, hide_index=True)
        st.subheader("Top SKU performance")
        st.dataframe(bundle["top_products"], use_container_width=True, hide_index=True)

    elif page == "Customers":
        customers = bundle["customers"]
        at_risk = int(customers["churn_risk"].eq("At risk").sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Customers analyzed", f"{len(customers):,}")
        col2.metric("At-risk customers", f"{at_risk:,}")
        col3.metric("Median recency", f"{customers['recency_days'].median():.0f} days")
        st.subheader("Loyalty segments")
        st.bar_chart(customers["loyalty_segment"].value_counts(), color="#167a58")
        st.subheader("RFM and customer profile")
        st.dataframe(customers.head(1000), use_container_width=True, hide_index=True)
        st.caption("The Pakistan bundle shows descriptive recency flags; it has a short 2022-2025 span and no current supervised churn artifact.")

    elif page == "Forecast":
        render_demand_forecast(bundle["daily_sales"], "relational_bundle")

    elif page == "Inventory":
        inventory = bundle["inventory"]
        critical = inventory["recommendation"].isin(["Critical stock", "Reorder", "Flagged stockout risk"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Store/SKU pairs", f"{len(inventory):,}")
        col2.metric("Need review", f"{int(critical.sum()):,}")
        col3.metric("Stockout-risk flags", f"{int(inventory['inventory_flag'].eq('STOCKOUT_RISK').sum()):,}")
        st.subheader("Inventory recommendations")
        city_options = ["All cities", *sorted(inventory["city"].dropna().unique().tolist())]
        recommendation_options = ["All signals", *sorted(inventory["recommendation"].dropna().unique().tolist())]
        city_filter, recommendation_filter = st.columns(2)
        selected_city = city_filter.selectbox("Store city", city_options)
        selected_recommendation = recommendation_filter.selectbox("Signal", recommendation_options)
        filtered_inventory = inventory
        if selected_city != "All cities":
            filtered_inventory = filtered_inventory.loc[filtered_inventory["city"].eq(selected_city)]
        if selected_recommendation != "All signals":
            filtered_inventory = filtered_inventory.loc[filtered_inventory["recommendation"].eq(selected_recommendation)]
        st.dataframe(filtered_inventory.head(1000), use_container_width=True, hide_index=True)
        st.caption(f"Showing up to 1,000 of {len(filtered_inventory):,} matching store/SKU rows.")
        st.caption("Recommendations compare the provided snapshot stock with reorder and safety-stock levels. The attached risk flags are simulated indicators.")
    st.stop()

elif df is None or df.empty:
    if data_source == "Relational RetailPulse bundle":
        st.info("The supplied RetailPulse transaction bundle is ready. Select **Load / refresh dataset** in the sidebar to scan transactions and join the customer, SKU, store, promotion, inventory, and risk tables.")
        st.caption("The transaction file is large; its first scan may take a few minutes. The resulting DuckDB database is cached under data/processed/.")
    elif data_source == "Online Retail II (primary)":
        st.info("Load Online Retail II from the sidebar. RetailPulse combines both official year sheets and filters returns/cancelled or non-positive sales rows for sales analytics.")
    else:
        render_empty_state()
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} transaction lines")
st.sidebar.caption(standalone_filename)

if page == "Overview":
    summary = build_sales_summary(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net sales", money(summary["sales"]))
    col2.metric("Orders", f"{summary['orders']:,}")
    col3.metric("Customers", f"{summary['customers']:,}")
    col4.metric("Products", f"{summary['products']:,}")

    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Sales over time")
        daily = get_sales_timeseries(df)
        st.line_chart(daily.set_index("date")["sales"], color="#167a58", height=310)
    with right:
        st.subheader("Top products")
        top_products = df.groupby("description", dropna=False)["sales"].sum().nlargest(8).sort_values()
        st.bar_chart(top_products, color="#d28147", height=310)
    st.subheader("Transaction sample")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

elif page == "Sales":
    summary = build_sales_summary(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Net sales", money(summary["sales"]))
    col2.metric("Estimated profit", money(summary["profit"]) if summary["profit"] is not None else "Not supplied")
    col3.metric("Average order value", money(summary["average_order_value"]))
    st.subheader("Daily sales")
    st.line_chart(get_sales_timeseries(df).set_index("date")["sales"], color="#167a58")
    if df["country"].notna().any():
        st.subheader("Sales by country")
        by_country = df.groupby("country")["sales"].sum().nlargest(15).sort_values()
        st.bar_chart(by_country, color="#d28147")

elif page == "Customers":
    if data_source == "Online Retail II (primary)":
        churn_path = settings.cache_dir / "churn_online_retail_ii.joblib"
        churn_artifact = joblib.load(churn_path) if churn_path.is_file() else None
        current_fingerprint = churn_data_fingerprint(df)
        if churn_artifact is not None and churn_artifact.get("data_fingerprint") != current_fingerprint:
            churn_artifact = None
        train_churn_clicked = st.button("Train and validate 90-day churn model")
        if train_churn_clicked:
            try:
                with st.spinner("Building monthly customer snapshots and evaluating on later dates..."):
                    train_churn_model(df, churn_path, horizon_days=90)
                st.rerun()
            except (ValueError, OSError) as error:
                st.warning(str(error))
        if churn_artifact is None:
            st.info("No matching trained churn model yet. The label is no purchase in the 90 days after each monthly customer snapshot.")
        else:
            metrics = churn_artifact["metrics"]
            auc_col, ap_col, brier_col, prevalence_col = st.columns(4)
            auc_col.metric("Held-out ROC AUC", f"{metrics['roc_auc']:.3f}")
            ap_col.metric("Average precision", f"{metrics['average_precision']:.3f}")
            brier_col.metric("Brier score", f"{metrics['brier']:.3f}")
            prevalence_col.metric("Churn-label rate", f"{metrics['prevalence']:.1%}")
            st.caption(
                f"Target: {churn_artifact['target_definition']} · "
                f"chronological train {churn_artifact['train_start']} to {churn_artifact['train_end']} · "
                f"test {churn_artifact['validation_start']} to {churn_artifact['validation_end']}"
            )
            scored = score_current_customers(df, churn_artifact)
            st.subheader("Customer churn risk scores")
            st.dataframe(scored.head(1000), use_container_width=True, hide_index=True)
            st.caption("Scores are predictive probabilities based on historical snapshots; validation is one chronological holdout.")
            drift = monitor_current_churn_features(df, churn_artifact)
            high_drift = drift.loc[drift["status"].eq("High drift"), "feature"].tolist()
            if high_drift:
                st.warning(f"High feature drift detected: {', '.join(high_drift)}. Retrain and revalidate before using these scores.")
            st.subheader("Churn input drift monitor")
            st.dataframe(drift, use_container_width=True, hide_index=True)
    else:
        rfm = build_customer_rfm(df)
        if rfm.empty:
            st.warning("Customer analysis needs a customer identifier column (for example, CustomerID).")
        else:
            risk_count = int(rfm["churn_risk"].eq("At risk").sum())
            col1, col2, col3 = st.columns(3)
            col1.metric("Customers analyzed", f"{len(rfm):,}")
            col2.metric("At-risk customers", f"{risk_count:,}")
            col3.metric("Median recency", f"{rfm['recency_days'].median():.0f} days")
            st.subheader("Customer segments")
            st.bar_chart(rfm["segment"].value_counts(), color="#167a58")
            st.subheader("RFM detail")
            st.dataframe(rfm.sort_values("monetary", ascending=False), use_container_width=True, hide_index=True)
            st.caption("At-risk status is a transparent recency heuristic, not a trained churn-probability model.")

elif page == "Forecast":
    source_key = "online_retail_ii" if data_source == "Online Retail II (primary)" else "standalone_dataset"
    render_demand_forecast(get_sales_timeseries(df), source_key)

elif page == "Inventory":
    inventory = build_inventory_view(df)
    st.subheader("Product demand signals")
    st.caption("Projected units use the trailing 30-day sales pace. On-hand stock is not in the uploaded transaction file, so this is not a reorder quantity.")
    st.dataframe(inventory, use_container_width=True, hide_index=True)
    if not inventory.empty:
        st.bar_chart(inventory.head(15).set_index("description")["projected_30d_units"], color="#d28147")

elif page == "Inventory":
    inventory = build_inventory_view(df)
    st.subheader("Product demand signals")
    st.caption("Projected units use the trailing 30-day sales pace. On-hand stock is not in the supplied schema, so this is not a reorder quantity.")
    st.dataframe(inventory, use_container_width=True, hide_index=True)
    if not inventory.empty:
        st.bar_chart(inventory.head(15).set_index("description")["projected_30d_units"], color="#d28147")