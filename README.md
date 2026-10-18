# RetailPulse

RetailPulse is a Python and Streamlit foundation for retail sales analytics, customer RFM analysis, demand forecasting, and product demand signals. The app accepts transaction-level CSV and Excel files and maps common Online Retail / Online Retail II headers into one internal schema.

The primary analysis source is the official two-sheet Online Retail II workbook, staged as `data/raw/online_retail_II.xlsx`. It includes invoice-level transactions from both 2009-2010 and 2010-2011, with invoice date, product, quantity, price, customer, and country fields. The loader combines both worksheets and applies the current positive-sales cleaning policy. The dashboard defaults to **Online Retail II (primary)**. The separate Pakistan RetailPulse bundle remains available as **Relational RetailPulse bundle**; it uses `sales_transactions.csv` and six reference tables, with duplicate inventory-flag copies represented once. These datasets have different currencies, schemas, and date ranges and are intentionally analyzed separately. The cleaned RetailPulse CSV is a sample and is not staged or selected as project data.

## Run locally

From the project root, activate the existing virtual environment (or create one), install the project, and start Streamlit:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
```

The sidebar accepts `.csv`, `.xlsx`, and `.xls`. Required data is an invoice date plus either sales or both quantity and unit price. Customer ID, invoice, description, and country enable additional views. Currency-formatted numeric fields such as `₹ 51` are parsed. Data is not committed; put source files in `data/raw/` and derived exports in `data/processed/`.

To load Online Retail II, choose **Online Retail II (primary)** and select **Load Online Retail II**. Both official worksheets are combined and normalized. To load the secondary Pakistan dataset, choose **Relational RetailPulse bundle** and select **Load / refresh dataset**. The bundle is `sales_transactions.csv` plus six reference tables in `data/raw/`; its two supplied inventory-flag files are byte-identical copies, so they are represented once. The generic one-file uploader remains available for temporary inspection. Runtime paths are environment-configured. The Pakistan fact CSV is scanned with DuckDB and cached in `RETAILPULSE_CACHE_DIR`, rather than copied into the image or loaded all at once into pandas.

Normalize a source file from the command line with:

```bash
python -m retailpulse.pipeline data/raw/transactions.csv data/processed/transactions.csv
```

## Current capabilities

- Sales totals, order/customer/product counts, trends, country sales, and product rankings.
- RFM-style recency, invoice frequency, and monetary value summaries with transparent value segments.
- A recency-based churn-risk heuristic; it is not a trained churn classifier or probability.
- A daily sales HistGradientBoosting model with lag, rolling, trend, and calendar features, selected only when pooled three-fold, chronological 90-day recursive validation beats the weekday baseline by at least 5%. The baseline remains the fallback when training history is too short or the model loses validation.
- A supervised customer churn classifier for Online Retail II. Monthly customer snapshots are labeled from subsequent 90-day purchasing behavior, trained chronologically, and scored with ROC AUC, average precision, and Brier score. The Pakistan bundle still uses descriptive RFM recency flags because its span is too short for reliable churn snapshots.
- Product demand signals from the trailing 30 days. No stock-on-hand or lead-time inputs are assumed, so the view does not invent reorder quantities.

The uploaded sample appears to be a pre-aggregated dataset with duplicated date-part fields and derived risk/recommendation columns. RetailPulse will use direct sales/cost/profit columns where detected and derive dates from year/month/day or year-month. Raw transaction data is preferred for order frequency and valid forward-looking analysis; precomputed segment or risk labels are not treated as model output.

## Validate

```bash
pytest
```

CI runs the tests and builds the Docker image on pushes and pull requests.

Train or retrain the daily demand model after loading the relational bundle:

```bash
python -m retailpulse.demand_model \
    --database data/processed/retailpulse_bundle.duckdb \
    --model-path data/processed/demand_forecast.joblib \
    --holdout-days 90
```

Train or retrain the Online Retail II churn model from the configured workbook:

```bash
retailpulse-train-churn --horizon-days 90
```

The Forecast page provides a train/validate button. It compares pooled WAPE across three disjoint chronological 90-day recursive folds and selects a trained estimator only for at least a 5% relative gain over the weekday baseline. The refreshed Pakistan bundle artifact scores 5.64% pooled WAPE versus 17.99% baseline and is selected. Online Retail II scores 39.16% versus 39.55%, so the dashboard correctly keeps its baseline active. Treat the latter as exploratory, and continue rolling-origin monitoring after material data changes.

For Online Retail II, the churn classifier held out later snapshot dates from 2011-05-25 through 2011-08-23 and used a 90-day embargo so all training labels were observable before validation. It scored ROC AUC 0.802, average precision 0.863, and Brier score 0.186. Its target is explicitly “no purchase in the 90 days after the monthly observation date”; this is a behavioral proxy and depends on the dataset's observation window, not a ground-truth business churn label.

Run the repeatable primary-source ETL with:

```bash
retailpulse-etl --source data/raw/online_retail_II.xlsx --output data/processed/online_retail_ii.parquet
```

It validates the normalized rows, writes Parquet atomically, and records a SHA-256 source manifest. The dashboard invokes the same ETL automatically and reuses the Parquet output while the source hash is unchanged. Optional MLflow tracking can be enabled with `pip install -e '.[mlops]'` and `MLFLOW_TRACKING_URI` (optionally `MLFLOW_EXPERIMENT_NAME`); training works without MLflow. Churn input monitoring compares current feature distributions with training snapshots using PSI and flags values at or above 0.25 as high drift.

## Production Deployment

The included Compose deployment keeps the source workbook, transaction file, and reference tables outside the image, mounts them read-only, stores DuckDB in a persistent named volume, runs as a non-root user with a read-only root filesystem, drops Linux capabilities, and binds to localhost by default. Put the Online Retail II workbook and Pakistan bundle files in `data/raw/`, and create an OIDC client with the callback URL `https://YOUR_PUBLIC_HOST/oauth2callback`. Create the ignored local secrets file and fill its placeholders with the provider's real values:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Never commit the real secrets file. Configure the OIDC provider to return verified email addresses and only include authorized addresses in `RETAILPULSE_ALLOWED_EMAILS`.

For the staged local datasets, start Compose from the repository root with:

```bash
export RETAILPULSE_ALLOWED_EMAILS="operator@example.com,analyst@example.com"
docker compose up --build -d
docker compose ps
docker compose logs -f retailpulse
```

Then open `http://localhost:8501`, sign in with an allowlisted account, and select **Online Retail II (primary)** to load both workbook sheets. Compose mounts `data/raw/` read-only. The database cache is stored in the `retailpulse-cache` volume and survives container replacement. Back up that volume if startup scan time matters; it can be rebuilt from the source files if lost. Do not expose port 8501 directly to the public internet: terminate TLS at a trusted reverse proxy or managed ingress, configure its trusted forwarded headers, and restrict network access. For remote deployment set `RETAILPULSE_BIND_ADDRESS=0.0.0.0` only behind that ingress.

The Compose memory limit is 4 GiB and CPU limit is 2 cores as a starting point; size them against real concurrency and refresh workloads. DuckDB uses a single local persistent database, so run one application replica per cache volume. For multi-replica or multi-user production, move analytical storage to a shared service and use an external identity/access policy. Configure provider-managed backups, log retention, vulnerability scanning, and alerting before handling sensitive customer data. The provided sample data is synthetic/demo data; validate privacy, retention, currency, forecasting backtests, and business thresholds before operational decisions.

## Project map

```text
app.py                    Streamlit application
src/retailpulse/data.py    Input reading, schema normalization, cleaning
src/retailpulse/analytics.py
                          Sales, RFM, baseline forecast, inventory signals
src/retailpulse/demand_model.py
                          Chronological training, validation, and forecast artifact
src/retailpulse/bundle.py  DuckDB relational bundle loading and aggregates
src/retailpulse/config.py  Environment-based runtime configuration
tests/                    Focused data and analytics tests
data/raw/                 Source datasets (ignored by Git)
data/processed/           Derived datasets (ignored by Git)
notebooks/                Exploration and model development
```

## Remaining External Work

The codebase now includes repeatable ETL, chronological forecast validation, a time-defined churn classifier, optional MLflow tracking, PSI feature monitoring, Docker, Compose, and CI image builds. Public deployment still requires your hosting account/domain, OIDC provider credentials, TLS ingress, persistent-volume backups, and operational alert routing. Inventory recommendations need real lead-time/service-level inputs before they can be treated as reorder quantities; the provided risk flags are simulated.