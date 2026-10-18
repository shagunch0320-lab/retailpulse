from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


REFERENCE_FILES = {
    "sku_master": "sku_master.csv",
    "customer_master": "customer_master.csv",
    "store_master": "store_master.csv",
    "inventory_snapshot": "inventory_snapshot.csv",
    "promotions": "promotions.csv",
    "sku_inventory_flags": "sku_inventory_flags.csv",
}


def _replace_csv_table(connection: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    connection.execute(
        f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
        [str(path)],
    )


def load_retail_bundle(
    transaction_path: str | Path,
    raw_dir: str | Path,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a large transaction CSV with DuckDB and return joined dashboard aggregates."""
    source = Path(transaction_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Transaction file not found: {source}")

    raw_path = Path(raw_dir).resolve()
    processed_path = Path(cache_dir).expanduser().resolve() if cache_dir else raw_path.parent / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    database_path = processed_path / "retailpulse_bundle.duckdb"
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS source_manifest (path VARCHAR, size BIGINT, modified_ns BIGINT)")
        stat = source.stat()
        previous = connection.execute("SELECT path, size, modified_ns FROM source_manifest LIMIT 1").fetchone()
        fingerprint = (str(source), stat.st_size, stat.st_mtime_ns)
        if previous != fingerprint:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute("DROP TABLE IF EXISTS transactions")
                connection.execute(
                    "CREATE TABLE transactions AS SELECT * FROM read_csv_auto(?, header=true, sample_size=100000)",
                    [str(source)],
                )
                transaction_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info('transactions')").fetchall()
                }
                required_columns = {
                    "date", "receipt_id", "store_id", "sku_id", "customer_id", "quantity", "total_value"
                }
                missing_columns = required_columns - transaction_columns
                if missing_columns:
                    raise ValueError(
                        f"Transaction CSV is missing required columns: {', '.join(sorted(missing_columns))}"
                    )
                connection.execute("DELETE FROM source_manifest")
                connection.execute("INSERT INTO source_manifest VALUES (?, ?, ?)", list(fingerprint))
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

        for table, filename in REFERENCE_FILES.items():
            path = raw_path / filename
            if path.is_file():
                _replace_csv_table(connection, table, path)

        existing_tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        required_tables = {"sku_master", "customer_master", "store_master", "inventory_snapshot"}
        missing_tables = required_tables - existing_tables
        if missing_tables:
            raise ValueError(f"Missing required reference files in {raw_path}: {', '.join(sorted(missing_tables))}")

        summary = connection.execute(
            """
            SELECT count(*) AS transaction_lines,
                   sum(try_cast(t.total_value AS DOUBLE)) AS sales,
                   count(DISTINCT t.receipt_id) AS orders,
                   count(DISTINCT t.customer_id) AS customers,
                   count(DISTINCT t.sku_id) AS products,
                   sum(try_cast(t.quantity AS DOUBLE) * try_cast(s.cost_price AS DOUBLE)) AS estimated_cost,
                   sum(try_cast(t.total_value AS DOUBLE) -
                       try_cast(t.quantity AS DOUBLE) * try_cast(s.cost_price AS DOUBLE)) AS estimated_profit
            FROM transactions t
            LEFT JOIN sku_master s USING (sku_id)
            """
        ).df()
        daily_sales = connection.execute(
            "SELECT CAST(date AS DATE) AS date, sum(try_cast(total_value AS DOUBLE)) AS sales "
            "FROM transactions GROUP BY 1 ORDER BY 1"
        ).df()
        top_products = connection.execute(
            """
            SELECT t.sku_id, s.sku_name AS description, s.category,
                   sum(try_cast(t.quantity AS DOUBLE)) AS units,
                   sum(try_cast(t.total_value AS DOUBLE)) AS sales
            FROM transactions t LEFT JOIN sku_master s USING (sku_id)
            GROUP BY 1, 2, 3 ORDER BY sales DESC LIMIT 20
            """
        ).df()
        sales_by_city = connection.execute(
            """
            SELECT st.city, sum(try_cast(t.total_value AS DOUBLE)) AS sales
            FROM transactions t LEFT JOIN store_master st USING (store_id)
            GROUP BY 1 ORDER BY sales DESC
            """
        ).df()
        customers = connection.execute(
            """
            WITH snapshot AS (SELECT max(CAST(date AS DATE)) + INTERVAL 1 DAY AS date FROM transactions),
            rfm AS (
                SELECT customer_id, max(CAST(date AS DATE)) AS last_purchase,
                       count(DISTINCT receipt_id) AS frequency,
                       sum(try_cast(total_value AS DOUBLE)) AS monetary
                FROM transactions GROUP BY customer_id
            )
            SELECT r.customer_id, c.age, c.gender, c.city, c.loyalty_segment,
                   c.preferred_channel, date_diff('day', r.last_purchase, snapshot.date) AS recency_days,
                   r.frequency, r.monetary,
                   CASE WHEN date_diff('day', r.last_purchase, snapshot.date) >= 90
                        THEN 'At risk' ELSE 'Active' END AS churn_risk
            FROM rfm r CROSS JOIN snapshot
            LEFT JOIN customer_master c ON c.cust_id = r.customer_id
            ORDER BY r.monetary DESC
            """
        ).df()
        inventory = connection.execute(
            """
            WITH latest AS (SELECT max(CAST(date AS DATE)) AS date FROM transactions),
            demand AS (
                SELECT t.store_id, t.sku_id,
                       sum(try_cast(t.quantity AS DOUBLE)) AS units_sold_30d,
                       sum(try_cast(t.total_value AS DOUBLE)) AS sales_30d
                FROM transactions t CROSS JOIN latest
                WHERE CAST(t.date AS DATE) > latest.date - INTERVAL 30 DAY
                GROUP BY 1, 2
            ),
            expanded_flags AS (
                SELECT sku_id, unnest(string_split(affected_stores, ';')) AS store_id, flag
                FROM sku_inventory_flags
            )
            SELECT d.store_id, st.city, d.sku_id, s.sku_name, s.category,
                   d.units_sold_30d, d.sales_30d, i.stock_on_hand, i.reorder_point,
                   i.safety_stock, coalesce(f.flag, '') AS inventory_flag,
                   CASE WHEN i.stock_on_hand <= i.safety_stock THEN 'Critical stock'
                        WHEN i.stock_on_hand <= i.reorder_point THEN 'Reorder'
                        WHEN f.flag = 'STOCKOUT_RISK' THEN 'Flagged stockout risk'
                        WHEN f.flag = 'SLOW_MOVER' THEN 'Slow mover'
                        ELSE 'Monitor' END AS recommendation
            FROM demand d
            LEFT JOIN inventory_snapshot i USING (store_id, sku_id)
            LEFT JOIN sku_master s USING (sku_id)
            LEFT JOIN store_master st USING (store_id)
            LEFT JOIN expanded_flags f USING (store_id, sku_id)
            ORDER BY CASE WHEN i.stock_on_hand <= i.safety_stock THEN 0
                          WHEN i.stock_on_hand <= i.reorder_point THEN 1 ELSE 2 END,
                     d.units_sold_30d DESC
            """
        ).df()
        if "promotions" in existing_tables:
            promotions = connection.execute(
                """
                SELECT t.promo_id, p.promo_name, p.promo_type,
                       sum(try_cast(t.total_value AS DOUBLE)) AS sales,
                       sum(try_cast(t.quantity AS DOUBLE)) AS units,
                       count(DISTINCT t.receipt_id) AS orders,
                       avg(try_cast(t.discount_pct AS DOUBLE)) AS avg_discount_pct
                FROM transactions t JOIN promotions p USING (promo_id)
                WHERE coalesce(t.promo_id, '') <> ''
                GROUP BY 1, 2, 3 ORDER BY sales DESC
                """
            ).df()
        else:
            promotions = pd.DataFrame()

        return {
            "summary": summary,
            "daily_sales": daily_sales,
            "top_products": top_products,
            "sales_by_city": sales_by_city,
            "customers": customers,
            "inventory": inventory,
            "promotions": promotions,
        }
    finally:
        connection.close()