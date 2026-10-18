from __future__ import annotations

from pathlib import Path
import re
from typing import IO

import pandas as pd


ALIASES = {
    "invoice": ("invoice", "invoiceno", "invoicenumber", "suminvoicenumber", "transactionid", "orderno", "orderid"),
    "invoice_date": ("invoicedate", "transactiondate", "orderdate", "date"),
    "customer_id": ("customerid", "customer", "customeridentifier"),
    "sku_id": ("skuid", "stockcode", "stockcodeid"),
    "description": ("description", "productdescription", "productname", "stockdescription", "product"),
    "quantity": ("quantity", "sumofquantity", "units", "unitssold"),
    "unit_price": ("unitprice", "sumofunitprice", "price", "sellingprice"),
    "sales": ("sales", "totalsales", "sumoftotalsales", "totalprice", "revenue", "customerrevenue"),
    "cost": ("estimatedcost", "sumofestimatedcost", "cost", "totalcost"),
    "profit": ("estimatedprofit", "sumofestimatedprofit", "profit", "grossprofit"),
    "country": ("country", "region", "market"),
    "year_month": ("yearmonth", "monthkey", "period"),
    "year": ("year",),
    "month": ("month", "monthname", "monthno", "sumofmonthno"),
    "day": ("day", "dayofmonth"),
}


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _number(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _derive_dates(frame: pd.DataFrame) -> pd.Series:
    if "year" in frame and "month" in frame:
        year = pd.to_numeric(frame["year"], errors="coerce")
        month_names = frame["month"].astype("string").str.strip()
        month_numbers = pd.to_numeric(month_names, errors="coerce")
        month_numbers = month_numbers.fillna(pd.to_datetime(month_names, format="%B", errors="coerce").dt.month)
        month_numbers = month_numbers.fillna(pd.to_datetime(month_names, format="%b", errors="coerce").dt.month)
        day = pd.to_numeric(frame.get("day", pd.Series(1, index=frame.index)), errors="coerce").fillna(1)
        return pd.to_datetime({"year": year, "month": month_numbers, "day": day}, errors="coerce")
    if "year_month" in frame:
        dates = pd.to_datetime(frame["year_month"], errors="coerce")
        if dates.notna().any():
            return dates
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def normalize_retail_data(source: pd.DataFrame) -> pd.DataFrame:
    """Normalize common retail transaction schemas into the RetailPulse contract."""
    if source.empty:
        raise ValueError("The uploaded dataset has no rows.")

    frame = source.copy()
    original_columns = list(frame.columns)
    normalized_to_original: dict[str, object] = {}
    for column in original_columns:
        normalized_to_original.setdefault(_key(column), column)

    rename: dict[object, str] = {}
    for target, names in ALIASES.items():
        for name in names:
            if name in normalized_to_original:
                source_name = normalized_to_original[name]
                if source_name not in rename:
                    rename[source_name] = target
                break
    frame = frame.rename(columns=rename)

    if "invoice_date" in frame:
        dates = pd.to_datetime(frame["invoice_date"], errors="coerce")
    else:
        dates = _derive_dates(frame)
    frame["invoice_date"] = dates

    for column in ("quantity", "unit_price", "sales", "cost", "profit"):
        if column in frame:
            frame[column] = _number(frame[column])

    if "sales" not in frame:
        if not {"quantity", "unit_price"}.issubset(frame.columns):
            raise ValueError("Could not identify sales, or both quantity and unit price. Check the dataset headers.")
        frame["sales"] = frame["quantity"] * frame["unit_price"]
    if "quantity" not in frame:
        frame["quantity"] = pd.NA
    if "unit_price" not in frame:
        frame["unit_price"] = pd.NA
    if "cost" not in frame:
        frame["cost"] = pd.NA
    if "profit" not in frame:
        frame["profit"] = pd.NA
    for column in ("invoice", "customer_id", "sku_id", "description", "country"):
        if column not in frame:
            frame[column] = pd.NA

    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.dropna(subset=["invoice_date", "sales"])
    if "quantity" in frame and frame["quantity"].notna().any():
        frame = frame.loc[frame["quantity"].isna() | frame["quantity"].gt(0)]
    if "unit_price" in frame and frame["unit_price"].notna().any():
        frame = frame.loc[frame["unit_price"].isna() | frame["unit_price"].gt(0)]
    frame = frame.loc[frame["sales"].gt(0)].copy()
    frame["description"] = frame["description"].fillna("Unspecified product").astype(str)
    frame["country"] = frame["country"].astype("string")
    frame["customer_id"] = frame["customer_id"].astype("string")
    frame["sku_id"] = frame["sku_id"].astype("string")
    frame["invoice"] = frame["invoice"].astype("string")

    if frame.empty:
        raise ValueError("No usable sales rows remain after parsing dates and removing invalid or non-positive sales.")
    return frame.reset_index(drop=True)


def load_retail_data(file: IO[bytes] | Path | str) -> pd.DataFrame:
    """Read a CSV or Excel file and return normalized transaction rows."""
    filename = getattr(file, "name", str(file)).lower()
    try:
        if filename.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(file)
        else:
            raw = pd.read_csv(file)
    except ImportError as error:
        raise ImportError("Excel support requires openpyxl. Install project dependencies with `pip install -e '.[dev]'`.") from error
    except (ValueError, OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read the uploaded dataset: {error}") from error
    return normalize_retail_data(raw)


def load_online_retail_ii(path: str | Path) -> pd.DataFrame:
    """Load and combine both official Online Retail II year worksheets."""
    workbook_path = Path(path).expanduser()
    expected_sheets = ("Year 2009-2010", "Year 2010-2011")
    try:
        workbook = pd.ExcelFile(workbook_path)
        missing_sheets = set(expected_sheets) - set(workbook.sheet_names)
        if missing_sheets:
            raise ValueError(
                f"Online Retail II workbook is missing expected sheets: {', '.join(sorted(missing_sheets))}"
            )
        frames = [pd.read_excel(workbook, sheet_name=sheet) for sheet in expected_sheets]
    except ImportError as error:
        raise ImportError("Excel support requires openpyxl. Install project dependencies with `pip install -e '.[dev]'`.") from error
    except (ValueError, OSError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Online Retail II workbook"):
            raise
        raise ValueError(f"Could not read the Online Retail II workbook: {error}") from error
    return normalize_retail_data(pd.concat(frames, ignore_index=True))