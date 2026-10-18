from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from retailpulse.config import get_settings
from retailpulse.data import load_online_retail_ii

ETL_SCHEMA_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_online_retail_ii(source_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """Combine, normalize, validate, and atomically write Online Retail II as Parquet."""
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    data = load_online_retail_ii(source)
    if data["invoice_date"].isna().any() or data["sales"].isna().any():
        raise ValueError("ETL validation failed: normalized dates and sales must be complete.")
    if (data["sales"] <= 0).any():
        raise ValueError("ETL validation failed: non-positive sales rows remain after cleaning.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".parquet", delete=False) as temporary:
            temp_name = temporary.name
        data.to_parquet(temp_name, index=False)
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    manifest: dict[str, object] = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(destination),
        "rows": len(data),
        "distinct_dates": int(data["invoice_date"].dt.normalize().nunique()),
        "start_date": data["invoice_date"].min().isoformat(),
        "end_date": data["invoice_date"].max().isoformat(),
        "sales_total": float(data["sales"].sum()),
        "invoice_count": int(data["invoice"].nunique()),
        "customer_count": int(data["customer_id"].nunique()),
        "product_count": int(data["description"].nunique()),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": ETL_SCHEMA_VERSION,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared_online_retail_ii(source_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Reuse normalized Parquet when its source hash matches; otherwise rerun ETL."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if source.is_file() and output.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") == ETL_SCHEMA_VERSION
                and manifest.get("source_sha256") == _sha256(source)
            ):
                return pd.read_parquet(output)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    prepare_online_retail_ii(source, output)
    return pd.read_parquet(output)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="ETL both Online Retail II sheets to validated Parquet.")
    parser.add_argument("--source", type=Path, default=settings.online_retail_ii_path)
    parser.add_argument("--output", type=Path, default=settings.cache_dir / "online_retail_ii.parquet")
    arguments = parser.parse_args()
    print(json.dumps(prepare_online_retail_ii(arguments.source, arguments.output), indent=2))


if __name__ == "__main__":
    main()