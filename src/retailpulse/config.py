from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    transaction_path: Path
    reference_dir: Path
    cache_dir: Path
    demand_model_path: Path
    online_retail_ii_path: Path
    require_login: bool
    allowed_emails: frozenset[str]


def get_settings() -> Settings:
    transaction_path = Path(
        os.getenv("RETAILPULSE_TRANSACTION_PATH", str(DEFAULT_RAW_DIR / "sales_transactions.csv"))
    ).expanduser()
    reference_dir = Path(os.getenv("RETAILPULSE_REFERENCE_DIR", str(DEFAULT_RAW_DIR))).expanduser()
    cache_dir = Path(os.getenv("RETAILPULSE_CACHE_DIR", str(PROJECT_ROOT / "data" / "processed"))).expanduser()
    demand_model_path = Path(
        os.getenv("RETAILPULSE_DEMAND_MODEL_PATH", str(cache_dir / "demand_forecast.joblib"))
    ).expanduser()
    online_retail_ii_path = Path(
        os.getenv("RETAILPULSE_ONLINE_RETAIL_II_PATH", str(DEFAULT_RAW_DIR / "online_retail_II.xlsx"))
    ).expanduser()
    allowed_emails = frozenset(
        email.strip().lower()
        for email in os.getenv("RETAILPULSE_ALLOWED_EMAILS", "").split(",")
        if email.strip()
    )
    return Settings(
        transaction_path=transaction_path,
        reference_dir=reference_dir,
        cache_dir=cache_dir,
        demand_model_path=demand_model_path,
        online_retail_ii_path=online_retail_ii_path,
        require_login=_as_bool(os.getenv("RETAILPULSE_REQUIRE_LOGIN")),
        allowed_emails=allowed_emails,
    )