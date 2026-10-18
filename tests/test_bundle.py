import pandas as pd
import pytest

from retailpulse.bundle import load_retail_bundle


def test_bundle_joins_reference_tables_and_uses_cache_dir(tmp_path) -> None:
    reference = tmp_path / "reference"
    cache = tmp_path / "persistent-cache"
    reference.mkdir()
    facts = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "receipt_id": ["R1", "R2"],
            "store_id": ["ST01", "ST01"],
            "sku_id": ["SKU1", "SKU1"],
            "customer_id": ["C1", "C1"],
            "quantity": [2, 1],
            "unit_price": [10, 10],
            "total_value": [20, 10],
            "channel": ["Online", "Online"],
            "discount_pct": [0, 5],
            "promo_id": ["", "P1"],
        }
    )
    facts.to_csv(tmp_path / "facts.csv", index=False)
    pd.DataFrame(
        {"sku_id": ["SKU1"], "sku_name": ["Sample SKU"], "category": ["Grocery"], "cost_price": [6]}
    ).to_csv(reference / "sku_master.csv", index=False)
    pd.DataFrame(
        {
            "cust_id": ["C1"], "age": [35], "gender": ["Female"], "city": ["Lahore"],
            "loyalty_segment": ["Gold"], "preferred_channel": ["Online"],
        }
    ).to_csv(reference / "customer_master.csv", index=False)
    pd.DataFrame({"store_id": ["ST01"], "city": ["Lahore"]}).to_csv(reference / "store_master.csv", index=False)
    pd.DataFrame(
        {"store_id": ["ST01"], "sku_id": ["SKU1"], "stock_on_hand": [5], "reorder_point": [6], "safety_stock": [2]}
    ).to_csv(reference / "inventory_snapshot.csv", index=False)
    pd.DataFrame({"promo_id": ["P1"], "promo_name": ["Test promo"], "promo_type": ["Discount"]}).to_csv(
        reference / "promotions.csv", index=False
    )
    pd.DataFrame(
        {"sku_id": ["SKU1"], "flag": ["STOCKOUT_RISK"], "affected_stores": ["ST01"]}
    ).to_csv(reference / "sku_inventory_flags.csv", index=False)

    bundle = load_retail_bundle(tmp_path / "facts.csv", reference, cache)

    assert bundle["summary"].loc[0, "sales"] == 30
    assert bundle["customers"].loc[0, "frequency"] == 2
    assert bundle["customers"].loc[0, "loyalty_segment"] == "Gold"
    assert bundle["inventory"].loc[0, "recommendation"] == "Reorder"
    assert bundle["promotions"].loc[0, "sales"] == 10
    assert (cache / "retailpulse_bundle.duckdb").is_file()

    invalid_facts = tmp_path / "invalid_facts.csv"
    pd.DataFrame({"not_a_transaction": ["bad"]}).to_csv(invalid_facts, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_retail_bundle(invalid_facts, reference, cache)

    recovered_bundle = load_retail_bundle(tmp_path / "facts.csv", reference, cache)
    assert recovered_bundle["summary"].loc[0, "sales"] == 30