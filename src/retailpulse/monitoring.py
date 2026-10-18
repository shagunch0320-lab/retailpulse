from __future__ import annotations

import numpy as np
import pandas as pd


def build_numeric_reference(values: pd.Series, bins: int = 10) -> dict[str, list[float] | list[int]]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        raise ValueError("Cannot create a drift reference from empty feature data.")
    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(clean, quantiles)).tolist()
    if float(clean.min()) == float(clean.max()):
        center = float(clean[0])
        width = max(abs(center) * 1e-6, 0.5)
        edges = [center - width, center + width]
    counts = np.histogram(clean, bins=np.asarray([-np.inf, *edges, np.inf]))[0]
    expected = (counts / counts.sum()).tolist()
    return {"edges": edges, "expected": expected}


def population_stability_index(reference: dict[str, list[float] | list[int]], current: pd.Series) -> float:
    values = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    edges = np.asarray([-np.inf, *reference["edges"], np.inf], dtype=float)
    expected = np.asarray(reference["expected"], dtype=float)
    observed = np.histogram(values, bins=edges)[0].astype(float)
    observed /= observed.sum()
    epsilon = 1e-6
    expected = np.clip(expected, epsilon, None)
    observed = np.clip(observed, epsilon, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def feature_drift_report(
    references: dict[str, dict[str, list[float] | list[int]]],
    current: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for feature, reference in references.items():
        if feature not in current:
            continue
        psi = population_stability_index(reference, current[feature])
        if np.isnan(psi):
            status = "No data"
        elif psi >= 0.25:
            status = "High drift"
        elif psi >= 0.10:
            status = "Moderate drift"
        else:
            status = "Stable"
        rows.append({"feature": feature, "psi": psi, "status": status})
    return pd.DataFrame(rows)