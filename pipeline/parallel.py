from typing import Any

import numpy as np
import pandas as pd
import ray
from scipy import stats


@ray.remote
def _univariate_stats(col_name: str, series_bytes: dict) -> tuple[str, dict]:
    series = pd.Series(series_bytes["data"], name=col_name)
    dtype = str(series.dtype)
    null_rate = float(series.isna().mean())
    nunique = int(series.nunique(dropna=True))
    top_values = series.dropna().astype(str).value_counts().head(5).index.tolist()

    is_numeric = pd.api.types.is_numeric_dtype(series)
    stats_dict: dict[str, Any] = {
        "dtype": dtype,
        "null_rate": null_rate,
        "nunique": nunique,
        "top_values": top_values,
        "mean": None,
        "std": None,
        "skewness": None,
        "kurtosis": None,
        "outlier_rate": None,
    }
    if is_numeric:
        clean = series.dropna()
        if len(clean) > 0:
            stats_dict["mean"] = float(clean.mean())
            stats_dict["std"] = float(clean.std()) if len(clean) > 1 else 0.0
            stats_dict["skewness"] = float(stats.skew(clean)) if len(clean) > 2 else 0.0
            stats_dict["kurtosis"] = float(stats.kurtosis(clean)) if len(clean) > 3 else 0.0
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                stats_dict["outlier_rate"] = float(((clean < lo) | (clean > hi)).mean())
            else:
                stats_dict["outlier_rate"] = 0.0
    return col_name, stats_dict


@ray.remote
def _mi_with_target(col_data: list, target_data: list, task: str) -> float:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    col = np.asarray(col_data, dtype=float).reshape(-1, 1)
    tgt = np.asarray(target_data)
    mask = ~np.isnan(col.flatten())
    if mask.sum() < 5:
        return 0.0
    col = col[mask]
    tgt = tgt[mask]
    try:
        if task == "classification":
            return float(mutual_info_classif(col, tgt, random_state=42)[0])
        return float(mutual_info_regression(col, tgt, random_state=42)[0])
    except Exception:
        return 0.0


def run_parallel(remote_fn, items: list) -> list:
    return ray.get([remote_fn.remote(*item) if isinstance(item, tuple) else remote_fn.remote(item) for item in items])


def compute_mini_profile(df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Compute lightweight stats for newly-created columns. Synchronous (cheap, small N)."""
    profile: dict[str, dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        clean = pd.to_numeric(series, errors="coerce").dropna()
        entry: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_rate": float(series.isna().mean()),
            "nunique": int(series.nunique(dropna=True)),
            "top_values": [],
            "mean": None,
            "std": None,
            "skewness": None,
            "kurtosis": None,
            "outlier_rate": None,
        }
        if len(clean) > 0:
            entry["mean"] = float(clean.mean())
            entry["std"] = float(clean.std()) if len(clean) > 1 else 0.0
            entry["skewness"] = float(stats.skew(clean)) if len(clean) > 2 else 0.0
            entry["kurtosis"] = float(stats.kurtosis(clean)) if len(clean) > 3 else 0.0
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                entry["outlier_rate"] = float(((clean < lo) | (clean > hi)).mean())
            else:
                entry["outlier_rate"] = 0.0
        profile[col] = entry
    return profile
