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


def run_parallel(remote_fn, items: list) -> list:
    return ray.get([remote_fn.remote(*item) if isinstance(item, tuple) else remote_fn.remote(item) for item in items])


def compute_correlation_clusters(X: pd.DataFrame, cut_threshold: float) -> dict[int, list[str]]:
    """Average-linkage hierarchical clustering on 1 - |corr|.

    Returns {cluster_id: [columns]}. Singleton columns get their own cluster.
    """
    cols = list(X.columns)
    if len(cols) == 0:
        return {}
    if len(cols) == 1:
        return {0: cols}
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        corr = X.corr().abs().to_numpy()
        # Symmetric distance matrix
        dist = 1.0 - np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(dist, 0.0)
        # squareform requires zero diagonal and exact symmetry
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=cut_threshold, criterion="distance")
    except Exception:
        # Fallback: every column in its own cluster
        return {i: [c] for i, c in enumerate(cols)}

    clusters: dict[int, list[str]] = {}
    for col, lab in zip(cols, labels):
        clusters.setdefault(int(lab), []).append(col)
    # Re-key to 0-indexed cluster IDs
    return {i: members for i, members in enumerate(clusters.values())}


def compute_linear_baseline(X: pd.DataFrame, y: np.ndarray, task: str, k: int) -> float:
    """Fit LogisticRegression / LinearRegression on the top-k MI features.

    Returns CV-AUC (classification) or CV-R² (regression). Cheap proxy for
    whether linear methods will work.
    """
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import cross_val_score
    except Exception:
        return 0.0

    if X.shape[1] == 0 or len(y) < 10:
        return 0.0

    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    try:
        if task == "classification":
            mi = mutual_info_classif(X_num.to_numpy(), y, random_state=42)
        else:
            mi = mutual_info_regression(X_num.to_numpy(), y, random_state=42)
    except Exception:
        return 0.0

    top_idx = np.argsort(mi)[::-1][: min(k, X.shape[1])]
    X_top = X_num.iloc[:, top_idx].to_numpy()
    cv = min(5, max(2, len(y) // 10))
    try:
        if task == "classification":
            scoring = "roc_auc" if len(set(y)) == 2 else "accuracy"
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            scoring = "r2"
            model = LinearRegression()
        scores = cross_val_score(model, X_top, y, cv=cv, scoring=scoring)
        return float(np.mean(scores))
    except Exception:
        return 0.0


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
