"""Deterministic feature-selection statistics — computed ONCE, written to disk.

Phase B of the pipeline. Computes every statistic the feature-selection LLM needs
(mutual information, RandomForest importance, mRMR ranking, variance, Pearson +
Spearman correlation pairs, correlation clusters, linear baseline, PCA explained
variance) and writes them as JSON artifacts under `.mitra/<run_id>/stats/`.

The estimators are reused from backend.agents.feature_engineering.tools.selector and pipeline.parallel so the
math lives in one place (no duplicate compute with the profiler/selector).

Usable two ways:
  - importable: compute_and_write_stats(state, stats_dir) — called by the orchestrator.
  - CLI: python -m pipeline.feature_stats --data <csv> --target <col> --config <yaml>
    computes the same artifacts over an already-numeric/encoded CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.agents.feature_engineering.config import ConfigSchema, load_config
from backend.agents.feature_engineering.parallel import compute_correlation_clusters, compute_linear_baseline
from backend.agents.feature_engineering.state import PipelineState
from backend.agents.feature_engineering.tools.selector import FeatureSelector


class FeatureStatsComputer:
    """Computes the feature-selection stat artifacts for one (X, y) matrix."""

    def __init__(self, cfg: ConfigSchema):
        self.cfg = cfg
        self.seed = cfg.pipeline.random_state

    def compute(self, X: pd.DataFrame, y: np.ndarray, task: str) -> dict:
        """Return a dict of named stat artifacts. X must be all-numeric."""
        fs = self.cfg.feature_selection
        n_cols = X.shape[1]

        # Mutual information and RF importance (reuse selector estimators).
        mi_scores = FeatureSelector._mi_scores(X, y, task, seed=self.seed)
        rf_scores = FeatureSelector._rf_importances(X, y, task, self.cfg, seed=self.seed)
        mrmr_ranked = FeatureSelector._mrmr(X, y, task, k=n_cols, seed=self.seed)

        # Variance + low-variance flags (honors the previously-unused threshold).
        variances = {c: float(X[c].var()) for c in X.columns}
        low_variance = [c for c, v in variances.items() if v < fs.variance_threshold]

        # Correlation high-pairs above threshold (compact; full matrices are large).
        pearson_pairs = self._high_corr_pairs(X, method="pearson", threshold=fs.correlation_threshold)
        spearman_pairs = self._high_corr_pairs(X, method="spearman", threshold=fs.correlation_threshold)

        clusters = compute_correlation_clusters(X, cut_threshold=fs.cluster_cut_threshold)
        baseline = compute_linear_baseline(X, y, task, k=fs.linear_baseline_k, seed=self.seed)
        pca = self._pca_stats(X, fs.pca_variance_retained)

        return {
            "mutual_info": {"scores": mi_scores, "ranked": _ranked(mi_scores)},
            "rf_importance": {"scores": rf_scores, "ranked": _ranked(rf_scores)},
            "mrmr_ranking": {"ranked": list(mrmr_ranked)},
            "variance": {"scores": variances, "low_variance": low_variance, "threshold": fs.variance_threshold},
            "correlation_pearson": {"high_pairs": pearson_pairs, "threshold": fs.correlation_threshold},
            "correlation_spearman": {"high_pairs": spearman_pairs, "threshold": fs.correlation_threshold},
            "clusters": {str(cid): members for cid, members in clusters.items()},
            "linear_baseline": {"score": baseline, "task": task},
            "pca": pca,
        }

    def _high_corr_pairs(self, X: pd.DataFrame, method: str, threshold: float) -> list[list]:
        """Pairs of columns with |corr| >= threshold. Bounded by max_corr_pairs columns."""
        cap = self.cfg.feature_stats.max_corr_pairs
        cols = list(X.columns)[:cap]  # bound the O(cols^2) matrix on very wide data
        if len(cols) < 2:
            return []
        corr = X[cols].corr(method=method).abs()
        pairs: list[list] = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                c = corr.at[a, b]
                if pd.notna(c) and c >= threshold:
                    pairs.append([a, b, float(c)])
        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs

    def _pca_stats(self, X: pd.DataFrame, variance_retained: float) -> dict:
        from sklearn.decomposition import PCA

        if X.shape[1] == 0 or X.shape[0] < 2:
            return {}
        n_comp = min(X.shape[1], max(1, X.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=self.seed)
        pca.fit(X.to_numpy())
        ratios = [float(v) for v in pca.explained_variance_ratio_.tolist()]
        cumulative = np.cumsum(ratios)
        # Smallest #components reaching the configured retained variance.
        n_for_threshold = int(np.searchsorted(cumulative, variance_retained) + 1)
        n_for_threshold = min(n_for_threshold, n_comp)
        out = {
            "explained_variance_ratio": ratios,
            "variance_retained": variance_retained,
            "n_components_for_threshold": n_for_threshold,
        }
        if self.cfg.feature_stats.keep_pca_components:
            out["components"] = [[float(v) for v in row] for row in pca.components_.tolist()]
        return out

    def write(self, stats: dict, stats_dir: Path) -> None:
        stats_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in stats.items():
            (stats_dir / f"{name}.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )


def _ranked(scores: dict[str, float]) -> list[str]:
    return [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def compute_and_write_stats(state: PipelineState, stats_dir: Path) -> dict:
    """Compute the stat artifacts for the current pipeline state and persist them.

    Operates on the post-encode/post-scale feature matrix (target excluded).
    """
    df = state.df
    feature_cols = [c for c in df.columns if c != state.target_column]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = state.target.to_numpy()
    computer = FeatureStatsComputer(state.config)
    stats = computer.compute(X, y, state.task)
    computer.write(stats, stats_dir)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.feature_stats",
        description="Compute feature-selection statistics over an already-numeric CSV.",
    )
    parser.add_argument("--data", type=str, required=True, help="Path to a numeric/encoded CSV")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--task", type=str, required=False, default=None, choices=["classification", "regression"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--out", type=str, required=False, default=None,
                        help="Output stats dir. Defaults to <workspace_root>/feature_stats/stats")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"target column {args.target!r} not in dataset columns {list(df.columns)}")
    target = df[args.target]
    features = df.drop(columns=[args.target])
    X = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = target.to_numpy()

    # Resolve task if omitted (numeric target with many uniques => regression).
    if args.task is not None:
        task = args.task
    else:
        threshold = cfg.pipeline.task_infer_nunique_threshold
        task = "regression" if (pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > threshold) else "classification"

    out_dir = Path(args.out) if args.out else Path(cfg.paths.workspace_root) / "feature_stats" / "stats"
    computer = FeatureStatsComputer(cfg)
    stats = computer.compute(X, y, task)
    computer.write(stats, out_dir)
    print(f"task: {task}")
    print(f"wrote {len(stats)} stat artifacts to {out_dir}")
    for name in stats:
        print(f"  - {name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
