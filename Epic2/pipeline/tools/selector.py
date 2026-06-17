"""FeatureSelector — method choice delegated to JudgeAgent.plan().

Builds a FeatureSelectorEvidence packet using clusters and a linear baseline
recomputed on the *current* dataframe (including FeatureCreator-added columns),
hands it to the Judge sub-agent, then executes the returned
SelectionPlanResponse cluster by cluster.

Per-cluster action vocabulary:
  mrmr, pca, mrmr_then_pca, drop, lasso, rf_importance.

Judge unavailable or both attempts fail → fallback: mRMR over all features
with top_k_features from config.
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.evidence import ClusterEvidence, FeatureSelectorEvidence
from pipeline.parallel import compute_correlation_clusters, compute_linear_baseline
from pipeline.state import PipelineState


def _deterministic_tag(columns) -> str:
    """Deterministic 8-char tag from column names.

    Python's built-in `hash` is salted per process and is forbidden anywhere a
    name lands in `state.df` or `feature_artifact.json` (spec §5, §7-AA).
    """
    return hashlib.md5("|".join(columns).encode("utf-8")).hexdigest()[:8]


class FeatureSelector(BaseTool):
    def __init__(self, model_call: Callable[[str], str] | None = None, judge=None):
        # model_call kept for backward-compat with adk_tools; selector itself
        # never calls the model — Judge does.
        self.model_call = model_call
        self.judge = judge

    def precondition(self, state: PipelineState) -> None:
        if state.df is None:
            raise PreconditionError("FeatureSelector: state.df is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config
        feature_cols = [c for c in df.columns if c != state.target_column]
        if not feature_cols:
            state.selected_columns = []
            state.selection_method = "none"
            return

        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = state.target.to_numpy()

        clusters_map = compute_correlation_clusters(X, cut_threshold=cfg.feature_selection.cluster_cut_threshold)
        baseline = compute_linear_baseline(
            X, y, state.task, k=cfg.feature_selection.linear_baseline_k,
            seed=cfg.pipeline.random_state,
        )

        cluster_evidence: list[ClusterEvidence] = []
        per_col_mi = self._per_col_mi(X, y, state.task, state)
        for cid, members in clusters_map.items():
            mis = [per_col_mi.get(c, 0.0) for c in members]
            if len(members) > 1:
                sub = X[members]
                corr = sub.corr().abs().to_numpy()
                n = corr.shape[0]
                mask = ~np.eye(n, dtype=bool)
                intra = float(np.nanmean(corr[mask])) if mask.any() else 0.0
            else:
                intra = 0.0
            cluster_evidence.append(
                ClusterEvidence(
                    cluster_id=cid,
                    members=members,
                    mean_mi=float(np.mean(mis)) if mis else 0.0,
                    max_mi=float(np.max(mis)) if mis else 0.0,
                    intra_cluster_corr=intra,
                )
            )

        evidence = FeatureSelectorEvidence(
            n_rows=len(df),
            n_features=len(feature_cols),
            task=state.task,
            linear_baseline_score=baseline,
            clusters=cluster_evidence,
        )

        plan = None
        if self.judge is not None:
            try:
                plan = self.judge.plan(evidence=evidence, cfg=cfg)
            except Exception as e:
                state.warnings.append(f"FeatureSelector: judge.plan threw {e}")

        if plan is None:
            state.warnings.append("FeatureSelector: Judge unavailable; fallback mRMR over all features")
            state.last_llm_source = "fallback"
            selected = self._mrmr(X, y, state.task, min(cfg.feature_selection.top_k_features, len(feature_cols)), state)
            if not selected:
                selected = feature_cols[: cfg.feature_selection.top_k_features]
            state.selected_columns = selected
            state.selection_method = "fallback:mrmr_all"
            return
        state.last_llm_source = "ok"

        # Execute the plan cluster by cluster.
        plan_actions = {a.cluster_id: a.action for a in plan.plan}
        kept: list[str] = []
        per_cluster_summary: list[str] = []
        for cid, members in clusters_map.items():
            action = plan_actions.get(cid, "mrmr")
            # Per-cluster top-k: split overall budget proportionally.
            cluster_k = max(1, int(cfg.feature_selection.top_k_features * len(members) / max(1, len(feature_cols))))
            cluster_k = min(cluster_k, len(members))
            sub_X = X[members]
            try:
                if action == "mrmr":
                    cols = self._mrmr(sub_X, y, state.task, cluster_k, state)
                elif action == "pca":
                    cols = self._pca(sub_X, cluster_k, state, cfg)
                elif action == "mrmr_then_pca":
                    half = max(1, cluster_k // 2)
                    mrmr_cols = self._mrmr(sub_X, y, state.task, half, state)
                    residual = [c for c in members if c not in mrmr_cols]
                    if residual:
                        pca_cols = self._pca(sub_X[residual], cluster_k - len(mrmr_cols), state, cfg)
                        cols = mrmr_cols + pca_cols
                    else:
                        cols = mrmr_cols
                elif action == "drop":
                    cols = []
                elif action == "lasso":
                    cols = self._lasso(sub_X, y, state.task, cluster_k, cfg, state)
                elif action == "rf_importance":
                    cols = self._rf(sub_X, y, state.task, cluster_k, cfg, state)
                else:
                    cols = members[:cluster_k]
            except Exception as e:
                state.warnings.append(f"Selector cluster {cid} action={action} failed: {e}; fallback top-MI")
                cols = self._top_mi(sub_X, y, state.task, cluster_k, seed=cfg.pipeline.random_state)
            kept.extend(cols)
            per_cluster_summary.append(f"c{cid}:{action}")

        # De-dup and cap at top_k_features
        seen: set[str] = set()
        deduped: list[str] = []
        for c in kept:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        if not deduped:
            deduped = feature_cols[: cfg.feature_selection.top_k_features]
            state.selection_method = "fallback:first_k"
        else:
            state.selection_method = "plan:[" + ",".join(per_cluster_summary) + "]"
        state.selected_columns = deduped[: cfg.feature_selection.top_k_features]

    # ---------- helpers ----------

    def _per_col_mi(self, X: pd.DataFrame, y, task: str, state) -> dict[str, float]:
        seed = state.config.pipeline.random_state
        try:
            if task == "classification":
                mi = mutual_info_classif(X.to_numpy(), y, random_state=seed)
            else:
                mi = mutual_info_regression(X.to_numpy(), y, random_state=seed)
            return {c: float(m) for c, m in zip(X.columns, mi)}
        except Exception:
            return {c: 0.0 for c in X.columns}

    @staticmethod
    def _mrmr(X: pd.DataFrame, y, task: str, k: int, state) -> list[str]:
        if X.shape[1] == 0:
            return []
        try:
            from mrmr import mrmr_classif, mrmr_regression
            y_series = pd.Series(y)
            if task == "classification":
                return mrmr_classif(X=X, y=y_series, K=min(k, X.shape[1]))
            return mrmr_regression(X=X, y=y_series, K=min(k, X.shape[1]))
        except Exception as e:
            state.warnings.append(f"mrmr unavailable: {e}; using top-MI proxy")
            return FeatureSelector._top_mi(X, y, task, k, seed=state.config.pipeline.random_state)

    @staticmethod
    def _top_mi(X: pd.DataFrame, y, task: str, k: int, seed: int = 42) -> list[str]:
        if X.shape[1] == 0:
            return []
        try:
            if task == "classification":
                mi = mutual_info_classif(X.to_numpy(), y, random_state=seed)
            else:
                mi = mutual_info_regression(X.to_numpy(), y, random_state=seed)
            order = np.argsort(mi)[::-1][: min(k, X.shape[1])]
            return [X.columns[i] for i in order]
        except Exception:
            return X.columns.tolist()[:k]

    @staticmethod
    def _pca(X: pd.DataFrame, k: int, state, cfg) -> list[str]:
        if X.shape[1] == 0 or k <= 0:
            return []
        n_comp = min(k, X.shape[1], max(1, X.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=cfg.pipeline.random_state)
        transformed = pca.fit_transform(X.to_numpy())
        tag = _deterministic_tag(list(X.columns))
        new_names = [f"pca_{tag}_{i}" for i in range(n_comp)]
        for i, name in enumerate(new_names):
            state.df[name] = transformed[:, i]
            state.created_columns.append({"name": name, "operation": "pca", "sources": X.columns.tolist()})
            state.column_types[name] = "numeric"
        state.transformers.append({
            "step": "selection_pca",
            "n_components": n_comp,
            "sources": X.columns.tolist(),
            "explained_variance": [float(v) for v in pca.explained_variance_ratio_.tolist()],
        })
        return new_names

    @staticmethod
    def _lasso(X: pd.DataFrame, y, task: str, k: int, cfg, state) -> list[str]:
        from sklearn.linear_model import Lasso, LogisticRegression
        if X.shape[1] == 0:
            return []
        if task == "regression":
            model = Lasso(alpha=cfg.feature_selection.lasso_alpha, random_state=cfg.pipeline.random_state).fit(X.to_numpy(), y)
            importance = np.abs(model.coef_)
        else:
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0 / cfg.feature_selection.lasso_alpha,
                random_state=cfg.pipeline.random_state,
            ).fit(X.to_numpy(), y)
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        order = np.argsort(importance)[::-1][: min(k, X.shape[1])]
        return [X.columns[i] for i in order]

    @staticmethod
    def _rf(X: pd.DataFrame, y, task: str, k: int, cfg, state) -> list[str]:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if X.shape[1] == 0:
            return []
        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=cfg.feature_selection.rf_n_estimators, random_state=cfg.pipeline.random_state
            ).fit(X.to_numpy(), y)
        else:
            model = RandomForestRegressor(
                n_estimators=cfg.feature_selection.rf_n_estimators, random_state=cfg.pipeline.random_state
            ).fit(X.to_numpy(), y)
        importance = model.feature_importances_
        order = np.argsort(importance)[::-1][: min(k, X.shape[1])]
        return [X.columns[i] for i in order]

    def postcondition(self, state: PipelineState) -> None:
        if state.selected_columns is None:
            raise PostconditionError("FeatureSelector: selected_columns is None")
