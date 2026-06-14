import json
import re
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.state import PipelineState

SELECTOR_PROMPT = """You pick a feature selection method. Choose ONE of:
mrmr, pca, mrmr_pca, lasso, rf_importance.

Guidance:
- High inter-correlation + low MI -> pca.
- Low inter-correlation + high MI -> mrmr.
- Mixed -> mrmr_pca (mRMR on significant features, PCA on residual).
- Small linear dataset -> lasso.
- Non-linear patterns -> rf_importance.
- Default fallback -> mrmr.

Context:
- Task: {task}
- Feature count: {n_features}
- Row count: {n_rows}
- Mean abs correlation: {mean_corr:.3f}
- Mean MI with target: {mean_mi:.3f}

Respond with ONLY this JSON, no prose:
{{"method": "<method>"}}
"""


def _extract_json_obj(text: str) -> dict:
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object: {text[:200]}")
    return json.loads(m.group(0))


class FeatureSelector(BaseTool):
    def __init__(self, model_call: Callable[[str], str]):
        self.model_call = model_call

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

        mean_corr = self._mean_abs_corr(X)
        mean_mi = self._mean_mi(X, y, state.task)

        prompt = SELECTOR_PROMPT.format(
            task=state.task,
            n_features=len(feature_cols),
            n_rows=len(df),
            mean_corr=mean_corr,
            mean_mi=mean_mi,
        )
        method = "mrmr"
        try:
            response = self.model_call(prompt)
            obj = _extract_json_obj(response)
            candidate = obj.get("method", "mrmr").lower()
            if candidate in {"mrmr", "pca", "mrmr_pca", "lasso", "rf_importance"}:
                method = candidate
        except Exception as e:
            state.warnings.append(f"FeatureSelector parse failed: {e}; using mrmr fallback")

        k = min(cfg.feature_selection.top_k_features, len(feature_cols))
        selected = self._apply(method, X, y, state.task, k, cfg, state)
        if not selected:
            selected = feature_cols[:k]
            method = "fallback_first_k"

        state.selected_columns = selected
        state.selection_method = method

    @staticmethod
    def _mean_abs_corr(X: pd.DataFrame) -> float:
        if X.shape[1] < 2:
            return 0.0
        corr = X.corr().abs().to_numpy()
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(np.nanmean(corr[mask])) if mask.any() else 0.0

    @staticmethod
    def _mean_mi(X: pd.DataFrame, y, task: str) -> float:
        try:
            if task == "classification":
                mi = mutual_info_classif(X.to_numpy(), y, random_state=42)
            else:
                mi = mutual_info_regression(X.to_numpy(), y, random_state=42)
            return float(np.mean(mi))
        except Exception:
            return 0.0

    def _apply(self, method: str, X: pd.DataFrame, y, task: str, k: int, cfg, state) -> list[str]:
        cols = X.columns.tolist()
        try:
            if method == "mrmr":
                return self._mrmr(X, y, task, k, state)
            if method == "pca":
                return self._pca(X, k, state, cfg)
            if method == "mrmr_pca":
                half = max(1, k // 2)
                mrmr_cols = self._mrmr(X, y, task, half, state)
                residual_cols = [c for c in cols if c not in mrmr_cols]
                if residual_cols:
                    pca_cols = self._pca(X[residual_cols], k - len(mrmr_cols), state, cfg)
                    return mrmr_cols + pca_cols
                return mrmr_cols
            if method == "lasso":
                return self._lasso(X, y, task, k, cfg, state)
            if method == "rf_importance":
                return self._rf(X, y, task, k, cfg, state)
        except Exception as e:
            state.warnings.append(f"Selector {method} failed: {e}; falling back to top-MI")
        return self._top_mi(X, y, task, k)

    @staticmethod
    def _mrmr(X: pd.DataFrame, y, task: str, k: int, state) -> list[str]:
        try:
            from mrmr import mrmr_classif, mrmr_regression
            y_series = pd.Series(y)
            if task == "classification":
                return mrmr_classif(X=X, y=y_series, K=k)
            return mrmr_regression(X=X, y=y_series, K=k)
        except Exception as e:
            state.warnings.append(f"mrmr unavailable: {e}; using top-MI proxy")
            return FeatureSelector._top_mi(X, y, task, k)

    @staticmethod
    def _top_mi(X: pd.DataFrame, y, task: str, k: int) -> list[str]:
        try:
            if task == "classification":
                mi = mutual_info_classif(X.to_numpy(), y, random_state=42)
            else:
                mi = mutual_info_regression(X.to_numpy(), y, random_state=42)
            order = np.argsort(mi)[::-1][:k]
            return [X.columns[i] for i in order]
        except Exception:
            return X.columns.tolist()[:k]

    @staticmethod
    def _pca(X: pd.DataFrame, k: int, state, cfg) -> list[str]:
        n_comp = min(k, X.shape[1], max(1, X.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=cfg.pipeline.random_state)
        transformed = pca.fit_transform(X.to_numpy())
        new_names = [f"pca_{i}" for i in range(n_comp)]
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
        if task == "regression":
            model = Lasso(alpha=cfg.feature_selection.lasso_alpha, random_state=cfg.pipeline.random_state).fit(X.to_numpy(), y)
            importance = np.abs(model.coef_)
        else:
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0 / cfg.feature_selection.lasso_alpha, random_state=cfg.pipeline.random_state
            ).fit(X.to_numpy(), y)
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        order = np.argsort(importance)[::-1][:k]
        return [X.columns[i] for i in order]

    @staticmethod
    def _rf(X: pd.DataFrame, y, task: str, k: int, cfg, state) -> list[str]:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=cfg.feature_selection.rf_n_estimators, random_state=cfg.pipeline.random_state
            ).fit(X.to_numpy(), y)
        else:
            model = RandomForestRegressor(
                n_estimators=cfg.feature_selection.rf_n_estimators, random_state=cfg.pipeline.random_state
            ).fit(X.to_numpy(), y)
        importance = model.feature_importances_
        order = np.argsort(importance)[::-1][:k]
        return [X.columns[i] for i in order]

    def postcondition(self, state: PipelineState) -> None:
        if state.selected_columns is None:
            raise PostconditionError("FeatureSelector: selected_columns is None")
