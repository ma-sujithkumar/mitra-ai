"""FeatureSelector — single-LLM-call selection over precomputed stat artifacts.

The deterministic stat phase (pipeline/feature_stats.py) writes mutual information,
information gain, mRMR ranking, variance, correlation pairs, clusters, linear baseline
and PCA artifacts to `.mitra/<run_id>/stats/`. This tool reads those artifacts,
makes ONE LLM call to decide which columns to keep/drop, and applies the result
deterministically. If the LLM call fails (or no model is configured), it falls back
to mRMR over all features at top_k_features.

The per-estimator helpers (_mrmr, _pca, _lasso, _laplacian_score, _top_mi,
_mi_scores, _information_gain) are reused by feature_stats.py so the math lives in
one place.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.responses import FeatureSelectionResponse, call_with_revision
from pipeline.state import PipelineState


def _deterministic_tag(columns) -> str:
    """Deterministic 8-char tag from column names.

    Python's built-in `hash` is salted per process and is forbidden anywhere a
    name lands in `state.df` or `feature_artifact.json` (spec §5, §7-AA).
    """
    return hashlib.md5("|".join(columns).encode("utf-8")).hexdigest()[:8]


SELECT_PROMPT = """You are a feature selection expert. Choose which feature columns to KEEP for a
{task} model. You are given precomputed statistics — you do NOT have the raw data.

## GOAL
Keep the most predictive, non-redundant features. Keep AT MOST {top_k} columns.
- Prefer columns with high mutual information and high information gain.
- Respect the mRMR ranking (minimum-redundancy maximum-relevance order).
- Drop near-zero-variance columns.
- For each high-correlation pair keep only ONE column (drop the redundant twin).

## PCA DECISION (size-dependent)
Decide whether to compress the kept features into PCA components (set "use_pca").
- Favor PCA when the dataset is high-dimensional: many features and/or few rows
  (rule of thumb: many features relative to rows, or > ~50 informative features),
  OR when many features are mutually correlated so a few components capture most variance.
- Skip PCA when there are few features, plenty of rows per feature, or interpretability
  matters and individual features are already strong and non-redundant.
- Use the PCA explained-variance stat below to judge how many components retain the variance.
- If use_pca is true you may set "pca_n_components" (else the pipeline uses the
  components needed to retain the configured variance). PCA replaces the kept raw
  columns with components; "keep" should still list the raw columns to feed into PCA.

## STATISTICS
n_rows: {n_rows}
n_features: {n_features}
features_per_row: {features_per_row}
linear_baseline_score: {linear_baseline}

Mutual information (column: score), highest first:
{mi_block}

Information gain (column: H(Y) - H(Y|X) in bits), highest first:
{ig_block}

mRMR ranking (best first):
{mrmr_block}

Low-variance columns (consider dropping):
{low_var_block}

Highly correlated pairs (keep one of each):
{corr_block}

PCA: {pca_block}

## RESPONSE SHAPE
Return ONLY a JSON object:
{{
  "keep": ["<column>", "..."],
  "drop": ["<column>", "..."],
  "use_pca": <true|false>,
  "pca_n_components": <int or null>,
  "rationale": "<one or two sentences, mention the PCA decision and why>"
}}
"""


class FeatureSelector(BaseTool):
    def __init__(self, model_call: Callable[[str], str] | None = None, judge=None):
        # model_call drives the single selection LLM call. judge kept only for
        # backward-compatible construction; no longer used.
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
        top_k = min(cfg.feature_selection.top_k_features, len(feature_cols))

        stats = self._load_stats(state)
        if self.model_call is None or stats is None:
            self._fallback(state, X, y, feature_cols, top_k,
                           reason="no model_call" if self.model_call is None else "no stats artifacts")
            return

        prompt = self._build_prompt(state, feature_cols, stats, top_k)
        parsed, source, failures = call_with_revision(
            self.model_call, prompt, FeatureSelectionResponse, set(), cfg,
            caller="FeatureSelector",
        )
        state.last_llm_source = source

        if parsed is None or source == "fallback":
            state.warnings.append(f"FeatureSelector LLM call failed (failures={failures}); fallback mRMR")
            self._fallback(state, X, y, feature_cols, top_k, reason="llm_failed")
            return

        keep = [c for c in parsed.keep if c in feature_cols]  # type: ignore[attr-defined]
        # De-dup, order-preserving.
        seen: set[str] = set()
        selected = [c for c in keep if not (c in seen or seen.add(c))][:top_k]
        if not selected:
            state.warnings.append("FeatureSelector LLM returned no valid columns; fallback mRMR")
            self._fallback(state, X, y, feature_cols, top_k, reason="empty_keep")
            return

        # The agent decided whether to compress the kept features via PCA.
        if getattr(parsed, "use_pca", False) and len(selected) >= 2:
            self._apply_pca(state, X[selected], parsed.pca_n_components, stats, top_k)
        else:
            state.selected_columns = selected
            state.selection_method = "llm_select"

    def _apply_pca(self, state: PipelineState, X_sub: pd.DataFrame, requested_n, stats: dict, top_k: int) -> None:
        """Replace the kept raw columns with PCA components. Component count is the
        agent's pca_n_components when valid, else the #components reaching
        pca_variance_retained (from the precomputed PCA stat)."""
        pca_stats = stats.get("pca") or {}
        default_n = pca_stats.get("n_components_for_threshold") or X_sub.shape[1]
        n = requested_n if (isinstance(requested_n, int) and requested_n > 0) else default_n
        n = max(1, min(int(n), X_sub.shape[1], top_k))
        names = self._pca(X_sub, n, state, state.config)
        if not names:
            state.warnings.append("FeatureSelector: PCA produced no components; keeping raw columns")
            state.selected_columns = list(X_sub.columns)[:top_k]
            state.selection_method = "llm_select"
            return
        state.selected_columns = names
        state.selection_method = f"llm_select+pca({len(names)})"

    # ---------- prompt assembly ----------

    def _load_stats(self, state: PipelineState) -> dict | None:
        if state.stats_dir is None or not state.stats_dir.exists():
            return None
        loaded: dict = {}
        for name in (
            "mutual_info", "information_gain", "mrmr_ranking", "variance",
            "correlation_pearson", "linear_baseline", "pca",
        ):
            path = state.stats_dir / f"{name}.json"
            if path.exists():
                try:
                    loaded[name] = json.loads(path.read_text(encoding="utf-8"))
                except Exception as e:
                    state.warnings.append(f"FeatureSelector: failed to read {name}.json: {e}")
        return loaded or None

    @staticmethod
    def _top_items(scores: dict, limit: int) -> str:
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return "\n".join(f"- {k}: {v:.4f}" for k, v in items) if items else "(none)"

    def _build_prompt(self, state: PipelineState, feature_cols: list[str], stats: dict, top_k: int) -> str:
        cfg = state.config
        mi = (stats.get("mutual_info") or {}).get("scores", {})
        ig = (stats.get("information_gain") or {}).get("scores", {})
        mrmr = (stats.get("mrmr_ranking") or {}).get("ranked", [])
        low_var = (stats.get("variance") or {}).get("low_variance", [])
        corr_pairs = (stats.get("correlation_pearson") or {}).get("high_pairs", [])
        baseline = (stats.get("linear_baseline") or {}).get("score", 0.0)
        pca = stats.get("pca") or {}

        corr_block = "\n".join(
            f"- {a} <-> {b} (corr={c:.2f})" for a, b, c in corr_pairs[:40]
        ) if corr_pairs else "(none)"
        pca_block = (
            f"{pca.get('n_components_for_threshold', 'n/a')} components explain "
            f"{pca.get('variance_retained', 'n/a')} of variance"
            if pca else "(not computed)"
        )
        n_rows = len(state.df)
        n_features = len(feature_cols)
        features_per_row = f"{n_features / max(1, n_rows):.4f}"
        return SELECT_PROMPT.format(
            task=state.task,
            top_k=top_k,
            n_rows=n_rows,
            n_features=n_features,
            features_per_row=features_per_row,
            linear_baseline=f"{baseline:.4f}",
            mi_block=self._top_items(mi, 40),
            ig_block=self._top_items(ig, 40),
            mrmr_block="\n".join(f"- {c}" for c in mrmr[:40]) if mrmr else "(none)",
            low_var_block=("\n".join(f"- {c}" for c in low_var) if low_var else "(none)"),
            corr_block=corr_block,
            pca_block=pca_block,
        )

    def _fallback(self, state: PipelineState, X: pd.DataFrame, y, feature_cols: list[str], top_k: int, reason: str) -> None:
        state.last_llm_source = "fallback"
        state.warnings.append(f"FeatureSelector fallback mRMR over all features ({reason})")
        selected = self._mrmr(X, y, state.task, top_k, state=state)
        if not selected:
            selected = feature_cols[:top_k]
        state.selected_columns = selected
        state.selection_method = "fallback:mrmr_all"

    # ---------- reusable estimators (shared with feature_stats.py) ----------

    @staticmethod
    def _mi_scores(X: pd.DataFrame, y, task: str, seed: int = 42) -> dict[str, float]:
        if X.shape[1] == 0:
            return {}
        try:
            if task == "classification":
                mi = mutual_info_classif(X.to_numpy(), y, random_state=seed)
            else:
                mi = mutual_info_regression(X.to_numpy(), y, random_state=seed)
            return {c: float(m) for c, m in zip(X.columns, mi)}
        except Exception:
            return {c: 0.0 for c in X.columns}

    def _per_col_mi(self, X: pd.DataFrame, y, task: str, state) -> dict[str, float]:
        return self._mi_scores(X, y, task, seed=state.config.pipeline.random_state)

    @staticmethod
    def _mrmr(X: pd.DataFrame, y, task: str, k: int, state=None, seed: int = 42) -> list[str]:
        if X.shape[1] == 0:
            return []
        if state is not None:
            seed = state.config.pipeline.random_state
        try:
            from mrmr import mrmr_classif, mrmr_regression
            y_series = pd.Series(y)
            if task == "classification":
                return mrmr_classif(X=X, y=y_series, K=min(k, X.shape[1]))
            return mrmr_regression(X=X, y=y_series, K=min(k, X.shape[1]))
        except Exception as e:
            if state is not None:
                state.warnings.append(f"mrmr unavailable: {e}; using top-MI proxy")
            return FeatureSelector._top_mi(X, y, task, k, seed=seed)

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
    def _information_gain(X: pd.DataFrame, y, task: str, cfg, seed: int = 42) -> dict[str, float]:
        """Per-feature information gain (entropy-based mutual information).

        Continuous features — and, for regression, the target — are first
        discretized via sklearn's ``KBinsDiscretizer`` (quantile strategy, n_bins
        from ``cfg.feature_selection.ig_n_bins``). Information gain is then
        sklearn's ``mutual_info_classif`` with ``discrete_features=True`` against
        the discretized target, which reduces to H(Y) − H(Y|X) over the empirical
        distribution.
        """
        from sklearn.preprocessing import KBinsDiscretizer
        if X.shape[1] == 0:
            return {}
        n_bins = cfg.feature_selection.ig_n_bins
        try:
            x_disc = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
            ).fit_transform(X.to_numpy())
            if task == "regression":
                y_disc = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
                ).fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
            else:
                y_disc = np.asarray(y)
            ig = mutual_info_classif(x_disc, y_disc, discrete_features=True, random_state=seed)
            return {c: float(s) for c, s in zip(X.columns, ig)}
        except Exception:
            return {c: 0.0 for c in X.columns}

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
    def _laplacian_score(X: pd.DataFrame, y, task: str, k: int, cfg, state) -> list[str]:
        """Top-k features by Laplacian Score (He, Cai, Niyogi 2005).

        Builds a symmetric k-NN affinity graph over rows via sklearn's
        ``kneighbors_graph``; the degree vector ``d`` gives D and the Laplacian is
        L = D − W. Each feature ``f`` is D-weighted mean-centred to ``f̃`` and
        scored by ``L_r = (f̃ᵀ L f̃) / (f̃ᵀ D f̃)``. Lower is better — features
        that vary smoothly across neighbouring samples preserve local manifold
        structure. Falls through to top-IG if sklearn raises.
        """
        from sklearn.neighbors import kneighbors_graph
        if X.shape[1] == 0 or k <= 0:
            return []
        k_eff = min(k, X.shape[1])
        if k_eff >= X.shape[1]:
            return list(X.columns)
        n_neighbors = min(cfg.feature_selection.laplacian_k_neighbors, X.shape[0] - 1)
        if n_neighbors < 1:
            return list(X.columns[:k_eff])
        try:
            W = kneighbors_graph(
                X.to_numpy(), n_neighbors=n_neighbors,
                mode="connectivity", include_self=False,
            )
            W = 0.5 * (W + W.T)  # symmetrize
            d = np.asarray(W.sum(axis=1)).ravel()
            d_total = float(d.sum())
            if d_total <= 0:
                return list(X.columns[:k_eff])
            scores: dict[str, float] = {}
            for col in X.columns:
                f = X[col].to_numpy(dtype=float)
                f_centered = f - float((d * f).sum()) / d_total
                denom = float((d * f_centered * f_centered).sum())
                if denom <= 0:
                    scores[col] = float("inf")
                    continue
                wf = np.asarray(W @ f_centered).ravel()
                numer = denom - float(f_centered @ wf)
                scores[col] = numer / denom
            order = sorted(scores.items(), key=lambda kv: kv[1])[:k_eff]
            return [c for c, _ in order]
        except Exception as e:
            if state is not None:
                state.warnings.append(f"laplacian_score unavailable: {e}; using top-IG proxy")
            seed = cfg.pipeline.random_state
            ig_scores = FeatureSelector._information_gain(X, y, task, cfg, seed=seed)
            order = sorted(ig_scores.items(), key=lambda kv: kv[1], reverse=True)[:k_eff]
            return [c for c, _ in order]

    def postcondition(self, state: PipelineState) -> None:
        if state.selected_columns is None:
            raise PostconditionError("FeatureSelector: selected_columns is None")
