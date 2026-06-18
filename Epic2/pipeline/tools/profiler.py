import numpy as np
import pandas as pd
import ray
from scipy import stats as scistats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.parallel import (
    _univariate_stats,
    compute_correlation_clusters,
    compute_linear_baseline,
    run_parallel,
)
from pipeline.state import PipelineState


class DataProfiler(BaseTool):
    def precondition(self, state: PipelineState) -> None:
        if state.df is None:
            raise PreconditionError("DataProfiler: state.df is None")
        if state.target is None:
            raise PreconditionError("DataProfiler: state.target is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config
        seed = cfg.pipeline.random_state
        items = [(col, {"data": df[col].tolist()}) for col in df.columns]
        results = run_parallel(_univariate_stats, items)
        profile: dict = {name: stats for name, stats in results}

        target_arr = state.target.to_numpy()
        for col in df.columns:
            col_data = pd.to_numeric(df[col], errors="coerce")
            mask = ~col_data.isna()
            if mask.sum() < 5:
                profile[col]["mi_with_target"] = 0.0
                continue
            try:
                X = col_data[mask].to_numpy().reshape(-1, 1)
                y = target_arr[mask.to_numpy()]
                if state.task == "classification":
                    mi = float(mutual_info_classif(X, y, random_state=seed)[0])
                else:
                    mi = float(mutual_info_regression(X, y, random_state=seed)[0])
                profile[col]["mi_with_target"] = mi
            except Exception:
                profile[col]["mi_with_target"] = 0.0

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            profile["_correlation_matrix"] = numeric_df.corr().to_dict()
        else:
            profile["_correlation_matrix"] = {}

        # Spec §6 / plan ambiguity #29 (resolved as gaps.txt #1): profiler
        # surfaces _clusters, _linear_baseline_score, _joint_mi_pairs into
        # state.profile. Selector still recomputes its own clusters after the
        # column set grows; profiler's values seed the Creator's co-occurrence
        # ranking and the Selector's evidence packet when nothing has changed.
        if not numeric_df.empty:
            profile["_clusters"] = compute_correlation_clusters(
                numeric_df, cut_threshold=cfg.feature_selection.cluster_cut_threshold
            )
            profile["_linear_baseline_score"] = compute_linear_baseline(
                numeric_df, target_arr, state.task,
                k=cfg.feature_selection.linear_baseline_k,
                seed=seed,
            )
            profile["_joint_mi_pairs"] = _joint_mi_pairs(
                numeric_df, target_arr, state.task,
                profile=profile, seed=seed, top_n=20,
            )
        else:
            profile["_clusters"] = {}
            profile["_linear_baseline_score"] = 0.0
            profile["_joint_mi_pairs"] = []

        null_mask_df = df.isna().astype(int)
        for col in df.columns:
            if df[col].isna().sum() == 0:
                profile[col]["null_mask_corr"] = {}
                continue
            corrs: dict[str, float] = {}
            for other in df.columns:
                if other == col:
                    continue
                other_num = pd.to_numeric(df[other], errors="coerce")
                if other_num.notna().sum() < 5:
                    continue
                try:
                    c = float(null_mask_df[col].corr(other_num.fillna(other_num.median())))
                    if not np.isnan(c) and abs(c) > 0.1:
                        corrs[other] = c
                except Exception:
                    pass
            profile[col]["null_mask_corr"] = corrs

        state.profile = profile

    def postcondition(self, state: PipelineState) -> None:
        if state.profile is None:
            raise PostconditionError("DataProfiler: state.profile is None")


def _joint_mi_pairs(
    numeric_df: pd.DataFrame,
    target_arr: np.ndarray,
    task: str,
    profile: dict,
    seed: int,
    top_n: int,
) -> list[tuple[str, str, float]]:
    """Top column pairs by joint MI with target.

    Candidate pairs are first ranked by the cheap MI-product proxy to keep cost
    bounded, then the top 3*top_n are re-scored with true joint MI on the
    concatenated [col_a, col_b] feature against the target.
    """
    cols = [c for c in numeric_df.columns]
    if len(cols) < 2:
        return []

    candidates: list[tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        mi_a = profile.get(a, {}).get("mi_with_target") or 0.0
        for b in cols[i + 1:]:
            mi_b = profile.get(b, {}).get("mi_with_target") or 0.0
            candidates.append((a, b, float(mi_a * mi_b)))
    candidates.sort(key=lambda t: t[2], reverse=True)
    short = candidates[: max(top_n * 3, top_n)]

    scored: list[tuple[str, str, float]] = []
    for a, b, _ in short:
        try:
            X = numeric_df[[a, b]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
            if task == "classification":
                mi_pair = float(mutual_info_classif(X, target_arr, random_state=seed).sum())
            else:
                mi_pair = float(mutual_info_regression(X, target_arr, random_state=seed).sum())
            scored.append((a, b, mi_pair))
        except Exception:
            scored.append((a, b, 0.0))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_n]
