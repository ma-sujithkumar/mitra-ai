import numpy as np
import pandas as pd
import ray
from scipy import stats as scistats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from backend.agents.feature_engineering.base import BaseTool, PostconditionError, PreconditionError
from backend.agents.feature_engineering.parallel import _univariate_stats, run_parallel
from backend.agents.feature_engineering.state import PipelineState


class DataProfiler(BaseTool):
    def precondition(self, state: PipelineState) -> None:
        if state.df is None:
            raise PreconditionError("DataProfiler: state.df is None")
        # Unsupervised runs have no target; mutual-information-with-target is
        # simply skipped below.

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config
        seed = cfg.pipeline.random_state
        items = [(col, {"data": df[col].tolist()}) for col in df.columns]
        results = run_parallel(_univariate_stats, items)
        profile: dict = {name: stats for name, stats in results}

        # mutual-information-with-target is target-dependent; skip it (set to 0.0)
        # for unsupervised runs where there is no target.
        target_arr = state.target.to_numpy() if state.target is not None else None
        for col in df.columns:
            if target_arr is None:
                profile[col]["mi_with_target"] = 0.0
                continue
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

        # Correlation matrix kept (cheap, vectorised) for any downstream consumer.
        # Clusters, linear baseline, joint-MI pairs and null-mask correlations are
        # NOT computed here anymore: clusters/baseline are computed once in
        # pipeline.feature_stats, and the joint-MI / null-mask signals only fed the
        # removed LLM evidence packets.
        numeric_df = df.select_dtypes(include=[np.number])
        profile["_correlation_matrix"] = numeric_df.corr().to_dict() if not numeric_df.empty else {}

        state.profile = profile

    def postcondition(self, state: PipelineState) -> None:
        if state.profile is None:
            raise PostconditionError("DataProfiler: state.profile is None")
