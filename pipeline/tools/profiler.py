import numpy as np
import pandas as pd
import ray
from scipy import stats as scistats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.parallel import _univariate_stats, run_parallel
from pipeline.state import PipelineState


class DataProfiler(BaseTool):
    def precondition(self, state: PipelineState) -> None:
        if state.df is None:
            raise PreconditionError("DataProfiler: state.df is None")
        if state.target is None:
            raise PreconditionError("DataProfiler: state.target is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
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
                    mi = float(mutual_info_classif(X, y, random_state=42)[0])
                else:
                    mi = float(mutual_info_regression(X, y, random_state=42)[0])
                profile[col]["mi_with_target"] = mi
            except Exception:
                profile[col]["mi_with_target"] = 0.0

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            profile["_correlation_matrix"] = numeric_df.corr().to_dict()
        else:
            profile["_correlation_matrix"] = {}

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
