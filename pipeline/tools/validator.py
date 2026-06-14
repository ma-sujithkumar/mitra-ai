import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.state import PipelineState


class FeatureValidator(BaseTool):
    def precondition(self, state: PipelineState) -> None:
        if state.selected_columns is None:
            raise PreconditionError("FeatureValidator: selected_columns is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
        keep_cols = [c for c in state.selected_columns if c in df.columns]
        missing = [c for c in state.selected_columns if c not in df.columns]
        for c in missing:
            state.warnings.append(f"Selected column missing from df, dropped: {c}")

        coerced: list[str] = []
        for col in list(keep_cols):
            if not pd.api.types.is_float_dtype(df[col]):
                ok = self._try_coerce(df, col, state)
                if not ok:
                    keep_cols.remove(col)
                    state.warnings.append(f"Coercion failed for {col}; dropped from selection")
                else:
                    coerced.append(col)

        if df[keep_cols].isna().sum().sum() > 0:
            df[keep_cols] = df[keep_cols].fillna(0.0)
            state.warnings.append("Residual NaNs in selected columns filled with 0.0")

        state.df = df[keep_cols].copy()
        state.df[state.target_column] = state.target.to_numpy()
        state.selected_columns = keep_cols

    @staticmethod
    def _try_coerce(df: pd.DataFrame, col: str, state: PipelineState) -> bool:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
            return True
        except Exception:
            pass
        try:
            df[col] = pd.to_datetime(df[col], errors="raise").astype("int64").astype(float)
            return True
        except Exception:
            pass
        try:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str)).astype(float)
            state.transformers.append({
                "step": "validator_coerce_label",
                "column": col,
                "strategy": "label",
                "classes": [str(c) for c in enc.classes_.tolist()],
            })
            return True
        except Exception:
            return False

    def postcondition(self, state: PipelineState) -> None:
        if state.target_column not in state.df.columns:
            raise PostconditionError("FeatureValidator: target column missing from output df")
        if state.df.columns[-1] != state.target_column:
            raise PostconditionError("FeatureValidator: target must be last column")
        if state.df.isna().sum().sum() > 0:
            raise PostconditionError("FeatureValidator: NaNs remain in output")
