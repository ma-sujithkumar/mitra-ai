import pandas as pd

from backend.agents.feature_engineering.base import BaseTool, PostconditionError, PreconditionError
from backend.agents.feature_engineering.state import PipelineState


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

        for col in list(keep_cols):
            if not pd.api.types.is_float_dtype(df[col]):
                ok = self._try_coerce(df, col)
                if not ok:
                    keep_cols.remove(col)
                    state.warnings.append(
                        f"Coercion failed for {col} (non-numeric, non-datetime); dropped from selection"
                    )

        if df[keep_cols].isna().sum().sum() > 0:
            df[keep_cols] = df[keep_cols].fillna(0.0)
            state.warnings.append("Residual NaNs in selected columns filled with 0.0")

        state.df = df[keep_cols].copy()
        # Unsupervised runs have no target column to re-attach.
        if state.target is not None and state.target_column is not None:
            state.df[state.target_column] = state.target.to_numpy()
        state.selected_columns = keep_cols

    @staticmethod
    def _try_coerce(df: pd.DataFrame, col: str) -> bool:
        """Strict coercion: float → datetime only.

        LabelEncoding here would silently relabel a column that should have
        been encoded earlier by the Encoder, hiding upstream typing or
        normalization bugs. Spec §5 "FeatureValidator coercion is stricter",
        §7-X.
        """
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
            return True
        except Exception:
            pass
        try:
            df[col] = pd.to_datetime(df[col], errors="raise").astype("int64").astype(float)
            return True
        except Exception:
            return False

    def postcondition(self, state: PipelineState) -> None:
        # Target-column placement checks only apply to supervised runs.
        if state.target is not None and state.target_column is not None:
            if state.target_column not in state.df.columns:
                raise PostconditionError("FeatureValidator: target column missing from output df")
            if state.df.columns[-1] != state.target_column:
                raise PostconditionError("FeatureValidator: target must be last column")
        if state.df.isna().sum().sum() > 0:
            raise PostconditionError("FeatureValidator: NaNs remain in output")
        if state.row_count_after_outlier is not None and len(state.df) != state.row_count_after_outlier:
            raise PostconditionError(
                f"FeatureValidator: row count {len(state.df)} != post-outlier {state.row_count_after_outlier}"
            )
