import pandas as pd
from sklearn.preprocessing import LabelEncoder

from backend.agents.feature_engineering.base import BaseTool, PostconditionError, PreconditionError
from backend.agents.feature_engineering.state import PipelineState


class Encoder(BaseTool):
    def precondition(self, state: PipelineState) -> None:
        if state.column_types is None:
            raise PreconditionError("Encoder: state.column_types is None")
        if not state.pre_encoding_done:
            raise PreconditionError("Encoder: pre_encoding_done is False (creator.run_pre must run first)")

    def run(self, state: PipelineState) -> None:
        df = state.df
        for col in list(df.columns):
            t = state.column_types.get(col, "numeric")
            # Numeric-coded categoricals are passed through; no re-encoding needed.
            if t in {"categorical", "binary"} and not pd.api.types.is_numeric_dtype(df[col]):
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                state.transformers.append({
                    "step": "encoding",
                    "column": col,
                    "strategy": "label",
                    "classes": [str(c) for c in encoder.classes_.tolist()],
                })

        # Target encoding if classification with string target (skip for clustering)
        if state.task == "classification" and state.target is not None and not pd.api.types.is_numeric_dtype(state.target):
            encoder = LabelEncoder()
            state.target = pd.Series(
                encoder.fit_transform(state.target.astype(str)), index=state.target.index, name=state.target_column
            )
            state.transformers.append({
                "step": "encoding",
                "column": state.target_column,
                "strategy": "label",
                "classes": [str(c) for c in encoder.classes_.tolist()],
            })

    def postcondition(self, state: PipelineState) -> None:
        non_numeric = [c for c in state.df.columns if not pd.api.types.is_numeric_dtype(state.df[c])]
        if non_numeric:
            raise PostconditionError(f"Encoder: non-numeric columns remain: {non_numeric}")
