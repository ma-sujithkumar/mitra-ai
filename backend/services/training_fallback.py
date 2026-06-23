from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

ProblemType = Literal["classification", "regression"]


@dataclass(frozen=True)
class FallbackArtifactsResult:
    created_paths: list[Path]
    problem_type: ProblemType
    target_column: str
    feature_columns: list[str]
    dropped_columns: list[str]
    train_rows: int
    test_rows: int


class FallbackTrainingArtifactError(Exception):
    """Raised when missing Epic-3 artifacts cannot be inferred safely."""


class FallbackTrainingArtifactBuilder:
    """Create minimal Epic-3 training artifacts from an uploaded tabular CSV.

    This is a safety net for the shared Epic-1/Epic-3 boundary: if LLM-based
    metadata/model-selection fails, the UI can still demo and validate the
    deterministic Epic-3 training path using schema-derived metadata, a simple
    model shortlist, and a train/test split.
    """

    classification_models = (
        ("DecisionTreeClassifier", "Fast tree baseline for fallback classification."),
        ("RandomForestClassifier", "Robust ensemble baseline for fallback classification."),
        ("GaussianNB", "Fast probabilistic baseline for fallback classification."),
    )
    regression_models = (
        ("DecisionTreeRegressor", "Fast tree baseline for fallback regression."),
        ("Ridge", "Fast linear baseline for fallback regression."),
        ("DummyRegressor", "Very fast sanity-check baseline for fallback regression."),
    )

    def __init__(self, *, train_fraction: float = 0.8) -> None:
        if not 0.05 < train_fraction < 0.95:
            raise ValueError("train_fraction must be between 0.05 and 0.95")
        self.train_fraction = train_fraction

    def ensure(
        self,
        *,
        session_path: Path,
        metadata_path: Path,
        model_config_path: Path,
        train_path: Path,
        test_path: Path,
        target_column: str | None,
        problem_type: str | None,
    ) -> FallbackArtifactsResult:
        source_csv = session_path / "data" / "data.csv"
        if not source_csv.is_file():
            raise FallbackTrainingArtifactError(
                f"Cannot create fallback artifacts because uploaded data.csv is missing: {source_csv}"
            )

        existing_metadata = self._read_json(metadata_path)
        resolved_target = self._resolve_target_column(
            requested=target_column,
            metadata=existing_metadata,
            session_path=session_path,
        )
        if not resolved_target:
            raise FallbackTrainingArtifactError(
                "Cannot create fallback artifacts without a target column. "
                "Select a target column before starting training."
            )

        df = pd.read_csv(source_csv)
        if resolved_target not in df.columns:
            raise FallbackTrainingArtifactError(
                f"Target column '{resolved_target}' is not present in uploaded data.csv"
            )

        resolved_problem_type = self._resolve_problem_type(
            requested=problem_type,
            metadata=existing_metadata,
            target_series=df[resolved_target],
        )

        prepared_df, feature_columns, dropped_columns = self._prepare_training_frame(
            df=df,
            target_column=resolved_target,
            problem_type=resolved_problem_type,
        )
        if len(prepared_df) < 2:
            raise FallbackTrainingArtifactError(
                "Cannot create train/test split because fewer than two usable rows remain."
            )
        if not feature_columns:
            raise FallbackTrainingArtifactError(
                "Cannot create fallback training data because no numeric feature columns remain."
            )

        created_paths: list[Path] = []
        if not train_path.is_file() or not test_path.is_file():
            train_df, test_df = self._split(prepared_df, resolved_target, resolved_problem_type)
            train_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.parent.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            created_paths.extend([train_path, test_path])
        else:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

        if not metadata_path.is_file():
            payload = self._metadata_payload(
                session_path=session_path,
                problem_type=resolved_problem_type,
                target_column=resolved_target,
                feature_columns=feature_columns,
                row_count=len(prepared_df),
                column_count=len(prepared_df.columns),
                dropped_columns=dropped_columns,
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            created_paths.append(metadata_path)

        if not model_config_path.is_file():
            payload = self._model_config_payload(resolved_problem_type)
            model_config_path.parent.mkdir(parents=True, exist_ok=True)
            model_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            created_paths.append(model_config_path)

        return FallbackArtifactsResult(
            created_paths=created_paths,
            problem_type=resolved_problem_type,
            target_column=resolved_target,
            feature_columns=feature_columns,
            dropped_columns=dropped_columns,
            train_rows=len(train_df),
            test_rows=len(test_df),
        )

    def _prepare_training_frame(
        self,
        *,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        work_df = df.dropna(subset=[target_column]).copy()
        numeric_features = [
            column
            for column in work_df.select_dtypes(include=["number", "bool"]).columns.tolist()
            if column != target_column
        ]
        dropped_columns = [
            column
            for column in work_df.columns
            if column not in numeric_features and column != target_column
        ]
        work_df = work_df[numeric_features + [target_column]].copy()
        for column in numeric_features:
            if work_df[column].isna().any():
                median = work_df[column].median()
                work_df[column] = work_df[column].fillna(0 if pd.isna(median) else median)
        return work_df, numeric_features, dropped_columns

    def _split(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_size = 1.0 - self.train_fraction
        stratify = None
        if problem_type == "classification":
            counts = df[target_column].value_counts(dropna=False)
            if len(counts) > 1 and int(counts.min()) >= 2:
                stratify = df[target_column]
        return train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

    def _metadata_payload(
        self,
        *,
        session_path: Path,
        problem_type: ProblemType,
        target_column: str,
        feature_columns: list[str],
        row_count: int,
        column_count: int,
        dropped_columns: list[str],
    ) -> dict[str, Any]:
        session_json = self._read_json(session_path / "session.json") or {}
        return {
            "problem_type": problem_type,
            "problem_subtype": problem_type,
            "data_format": "tabular",
            "target_column": target_column,
            "target_col": target_column,
            "target_col_type": "numeric" if problem_type == "regression" else "categorical",
            "output_cols": [target_column],
            "input_cols": feature_columns,
            "row_count": row_count,
            "column_count": column_count,
            "dropped_non_numeric_input_cols": dropped_columns,
            "description": (
                "Fallback metadata generated deterministically because LLM metadata "
                "or model-selection artifacts were unavailable."
            ),
            "source": "epic3_fallback_artifact_builder",
            "original_filename": session_json.get("original_filename"),
        }

    def _model_config_payload(self, problem_type: ProblemType) -> list[dict[str, Any]]:
        models = self.regression_models if problem_type == "regression" else self.classification_models
        return [
            {
                "name": model_name,
                "model_name": model_name,
                "task_type": problem_type,
                "priority": index,
                "rationale": rationale,
                "default_hyperparameters": {},
                "hp_space": {},
                "source": "epic3_fallback_artifact_builder",
            }
            for index, (model_name, rationale) in enumerate(models, start=1)
        ]

    def _resolve_target_column(
        self,
        *,
        requested: str | None,
        metadata: dict[str, Any] | None,
        session_path: Path,
    ) -> str | None:
        if requested:
            return requested
        if metadata:
            for key in ("target_column", "target_col"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    return value
            output_cols = metadata.get("output_cols")
            if isinstance(output_cols, list) and output_cols:
                return str(output_cols[0])
        run_config = self._read_json(session_path / "reports" / "run_config.json")
        if run_config:
            value = run_config.get("target_col")
            if isinstance(value, str) and value:
                return value
        return None

    def _resolve_problem_type(
        self,
        *,
        requested: str | None,
        metadata: dict[str, Any] | None,
        target_series: pd.Series,
    ) -> ProblemType:
        for value in (requested, metadata.get("problem_subtype") if metadata else None, metadata.get("problem_type") if metadata else None):
            if value in {"classification", "regression"}:
                return value  # type: ignore[return-value]
            if value == "supervised" and metadata:
                target_type = metadata.get("target_col_type")
                if target_type == "numeric":
                    return "regression"
                return "classification"
        if pd.api.types.is_numeric_dtype(target_series):
            return "regression"
        return "classification"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None
