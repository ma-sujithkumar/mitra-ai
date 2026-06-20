"""Pre-training pipeline preparation service.

Chains the steps that must run before model training begins:
  1. Feature engineering  -> engineered_dataset.csv + feature_artifact.json
  2. Feature-selection adapter -> feature_selection.json
  3. Train / test split        -> data/train.csv, data/test.csv
  4. Model selection           -> model_config.json

All outputs land in the session directory under the configured subdirectories.
Callers (TrainingService, run_pipeline CLI) call PipelinePrep.run() and then
hand model_config.json to TrainingOrchestrator.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backend.agents.feature_engineering.orchestrator import FeatureEngineerOrchestrator
from backend.agents.metadata_gen_agent import LlmSettings
from backend.agents.model_selection.schemas import EngineeredFeature, FeatureSelectionInput
from backend.agents.model_selection.selector import select_models
from backend.config_loader import ConfigLoader
from backend.orchestration.d2v_bridge import D2VBridge
from backend.services.feature_status import FeatureEngineeringStatusReader

logger = logging.getLogger(__name__)


class FeatureArtifactAdapter:
    """Converts feature_artifact.json (Epic-2 schema) to FeatureSelectionInput."""

    @staticmethod
    def adapt(feature_artifact: dict) -> FeatureSelectionInput:
        """Map Epic-2 feature_artifact fields to model_selection FeatureSelectionInput."""
        keep_cols: list[str] = []
        drop_cols: list[str] = []
        engineered: list[EngineeredFeature] = []
        rationale: dict[str, str] = {}

        for feature_name, feature_info in feature_artifact.get("features", {}).items():
            action = feature_info.get("action", "keep")
            if action == "drop":
                drop_cols.append(feature_name)
                rationale[feature_name] = feature_info.get("reason", "dropped by feature engineering")
            else:
                keep_cols.append(feature_name)
                if feature_info.get("engineered", False):
                    engineered.append(
                        EngineeredFeature(
                            name=feature_name,
                            source=feature_info.get("source", ""),
                            description=feature_info.get("description", ""),
                        )
                    )
                rationale[feature_name] = feature_info.get("reason", "kept by feature engineering")

        return FeatureSelectionInput(
            keep=keep_cols,
            drop=drop_cols,
            engineered=engineered,
            rationale=rationale,
        )


class PipelinePrep:
    """Orchestrates all pre-training steps for one pipeline run."""

    def __init__(
        self,
        config_loader: ConfigLoader,
        session_dir: Path,
        llm_settings: Optional[LlmSettings] = None,
    ) -> None:
        self.config_loader = config_loader
        self.session_dir = session_dir
        # Resolved LLM credentials (LlmSettingsResolver / .env). Feature
        # engineering uses these via build_llm_model -- the same path as
        # metadata_gen -- and reads no key from config.yaml.
        self.llm_settings = llm_settings
        self.data_dir = session_dir / "data"
        self.reports_dir = session_dir / "reports"

    def run(
        self,
        raw_data_path: Path,
        target_column: str,
        metadata_path: Path,
        mini_data_path: Optional[Path] = None,
        max_models: Optional[int] = None,
    ) -> Path:
        """Run all pre-training steps and return the path to model_config.json.

        Args:
            raw_data_path: Path to the uploaded dataset (CSV).
            target_column: Name of the prediction target column.
            metadata_path: Path to metadata.json produced by the metadata agent.
            mini_data_path: Optional path to mini_data.csv (sampled). Derived from
                raw_data_path if not provided.
            max_models: Override config max_ml_models for this run.

        Returns:
            Path to the written model_config.json.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        max_models = max_models or self.config_loader.pipeline.max_ml_models
        mini_data_path = mini_data_path or raw_data_path

        # Step 1: feature engineering — artifacts land in reports/feature_engineering/
        feature_output_dir, run_id, resolved_task = self._run_feature_engineering(
            raw_data_path=raw_data_path,
            target_column=target_column,
        )
        logger.info("=> feature engineering complete: run_id=%s dir=%s", run_id, feature_output_dir)

        engineered_csv = feature_output_dir / "engineered_dataset.csv"
        feature_artifact_path = feature_output_dir / "feature_artifact.json"

        # Step 1b: persist structured feature_run.json so the UI can read status.
        self._write_feature_run_status(feature_output_dir)

        # Step 1c: Dataset2Vec warm-start query (non-fatal).
        self._run_d2v_query(
            engineered_csv=engineered_csv,
            target_column=target_column,
            task_type=resolved_task,
        )

        # Step 2: adapt feature_artifact -> feature_selection.json
        feature_selection_path = self.reports_dir / "feature_selection.json"
        self._adapt_feature_selection(feature_artifact_path, feature_selection_path)
        logger.info("=> feature_selection.json written: %s", feature_selection_path)

        # Step 3: train / test split on the engineered dataset
        self._split_dataset(
            engineered_csv=engineered_csv,
            target_column=target_column,
            split_ratio=self.config_loader.pipeline.train_test_split,
        )
        logger.info(
            "=> train/test split done: ratio=%.2f",
            self.config_loader.pipeline.train_test_split,
        )

        # Step 4: model selection. The model-selection schema (MetadataInput)
        # requires the normalized form (enum problem_type + string input_cols),
        # but the metadata agent emits problem_type='supervised' and input_cols
        # as {name, col_type} dicts. Normalize before selecting.
        selection_metadata_path = self._normalize_metadata_for_selection(
            metadata_path=metadata_path,
            target_column=target_column,
            resolved_task=resolved_task,
        )
        model_config_path = self.reports_dir / "model_config.json"
        self._run_model_selection(
            metadata_path=selection_metadata_path,
            feature_selection_path=feature_selection_path,
            mini_data_path=mini_data_path,
            model_config_path=model_config_path,
            max_models=max_models,
        )
        logger.info("=> model_config.json written: %s", model_config_path)

        return model_config_path

    def _run_feature_engineering(
        self,
        raw_data_path: Path,
        target_column: str,
    ) -> tuple[Path, str, str]:
        """Run FE orchestrator and return (output_dir, run_id, task_type)."""
        if self.llm_settings is None or not (self.llm_settings.api_key or self.llm_settings.gateway_url):
            raise RuntimeError(
                "Feature engineering requires resolved LLM credentials "
                "(LlmSettings from LlmSettingsResolver / .env)."
            )
        fe_output_dir = self.session_dir / self.config_loader.feature_engineering_api.output_subdir
        orchestrator = FeatureEngineerOrchestrator(
            data_path=raw_data_path,
            target_column=target_column,
            model_string=self.llm_settings.model,
            output_dir=fe_output_dir,
            llm_settings=self.llm_settings,
        )
        output_path, run_id = orchestrator.run()
        # Read the resolved task type from the artifact so D2V can use it.
        artifact_path = output_path / "feature_artifact.json"
        resolved_task = "classification"
        if artifact_path.is_file():
            artifact_data = json.loads(artifact_path.read_text(encoding="utf-8"))
            resolved_task = artifact_data.get("task", "classification")
        return output_path, run_id, resolved_task

    def _write_feature_run_status(self, fe_dir: Path) -> None:
        """Parse FE raw artifacts and write feature_run.json atomically."""
        reader = FeatureEngineeringStatusReader(self.session_dir)
        status_payload = reader.read()
        status_path = fe_dir / self.config_loader.feature_engineering_api.run_status_filename
        fe_dir.mkdir(parents=True, exist_ok=True)
        # Write atomically using tempfile to avoid partial reads by the UI poller.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=fe_dir, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as status_file:
                json.dump(status_payload, status_file, indent=2)
            os.replace(tmp_path, status_path)
        except Exception:
            os.unlink(tmp_path)
            raise
        logger.info("=> feature_run.json written: %s", status_path)

    def _run_d2v_query(
        self,
        engineered_csv: Path,
        target_column: str,
        task_type: str,
    ) -> None:
        """Query Dataset2Vec for similar past datasets; non-fatal on failure.

        Dataset2Vec only supports classification datasets, so it is skipped for
        regression/unsupervised tasks (the encoder/meta-KB are classification-only).
        """
        if task_type != "classification":
            logger.info("=> D2V skipped: task_type=%s (classification only)", task_type)
            return
        d2v_db_dir = self.config_loader.paths.d2v_db_dir
        if not d2v_db_dir:
            logger.debug("=> D2V_DB_DIR not configured; skipping Dataset2Vec query")
            return
        db_path = Path(d2v_db_dir)
        if not db_path.is_absolute():
            db_path = self.config_loader.repo_root / db_path
        output_path = self.reports_dir / "dataset_prior.json"
        try:
            bridge = D2VBridge(db_dir=db_path)
            prior = bridge.query(
                csv_path=engineered_csv,
                target_column=target_column,
                task_type=task_type,
            )
            if prior is not None:
                output_path.write_text(prior.model_dump_json(indent=2), encoding="utf-8")
                logger.info("=> dataset_prior.json written: %s", output_path)
            else:
                logger.info("=> D2V returned no prior (cold start)")
        except Exception as d2v_error:
            logger.warning("=> D2V query failed (non-fatal): %s", d2v_error)

    def _adapt_feature_selection(
        self,
        feature_artifact_path: Path,
        output_path: Path,
    ) -> None:
        with feature_artifact_path.open(encoding="utf-8") as artifact_file:
            artifact = json.load(artifact_file)
        selection = FeatureArtifactAdapter.adapt(artifact)
        output_path.write_text(
            selection.model_dump_json(indent=2), encoding="utf-8"
        )

    def _split_dataset(
        self,
        engineered_csv: Path,
        target_column: str,
        split_ratio: float,
    ) -> None:
        dataframe = pd.read_csv(engineered_csv)
        cutoff_index = int(len(dataframe) * split_ratio)
        train_df = dataframe.iloc[:cutoff_index]
        test_df = dataframe.iloc[cutoff_index:]
        train_df.to_csv(self.data_dir / "train.csv", index=False)
        test_df.to_csv(self.data_dir / "test.csv", index=False)
        logger.debug(
            "=> split: total=%d train=%d test=%d",
            len(dataframe),
            len(train_df),
            len(test_df),
        )

    def _normalize_metadata_for_selection(
        self,
        metadata_path: Path,
        target_column: str,
        resolved_task: str,
    ) -> Path:
        """Write a model_selection.MetadataInput-compatible metadata file.

        Converts the metadata agent's output into the strict shape the
        model-selection schema expects:
          - problem_type: 'supervised' -> 'classification'/'regression' (uses the
            FE-resolved task), 'unsupervised' kept, enum values passed through.
          - input_cols / drop_cols: [{name, col_type}, ...] -> [name, ...].
          - output_cols: filled from the target column when empty.
          - col_types: {name: col_type} derived from the dict-form columns.
        """
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))

        problem_type = raw.get("problem_type")
        if problem_type not in {"classification", "regression", "unsupervised"}:
            # 'supervised' (or anything non-enum): use the task FE already resolved.
            problem_type = resolved_task if resolved_task in {"classification", "regression"} else "classification"

        def _names(columns: Any) -> list[str]:
            names: list[str] = []
            for column in columns or []:
                if isinstance(column, dict):
                    name = column.get("name")
                    if name:
                        names.append(str(name))
                elif column:
                    names.append(str(column))
            return names

        def _col_types(columns: Any) -> dict[str, str]:
            mapping: dict[str, str] = {}
            for column in columns or []:
                if isinstance(column, dict) and column.get("name") and column.get("col_type"):
                    mapping[str(column["name"])] = str(column["col_type"])
            return mapping

        input_cols = _names(raw.get("input_cols"))
        drop_cols = _names(raw.get("drop_cols") or raw.get("cols_to_drop"))
        output_cols = _names(raw.get("output_cols")) or ([target_column] if target_column else [])
        col_types = _col_types(raw.get("input_cols"))
        if target_column and raw.get("target_col_type"):
            col_types[target_column] = str(raw["target_col_type"])

        normalized = {
            "problem_type": problem_type,
            "output_cols": output_cols,
            "input_cols": input_cols,
            "drop_cols": drop_cols,
            "col_types": col_types,
            "data_format": raw.get("data_format", "tabular"),
            "user_description": raw.get("user_description", ""),
        }
        selection_path = self.reports_dir / "metadata_model_selection.json"
        selection_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        return selection_path

    def _run_model_selection(
        self,
        metadata_path: Path,
        feature_selection_path: Path,
        mini_data_path: Path,
        model_config_path: Path,
        max_models: int,
    ) -> None:
        model_library_root = self.config_loader.training_api.model_library_root
        select_models(
            metadata_path=metadata_path,
            feature_selection_path=feature_selection_path,
            mini_data_path=mini_data_path,
            model_library_root=model_library_root,
            output_path=model_config_path,
            max_models=max_models,
            report_path=self.reports_dir / "model_selection_report.md",
        )
