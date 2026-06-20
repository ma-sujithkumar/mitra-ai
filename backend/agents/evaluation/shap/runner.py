"""SHAPRunner: orchestrates the full per-model SHAP pipeline.

Sequences the existing SHAP components (ModelLoader -> DatasetLoader ->
SchemaValidator -> ExplainerFactory -> SHAPService -> Exporters) so the
DAG eval_runner can call a single run_for_model() instead of staging each
component individually.  No logic is duplicated from the existing classes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.agents.evaluation.shap.explainers.explainer_factory import ExplainerFactory
from backend.agents.evaluation.shap.explainers.shap_service import SHAPService
from backend.agents.evaluation.shap.exporters.global_importance_exporter import GlobalImportanceExporter
from backend.agents.evaluation.shap.exporters.feature_shap_mapping_exporter import FeatureSHAPMappingExporter
from backend.agents.evaluation.shap.loaders.dataset_loader import DatasetLoader
from backend.agents.evaluation.shap.loaders.model_loader import ModelLoader
from backend.agents.evaluation.shap.session_context import SessionContext
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger
from backend.agents.evaluation.shap.validators.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class SHAPRunner:
    """Runs the full SHAP explainability pipeline for one model and writes outputs.

    Outputs per model (under <shap_output_dir>/<model_name>/):
      csv/global_feature_importance.csv   — used by judge adapter
      csv/feature_shap_mapping.csv
    """

    def __init__(self, shap_output_dir: Path, session_id: str) -> None:
        self.shap_output_dir = Path(shap_output_dir)
        self.session_id = session_id

    def run_for_model(
        self,
        model_name: str,
        model_path: Path,
        dataset_path: Path,
        target_column: str,
        max_shap_samples: int = 1000,
    ) -> Optional[Path]:
        """Run SHAP for one trained model and return the output directory path.

        Args:
            model_name: Human-readable model name (used for output subdirectory).
            model_path: Path to the serialised model artifact (.pkl / .joblib).
            dataset_path: Path to engineered_dataset.csv (or test.csv).
            target_column: Name of the prediction target column.
            max_shap_samples: Cap on rows passed to SHAP (performance guard).

        Returns:
            Path to the model's SHAP output directory, or None on failure.
        """
        model_output_dir = self.shap_output_dir / model_name
        csv_dir = model_output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        log_path = model_output_dir / "shap_execution.log"
        execution_logger = ExecutionLogger(
            session_id=f"{self.session_id}_{model_name}",
            log_file_path=log_path,
        )

        session_ctx = SessionContext(
            session_id=self.session_id,
            supplied_model_name=model_name,
            pickle_file_path=str(model_path),
            engineered_dataset_path=str(dataset_path),
            target_column_name=target_column,
        )

        loaded_model = ModelLoader(execution_logger).load(model_path)
        loaded_dataset = DatasetLoader(execution_logger).load(dataset_path)

        schema_result = SchemaValidator(
            execution_logger,
            target_column_candidates=(target_column,),
        ).validate(
            loaded_dataset=loaded_dataset,
            loaded_model=loaded_model,
            session_context=session_ctx,
        )

        # Apply SHAP sample cap to avoid OOM on large datasets (DESIGN_PLAN R14)
        feature_df = schema_result.feature_dataframe
        if len(feature_df) > max_shap_samples:
            feature_df = feature_df.sample(n=max_shap_samples, random_state=42)
            logger.info(
                "=> SHAP: sampled %d/%d rows for model=%s",
                max_shap_samples, len(schema_result.feature_dataframe), model_name,
            )

        built_explainer = ExplainerFactory(execution_logger).build(
            model_family=loaded_model.model_family,
            model_object=loaded_model.model_object,
            feature_dataframe=feature_df,
            session_context=session_ctx,
        )

        shap_result = SHAPService(execution_logger).compute(
            built_explainer=built_explainer,
            feature_dataframe=feature_df,
            feature_names=schema_result.feature_names,
            model_object=loaded_model.model_object,
            detected_class_name=loaded_model.detected_class_name,
            session_context=session_ctx,
        )

        GlobalImportanceExporter(execution_logger).export(
            global_importance_dataframe=shap_result.global_importance_dataframe,
            output_path=csv_dir / "global_feature_importance.csv",
        )
        FeatureSHAPMappingExporter(execution_logger).export(
            mapping_dataframe=shap_result.mapping_dataframe,
            output_path=csv_dir / "feature_shap_mapping.csv",
        )
        logger.info("=> SHAP complete for model=%s dir=%s", model_name, model_output_dir)
        return model_output_dir
