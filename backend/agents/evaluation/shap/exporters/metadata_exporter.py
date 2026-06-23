"""Writes metadata.json by assembling facts from SessionContext and SHAPResult.

Implements spec.md Section 18. Must be callable on both the success path
(shap_result fully populated) and early-failure paths (shap_result=None),
so all fields sourced from SHAPResult are guarded with an Optional check.
"""

import json
from pathlib import Path
from typing import Any, Optional

from backend.agents.evaluation.shap.errors import ExportError
from backend.agents.evaluation.shap.models.shap_result import SHAPResult
from backend.agents.evaluation.shap.session_context import SessionContext
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger

_METADATA_FILENAME: str = "metadata.json"


class MetadataExporter:
    """Assembles and writes metadata.json for a pipeline execution (spec.md Sec 18).

    Responsibilities:
      - Collect fields from SessionContext (session_id, model info, timestamps,
        warnings, error state, explainer name).
      - Collect prediction_type from SHAPResult when available.
      - Write the assembled dict to metadata.json at the specified path.
      - Handle None shap_result gracefully for early-failure invocations
        (architecture.md Section 5 failure-path requirement, AC-15).
      - Log the metadata-generation event via ExecutionLogger.
    """

    def __init__(self, execution_logger: ExecutionLogger) -> None:
        """Initializes the exporter with a session-scoped logger.

        Args:
            execution_logger: Session-scoped logger for recording the
                metadata_generation event (spec.md Sec 19).
        """
        self._execution_logger: ExecutionLogger = execution_logger

    def export(
        self,
        session_context: SessionContext,
        shap_result: Optional[SHAPResult],
        output_path: Path,
    ) -> Path:
        """Assemble and write metadata.json to the specified path.

        Args:
            session_context: Mutable pipeline state after all completed stages.
                All SessionContext fields available at call time are included in
                the metadata output.
            shap_result: Completed SHAP result for reading prediction_type. Pass
                None when the pipeline failed before SHAP computation -- the
                resulting metadata.json will have prediction_type set to null.
            output_path: Full path where metadata.json will be written.
                Typically obtained from OutputManager.metadata_path().

        Returns:
            The resolved output_path that was written.

        Raises:
            ExportError: If metadata.json cannot be created or written.
        """
        self._execution_logger.log_metadata_generation(
            f"Assembling metadata for session '{session_context.session_id}' "
            f"=> {output_path}"
        )

        metadata_dict: dict[str, Any] = self._assemble_metadata(
            session_context, shap_result
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata_dict, metadata_file, indent=2, default=str)
        except OSError as io_error:
            raise ExportError(
                f"Failed to write metadata JSON to '{output_path}': {io_error}"
            ) from io_error

        self._execution_logger.log_metadata_generation(
            f"Metadata JSON written: {output_path}"
        )
        return output_path

    @staticmethod
    def _assemble_metadata(
        session_context: SessionContext,
        shap_result: Optional[SHAPResult],
    ) -> dict[str, Any]:
        """Build the metadata dictionary from SessionContext and SHAPResult.

        Args:
            session_context: Pipeline state container.
            shap_result: SHAP computation result, or None if pipeline failed
                before that stage.

        Returns:
            Dictionary ready for json.dump(). All None values are preserved
            so downstream consumers see a consistent schema regardless of how
            far the pipeline progressed.
        """
        model_name_validation_status_value: Optional[str] = (
            session_context.model_name_validation_status.value
            if session_context.model_name_validation_status is not None
            else None
        )

        prediction_type: Optional[str] = (
            shap_result.prediction_type if shap_result is not None else None
        )

        return {
            "session_id": session_context.session_id,
            "provided_model_name": session_context.supplied_model_name,
            "detected_model_type": session_context.detected_model_type,
            "validation_status": session_context.execution_status.value,
            "model_name_validation_status": model_name_validation_status_value,
            "model_name_validation_message": session_context.model_name_validation_message,
            "explainer": session_context.explainer_name,
            "prediction_type": prediction_type,
            "num_samples": session_context.num_samples,
            "num_features": session_context.num_features,
            "execution_timestamp": session_context.created_at.isoformat(),
            "warnings": session_context.warnings,
            "error_message": session_context.error_message,
        }
