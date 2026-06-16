"""Session-scoped execution logger for the SHAP explainability pipeline.

Implements spec.md Section 19: one execution.log file per session, with every
mandated event category exposed as a named, declarative method so call sites never
construct ad hoc log messages with inconsistent shapes.
"""

import logging
from enum import Enum
from pathlib import Path


class ExecutionEvent(Enum):
    """Sec 19 logging event categories."""

    EXECUTION_START = "execution_start"
    DATASET_VALIDATION = "dataset_validation"
    MODEL_VALIDATION = "model_validation"
    MODEL_LOADING = "model_loading"
    MODEL_TYPE_DETECTION = "model_type_detection"
    MODEL_NAME_VALIDATION = "model_name_validation"
    SCHEMA_VALIDATION = "schema_validation"
    EXPLAINER_SELECTION = "explainer_selection"
    SHAP_GENERATION = "shap_generation"
    PLOT_GENERATION = "plot_generation"
    CSV_EXPORT = "csv_export"
    METADATA_GENERATION = "metadata_generation"
    EXECUTION_COMPLETION = "execution_completion"
    EXECUTION_FAILURE = "execution_failure"


_SUPPORTED_LOG_LEVEL_NAMES: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
_LOGGER_NAME_PREFIX: str = "shap_explainability.session"
_LOG_RECORD_FORMAT: str = "%(asctime)s | %(levelname)s | %(event)s | %(message)s"


class ExecutionLogger:
    """Writes one execution.log file per session_id with declarative event methods.

    Each Sec 19 event category has its own named method (for example
    log_model_loading) so calling code reads as a direct statement of which stage
    produced the log line, instead of free-text logger.info(...) calls scattered
    across the pipeline with inconsistent event naming.
    """

    def __init__(self, session_id: str, log_file_path: Path, log_level: str = "INFO") -> None:
        """Initializes a session-scoped logger writing to a single log file.

        Args:
            session_id: Unique execution identifier (Sec 4.1), used to scope the
                underlying logging.Logger instance so concurrent sessions in the
                same process do not share handlers.
            log_file_path: Absolute path of the execution.log file to write (Sec
                21), typically obtained from OutputManager.log_path().
            log_level: One of DEBUG, INFO, WARNING, ERROR (CFG-02). Defaults to INFO.

        Raises:
            ValueError: If log_level is not one of the supported level names.
            OSError: If the log file's parent directory does not exist and cannot
                be created.
        """
        if log_level not in _SUPPORTED_LOG_LEVEL_NAMES:
            raise ValueError(
                f"log_level '{log_level}' is not one of {_SUPPORTED_LOG_LEVEL_NAMES}."
            )

        self.session_id: str = session_id
        self.log_file_path: Path = log_file_path
        self.log_level: str = log_level
        self._logger: logging.Logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        """Creates (or reuses) the underlying logging.Logger with a file handler.

        Reuses the existing handler when an ExecutionLogger for the same session_id
        is constructed more than once in the same process, so log lines are never
        duplicated.
        """
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger_name = f"{_LOGGER_NAME_PREFIX}.{self.session_id}"
        session_logger = logging.getLogger(logger_name)
        session_logger.setLevel(getattr(logging, self.log_level))
        session_logger.propagate = False

        if not session_logger.handlers:
            file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_LOG_RECORD_FORMAT))
            session_logger.addHandler(file_handler)

        return session_logger

    def _log_event(self, event: ExecutionEvent, message: str, level: int = logging.INFO) -> None:
        """Writes one log record tagged with the given Sec 19 event category."""
        self._logger.log(level, message, extra={"event": event.value})

    def log_execution_start(self, message: str = "Pipeline execution started.") -> None:
        """Logs the execution-start event."""
        self._log_event(ExecutionEvent.EXECUTION_START, message)

    def log_dataset_validation(self, message: str, level: int = logging.INFO) -> None:
        """Logs the dataset-validation event."""
        self._log_event(ExecutionEvent.DATASET_VALIDATION, message, level)

    def log_model_validation(self, message: str, level: int = logging.INFO) -> None:
        """Logs the model-validation event."""
        self._log_event(ExecutionEvent.MODEL_VALIDATION, message, level)

    def log_model_loading(self, message: str, level: int = logging.INFO) -> None:
        """Logs the model-loading event."""
        self._log_event(ExecutionEvent.MODEL_LOADING, message, level)

    def log_model_type_detection(self, message: str, level: int = logging.INFO) -> None:
        """Logs the model-type-detection event."""
        self._log_event(ExecutionEvent.MODEL_TYPE_DETECTION, message, level)

    def log_model_name_validation(self, message: str, level: int = logging.INFO) -> None:
        """Logs the model-name-validation event.

        Sec 8 Rule 2 (supplied model_name differs from detected type) is a
        non-terminating warning: callers should pass level=logging.WARNING for that
        case while still allowing the pipeline to continue.
        """
        self._log_event(ExecutionEvent.MODEL_NAME_VALIDATION, message, level)

    def log_schema_validation(self, message: str, level: int = logging.INFO) -> None:
        """Logs the schema-validation event."""
        self._log_event(ExecutionEvent.SCHEMA_VALIDATION, message, level)

    def log_explainer_selection(self, message: str) -> None:
        """Logs the explainer-selection event."""
        self._log_event(ExecutionEvent.EXPLAINER_SELECTION, message)

    def log_shap_generation(self, message: str) -> None:
        """Logs the SHAP-generation event."""
        self._log_event(ExecutionEvent.SHAP_GENERATION, message)

    def log_plot_generation(self, message: str) -> None:
        """Logs the plot-generation event."""
        self._log_event(ExecutionEvent.PLOT_GENERATION, message)

    def log_csv_export(self, message: str) -> None:
        """Logs the CSV-export event."""
        self._log_event(ExecutionEvent.CSV_EXPORT, message)

    def log_metadata_generation(self, message: str) -> None:
        """Logs the metadata-generation event."""
        self._log_event(ExecutionEvent.METADATA_GENERATION, message)

    def log_execution_completion(self, message: str = "Pipeline execution completed.") -> None:
        """Logs the execution-completion event."""
        self._log_event(ExecutionEvent.EXECUTION_COMPLETION, message)

    def log_execution_failure(self, message: str) -> None:
        """Logs the execution-failure event at ERROR level (Sec 20)."""
        self._log_event(ExecutionEvent.EXECUTION_FAILURE, message, logging.ERROR)
