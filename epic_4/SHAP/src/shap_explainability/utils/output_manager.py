"""Resolves and creates the per-session output directory tree.

Implements spec.md Section 5 (output configuration) and the output-artifact layout
defined in architecture.md Section 1.2. OutputManager is the single place that
constructs output paths; every plot, CSV, metadata, and log path used elsewhere in
the pipeline must be obtained from this class so the on-disk layout can change in
one place without touching exporters or visualizations.
"""

from pathlib import Path


class OutputManagerError(Exception):
    """Raised when the configured output directories cannot be created."""


_PLOTS_SUBDIRECTORY_NAME: str = "plots"
_CSV_SUBDIRECTORY_NAME: str = "csv"
_METADATA_SUBDIRECTORY_NAME: str = "metadata"
_LOGS_SUBDIRECTORY_NAME: str = "logs"
_METADATA_FILENAME: str = "metadata.json"
_EXECUTION_LOG_FILENAME: str = "execution.log"


class OutputManager:
    """Owns the <output_root>/<session_id>/ folder tree and exposes path getters."""

    def __init__(self, output_root: Path, session_id: str) -> None:
        """Initializes the manager for one session's output tree.

        Args:
            output_root: Configured root directory under which all session folders
                are created (CFG-01). Not supplied through the integration payload
                (Sec 5).
            session_id: Unique execution identifier (Sec 4.1) used as the
                session-specific subfolder name.
        """
        self.output_root: Path = Path(output_root)
        self.session_id: str = session_id
        self.session_directory: Path = self.output_root / self.session_id

    def initialize(self) -> None:
        """Creates the session directory and all artifact subfolders if missing.

        Equivalent to mkdir -p for every directory in the session's output tree;
        safe to call multiple times.

        Raises:
            OutputManagerError: If any directory cannot be created, for example due
                to insufficient permissions or because a path component already
                exists as a regular file.
        """
        for directory_path in self._all_session_directories():
            try:
                directory_path.mkdir(parents=True, exist_ok=True)
            except OSError as os_error:
                raise OutputManagerError(
                    f"Could not create output directory '{directory_path}': {os_error}"
                ) from os_error

    def _all_session_directories(self) -> tuple[Path, ...]:
        """Returns every directory that must exist under the session folder."""
        return (
            self.session_directory,
            self.session_directory / _PLOTS_SUBDIRECTORY_NAME,
            self.session_directory / _CSV_SUBDIRECTORY_NAME,
            self.session_directory / _METADATA_SUBDIRECTORY_NAME,
            self.session_directory / _LOGS_SUBDIRECTORY_NAME,
        )

    def plot_path(self, plot_filename: str) -> Path:
        """Returns the full path for a plot artifact (e.g. summary_plot.png)."""
        return self.session_directory / _PLOTS_SUBDIRECTORY_NAME / plot_filename

    def csv_path(self, csv_filename: str) -> Path:
        """Returns the full path for a CSV artifact (e.g. global_feature_importance.csv)."""
        return self.session_directory / _CSV_SUBDIRECTORY_NAME / csv_filename

    def metadata_path(self) -> Path:
        """Returns the full path for metadata.json."""
        return self.session_directory / _METADATA_SUBDIRECTORY_NAME / _METADATA_FILENAME

    def log_path(self) -> Path:
        """Returns the full path for execution.log."""
        return self.session_directory / _LOGS_SUBDIRECTORY_NAME / _EXECUTION_LOG_FILENAME
