from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Per-session activity log filename and the shared ASCII log format. Kept ASCII
# only (no unicode) per project conventions.
ACTIVITY_LOG_FILENAME = "activity.log"
LOG_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DEFAULT_LEVEL = "INFO"

# A single space separates the four fixed fields; the message is the remainder.
# isoformat timestamps and level/stage tokens contain no spaces, so a 4-way
# split reliably recovers the structured fields when reading the log back.
_LOG_FIELD_COUNT = 4


class ActivityLog:
    """Appends human-readable activity lines to a per-session ``activity.log``
    and mirrors them to the shared ``mitra`` logger (so they also land in the
    global rotating ``mitra.log``). Used to give the UI a live, downloadable
    record of every action and backend event for a run."""

    def __init__(self, session_path: Path) -> None:
        self.session_path = session_path
        self.log_path = session_path / ACTIVITY_LOG_FILENAME
        self._logger = logging.getLogger("mitra.activity")

    def record(
        self,
        stage: str,
        message: str,
        level: str = DEFAULT_LEVEL,
    ) -> dict[str, str]:
        normalized_level = level.upper()
        timestamp = datetime.now().isoformat(timespec="seconds")
        log_line = f"{timestamp} {normalized_level} {stage} {message}"
        # mkdir -p the session dir before writing in case it was not created yet.
        self.session_path.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(log_line + "\n")
        self._logger.log(
            logging.getLevelName(normalized_level)
            if isinstance(logging.getLevelName(normalized_level), int)
            else logging.INFO,
            "[%s] %s => %s",
            self.session_path.name,
            stage,
            message,
        )
        return {
            "timestamp": timestamp,
            "level": normalized_level,
            "stage": stage,
            "message": message,
        }

    def read(self) -> list[dict[str, str]]:
        if not self.log_path.is_file():
            return []
        entries: list[dict[str, str]] = []
        for raw_line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            parts = raw_line.split(" ", _LOG_FIELD_COUNT - 1)
            if len(parts) < _LOG_FIELD_COUNT:
                # Tolerate any legacy/free-form line so reading never crashes.
                entries.append(
                    {
                        "timestamp": "",
                        "level": DEFAULT_LEVEL,
                        "stage": "",
                        "message": raw_line,
                    }
                )
                continue
            timestamp, level, stage, message = parts
            entries.append(
                {
                    "timestamp": timestamp,
                    "level": level,
                    "stage": stage,
                    "message": message,
                }
            )
        return entries


def configure_file_logging(
    log_file: Path,
    level: str,
    max_bytes: int,
    backup_count: int,
) -> None:
    """Attach a rotating file handler to the root logger so all backend
    activity is persisted to ``log_file`` (in addition to the console).

    The handler is attached to the root logger (not just the ``mitra``
    namespace) because every backend module logs via
    ``logging.getLogger(__name__)`` (e.g. ``backend.orchestration.judge_loop``,
    ``backend.services.training_service``), which propagates to root, not to
    ``mitra``. Attaching only to ``mitra`` silently dropped every INFO log
    from the training/judge/evaluation pipeline (root's default level is
    WARNING), making it impossible to measure per-turn timing after the
    initial feature-engineering stage.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    has_file_handler = any(
        isinstance(handler, RotatingFileHandler)
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_LINE_FORMAT))
        root_logger.addHandler(file_handler)
    resolved_level = logging.getLevelName(level.upper())
    root_logger.setLevel(
        resolved_level if isinstance(resolved_level, int) else logging.INFO
    )
