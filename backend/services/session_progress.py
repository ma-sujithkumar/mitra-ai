"""Per-session pipeline progress derived from on-disk artifacts.

Lets the app resume an existing session from its last checkpoint: phases whose
artifacts already exist are reported complete so their agents are not re-run, and
``first_incomplete_phase`` tells the UI where to land. The ordered phase ->
artifact map comes from config (``[pipeline_phases]``) so nothing is hardcoded.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.config_loader import ConfigLoader

# Phase whose completion is additionally gated on the validation report's
# "passed" flag (presence alone is not enough for validation).
VALIDATION_PHASE = "validation"

# Phase status values.
STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"

# A phase counts as "done" (no re-run needed) when it reaches one of these.
DONE_STATUSES = {STATUS_COMPLETE, STATUS_PASSED}


class SessionProgress:
    """Computes pipeline phase completion for a single session."""

    def __init__(
        self,
        *,
        session_dir: Path,
        config_loader: ConfigLoader,
    ) -> None:
        self.session_dir = session_dir
        self.phase_artifacts = config_loader.pipeline_phases.phase_artifacts

    def _all_artifacts_exist(self, artifacts: list[str]) -> bool:
        return all((self.session_dir / artifact).is_file() for artifact in artifacts)

    def _validation_passed(self, artifacts: list[str]) -> bool:
        # Validation has a single report artifact; treat unreadable/malformed
        # reports as not passing rather than raising.
        report_path = self.session_dir / artifacts[0]
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return report.get("passed") is True

    def _status_for_phase(self, phase_name: str, artifacts: list[str]) -> str:
        if not self._all_artifacts_exist(artifacts):
            return STATUS_PENDING
        if phase_name == VALIDATION_PHASE:
            return STATUS_PASSED if self._validation_passed(artifacts) else STATUS_FAILED
        return STATUS_COMPLETE

    def phase_status(self) -> dict[str, str]:
        """Return {phase_name: status} in pipeline order."""
        return {
            phase_name: self._status_for_phase(phase_name, artifacts)
            for phase_name, artifacts in self.phase_artifacts.items()
        }

    def first_incomplete_phase(self) -> str | None:
        """First phase (in order) not yet done, or None when all are done."""
        for phase_name, status in self.phase_status().items():
            if status not in DONE_STATUSES:
                return phase_name
        return None
