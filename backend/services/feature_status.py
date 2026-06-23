"""Feature engineering run-status reader.

Reads the structured feature_run.json written by PipelinePrep after
FeatureEngineerOrchestrator.run() completes, and exposes the data in a
shape the evaluation router can return directly to the frontend.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Canonical ordered step list mirroring FeatureEngineerOrchestrator._run_pipeline.
# Each entry: (name, display_label, agent_type)
CANONICAL_STEPS: list[tuple[str, str, str]] = [
    ("profile_data",         "Profile dataset",            "rule"),
    ("infer_types",          "Infer semantic types",       "rule"),
    ("handle_missing",       "Handle missing values",      "rule"),
    ("handle_outliers",      "Handle outliers",            "rule"),
    ("encode_features",      "Encode features",            "rule"),
    ("create_features",      "Create derived features",    "rule"),
    ("scale_features",       "Scale features",             "rule"),
    ("compute_feature_stats","Compute feature statistics", "rule"),
    ("select_features",      "Select features (LLM)",      "llm"),
    ("validate_features",    "Validate feature set",       "rule"),
    ("write_report",         "Write feature report",       "rule"),
]

# Map step name to canonical index for fast lookup.
STEP_INDEX: dict[str, int] = {name: idx for idx, (name, _, _) in enumerate(CANONICAL_STEPS)}

# Regex to parse a line from execution_log.txt.
# Format: [YYYY-MM-DDTHH:MM:SS] <step_name> <status> (<elapsed>s)[ llm=<source>][ <detail>]
_LOG_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+(?P<name>\S+)\s+(?P<status>ok|error)\s+"
    r"\((?P<elapsed>[0-9.]+)s\)(?:\s+llm=(?P<llm_source>\S+))?(?:\s+(?P<detail>.+))?$"
)


class FeatureEngineeringStatusReader:
    """Reads feature_run.json (or raw execution_log.txt as fallback) for one session."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.fe_dir = session_dir / "reports" / "feature_engineering"

    def read(self) -> dict[str, Any]:
        """Return the full status payload for the /feature-engineering endpoint."""
        run_status = self._read_json_or_none(self.fe_dir / "feature_run.json")
        if run_status is not None:
            return run_status

        # Fallback: synthesise from raw artifacts if feature_run.json isn't written yet.
        steps = self._parse_execution_log()
        artifact = self._read_json_or_none(self.fe_dir / "feature_artifact.json") or {}
        reasoning = self._extract_reasoning(artifact)
        agents = self._derive_agents(steps)
        overall_status = self._overall_status(steps)
        return {
            "status": overall_status,
            "steps": steps,
            "agents": agents,
            "summary": self._build_summary(artifact),
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_execution_log(self) -> list[dict[str, Any]]:
        log_path = self.fe_dir / "execution_log.txt"
        completed: dict[str, dict[str, Any]] = {}
        if log_path.is_file():
            for raw_line in log_path.read_text(encoding="utf-8").splitlines():
                match = _LOG_LINE_RE.match(raw_line.strip())
                if match:
                    completed[match.group("name")] = {
                        "status": match.group("status"),
                        "elapsed_sec": float(match.group("elapsed")),
                        "llm_source": match.group("llm_source"),
                        "detail": match.group("detail") or "",
                    }

        steps: list[dict[str, Any]] = []
        for step_name, step_label, agent_type in CANONICAL_STEPS:
            if step_name in completed:
                info = completed[step_name]
                steps.append({
                    "name": step_name,
                    "label": step_label,
                    "agent_type": agent_type,
                    "status": info["status"],
                    "elapsed_sec": info["elapsed_sec"],
                    "llm_source": info["llm_source"],
                    "detail": info["detail"],
                })
            else:
                steps.append({
                    "name": step_name,
                    "label": step_label,
                    "agent_type": agent_type,
                    "status": "pending",
                    "elapsed_sec": None,
                    "llm_source": None,
                    "detail": "",
                })
        return steps

    def _extract_reasoning(self, artifact: dict[str, Any]) -> dict[str, Any]:
        """Extract FeatureSelector LLM rationale from raw_responses.txt."""
        llm_reasoning: str | None = None
        raw_path = self.fe_dir / "raw_responses.txt"
        if raw_path.is_file():
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
            # raw_responses.txt contains multiple JSON blobs separated by delimiters.
            # Find any JSON object with a "rationale" key.
            for json_match in re.finditer(r'\{[^{}]*"rationale"\s*:\s*"([^"]+)"[^{}]*\}', raw_text, re.DOTALL):
                llm_reasoning = json_match.group(1).strip()
                break
        return {
            "llm_reasoning": llm_reasoning,
            "selection_method": artifact.get("selection_method"),
        }

    @staticmethod
    def _build_summary(artifact: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": artifact.get("task"),
            "target_column": artifact.get("target_column"),
            "dropped_columns": artifact.get("dropped_columns", []),
            "created_columns": artifact.get("created_columns", []),
            "selected_columns": artifact.get("selected_columns", []),
            "selection_method": artifact.get("selection_method"),
            "warnings": artifact.get("warnings", []),
        }

    @staticmethod
    def _derive_agents(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        select_step = next((step for step in steps if step["name"] == "select_features"), None)
        feature_state = "pending"
        judge_state = "pending"
        if select_step:
            if select_step["status"] == "ok":
                feature_state = "done"
                llm_source = select_step.get("llm_source") or ""
                judge_state = "done" if "judge" in llm_source else "pending"
            elif select_step["status"] == "error":
                feature_state = "error"
        return [
            {"id": "feature", "state": feature_state},
            {"id": "judge",   "state": judge_state},
        ]

    @staticmethod
    def _overall_status(steps: list[dict[str, Any]]) -> str:
        statuses = {step["status"] for step in steps}
        if "pending" in statuses:
            return "running" if "ok" in statuses or "error" in statuses else "pending"
        if "error" in statuses:
            return "partial_failure"
        return "done"

    @staticmethod
    def _read_json_or_none(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
