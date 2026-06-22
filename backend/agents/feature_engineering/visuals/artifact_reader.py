"""Reads all post-run pipeline artifacts into typed properties for visualizers."""
from __future__ import annotations

import json
import re
from pathlib import Path


RAW_LOG_BLOCK_RE = re.compile(
    r"===== caller=(\S+) attempt=(\S+) status=(\S+) failures=(\[.*?\]) =====\s*(.*?)(?====== caller=|$)",
    re.DOTALL,
)
STEP_LOG_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\] (\w[\w_]*) (ok|error) \(([0-9.]+)s\)(?: llm=(\S+))?"
)


class ArtifactReader:
    """Loads all pipeline output artifacts and exposes them as typed properties."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._artifact: dict = self._load_json("feature_artifact.json")
        self._profile: dict = self._load_json("profile.json")

        stats_dir_str = self._artifact.get("stats_dir")
        if stats_dir_str and Path(stats_dir_str).exists():
            self._stats_dir = Path(stats_dir_str)
        else:
            run_id = self._artifact.get("run_id", "")
            self._stats_dir = Path(".mitra") / run_id / "stats"

        self._mi_data: dict = self._load_stats_json("mutual_info.json")
        self._rf_data: dict = self._load_stats_json("rf_importance.json")
        self._mrmr_data: dict = self._load_stats_json("mrmr_ranking.json")
        self._variance_data: dict = self._load_stats_json("variance.json")
        self._pearson_data: dict = self._load_stats_json("correlation_pearson.json")
        self._clusters_data: dict = self._load_stats_json("clusters.json")
        self._baseline_data: dict = self._load_stats_json("linear_baseline.json")
        self._pca_data_raw: dict = self._load_stats_json("pca.json")

        self._llm_selection: dict = self._parse_raw_responses()
        self._timeline_events_raw: list[dict] = self._parse_execution_log()

    # --- private loaders ---

    def _load_json(self, filename: str) -> dict:
        path = self._output_dir / filename
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_stats_json(self, filename: str) -> dict:
        path = self._stats_dir / filename
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _parse_raw_responses(self) -> dict:
        """Return the last successful FeatureSelector LLM response payload."""
        raw_log_path = self._output_dir / "raw_responses.txt"
        if not raw_log_path.exists():
            return {}
        try:
            content = raw_log_path.read_text(encoding="utf-8")
        except Exception:
            return {}

        best_payload: dict = {}
        for match in RAW_LOG_BLOCK_RE.finditer(content):
            caller = match.group(1)
            status = match.group(3)
            body = match.group(5).strip()
            if caller != "FeatureSelector":
                continue
            if status not in ("ok", "ok:revised"):
                continue
            parsed = self._extract_json(body)
            if parsed and isinstance(parsed, dict):
                best_payload = parsed

        return best_payload

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except Exception:
                pass
        for pattern in (r"(\{.*?\})", r"(\{.*\})"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    continue
        try:
            return json.loads(text)
        except Exception:
            return None

    def _parse_execution_log(self) -> list[dict]:
        log_path = self._output_dir / "execution_log.txt"
        if not log_path.exists():
            return []
        events: list[dict] = []
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                match = STEP_LOG_RE.search(line)
                if match:
                    events.append({
                        "step": match.group(2),
                        "timestamp": match.group(1),
                        "elapsed_s": float(match.group(4)),
                        "status": match.group(3),
                        "llm_source": match.group(5) or "",
                    })
        except Exception:
            pass
        return events

    # --- public typed properties ---

    @property
    def run_id(self) -> str:
        return self._artifact.get("run_id", "unknown")

    @property
    def task(self) -> str:
        return self._artifact.get("task", "unknown")

    @property
    def target_column(self) -> str:
        return self._artifact.get("target_column", "")

    @property
    def dropped_columns(self) -> list[str]:
        return list(self._artifact.get("dropped_columns", []))

    @property
    def created_columns(self) -> list[dict]:
        return list(self._artifact.get("created_columns", []))

    @property
    def transformers(self) -> list[dict]:
        return list(self._artifact.get("transformers", []))

    @property
    def selected_columns(self) -> list[str]:
        return list(self._artifact.get("selected_columns", []))

    @property
    def selection_method(self) -> str:
        return self._artifact.get("selection_method") or "unknown"

    @property
    def warnings(self) -> list[str]:
        return list(self._artifact.get("warnings", []))

    @property
    def profile(self) -> dict[str, dict]:
        return dict(self._profile)

    @property
    def mi_scores(self) -> dict[str, float]:
        return dict(self._mi_data.get("scores", {}))

    @property
    def rf_scores(self) -> dict[str, float]:
        return dict(self._rf_data.get("scores", {}))

    @property
    def mrmr_ranked(self) -> list[str]:
        return list(self._mrmr_data.get("ranked", []))

    @property
    def low_variance_columns(self) -> list[str]:
        return list(self._variance_data.get("low_variance", []))

    @property
    def pearson_pairs(self) -> list[list]:
        return list(self._pearson_data.get("high_pairs", []))

    @property
    def clusters(self) -> dict[str, list[str]]:
        return dict(self._clusters_data)

    @property
    def pca_data(self) -> dict:
        return dict(self._pca_data_raw)

    @property
    def linear_baseline(self) -> dict:
        return dict(self._baseline_data)

    @property
    def selection_rationale(self) -> str:
        return str(self._llm_selection.get("rationale", ""))

    @property
    def timeline_events(self) -> list[dict]:
        return list(self._timeline_events_raw)
