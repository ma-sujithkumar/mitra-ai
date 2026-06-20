from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi.responses import FileResponse

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_session_manager
from backend.session import SessionManager


router = APIRouter(prefix="/api/runs", tags=["evaluation"])

# Artifact file names written by the orchestration pipeline. Kept as module
# constants (mirrors the path segments already hardcoded in runs.py) so the
# router resolves them in one place rather than scattering literals.
JUDGE_DECISION_FILENAME = "judge_decision.json"
TRAINING_SUMMARY_FILENAME = "training_summary.json"
TOKEN_USAGE_FILENAME = "token_usage.json"
SHAP_IMPORTANCE_FILENAME = "global_feature_importance.csv"

# Directories scanned (recursively) when listing on-demand plots for a session.
PLOT_SEARCH_SUBDIRS = ("plots", "evaluation")
PLOT_FILE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg")


class EvaluationArtifactReader:
    """Reads evaluation artifacts from a single session directory.

    Resolves the canonical session layout::

        <session_dir>/reports/judge_decision.json
        <session_dir>/reports/training_summary.json
        <session_dir>/evaluation/shap/<model_name>/csv/global_feature_importance.csv
        <session_dir>/plots/<stage>/<plot>.png
        <session_dir>/token_usage.json
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.reports_dir = session_dir / "reports"
        self.evaluation_dir = session_dir / "evaluation"

    def judge_decision(self) -> dict[str, Any] | None:
        return self._read_json_or_none(self.reports_dir / JUDGE_DECISION_FILENAME)

    def training_summary(self) -> dict[str, Any] | None:
        return self._read_json_or_none(self.reports_dir / TRAINING_SUMMARY_FILENAME)

    def token_usage(self) -> dict[str, Any] | None:
        return self._read_json_or_none(self.session_dir / TOKEN_USAGE_FILENAME)

    def build_leaderboard(self) -> dict[str, Any]:
        """Merge judge ranking with training metrics into one leaderboard list."""
        judge_decision = self.judge_decision()
        training_summary = self.training_summary()

        if judge_decision is None and training_summary is None:
            return {"status": "pending", "selected_model": None, "models": []}

        # Index training metrics by model name for an O(1) merge.
        metrics_by_model = self._index_training_metrics(training_summary)
        selected_model = (judge_decision or {}).get("selected_model")

        leaderboard_rows: list[dict[str, Any]] = []
        ranked_models = (judge_decision or {}).get("ranked_models") or []
        for ranked_model in ranked_models:
            model_name = ranked_model.get("model_name")
            training_record = metrics_by_model.get(model_name, {})
            leaderboard_rows.append({
                "rank": ranked_model.get("rank"),
                "model_name": model_name,
                "score": ranked_model.get("score"),
                "verdict": ranked_model.get("verdict"),
                "reasons": ranked_model.get("reasons", []),
                "validation_score": training_record.get("validation_score"),
                "metrics": training_record.get("metrics", {}),
                "winner": model_name == selected_model,
            })

        # If the judge has not run yet, fall back to a metrics-only leaderboard
        # so the UI can still render trained-model results.
        if not leaderboard_rows and metrics_by_model:
            leaderboard_rows = self._leaderboard_from_metrics_only(metrics_by_model)

        status = "complete" if judge_decision is not None else "training_only"
        return {
            "status": status,
            "selected_model": selected_model,
            "models": leaderboard_rows,
        }

    def verdict(self) -> dict[str, Any]:
        """Return the raw judge decision (selected model + reasoning trace)."""
        judge_decision = self.judge_decision()
        if judge_decision is None:
            return {"status": "pending", "selected_model": None}
        judge_decision.setdefault("status", "complete")
        return judge_decision

    def shap_importance(self, model_name: str | None) -> dict[str, Any]:
        """Return SHAP global feature importance for a model.

        When ``model_name`` is omitted the selected (winning) model is used,
        falling back to the first model with a SHAP CSV on disk.
        """
        resolved_model = model_name or self._default_shap_model()
        if resolved_model is None:
            return {"status": "pending", "model_name": None, "features": []}

        csv_path = (
            self.evaluation_dir / "shap" / resolved_model / "csv" / SHAP_IMPORTANCE_FILENAME
        )
        if not csv_path.is_file():
            return {"status": "pending", "model_name": resolved_model, "features": []}

        features = self._read_shap_csv(csv_path)
        return {
            "status": "complete",
            "model_name": resolved_model,
            "features": features,
        }

    def list_plots(self) -> dict[str, Any]:
        """List every plot image under the session, as web-relative paths."""
        plots: list[dict[str, str]] = []
        for search_subdir in PLOT_SEARCH_SUBDIRS:
            search_root = self.session_dir / search_subdir
            if not search_root.is_dir():
                continue
            for plot_path in sorted(search_root.rglob("*")):
                if plot_path.suffix.lower() not in PLOT_FILE_SUFFIXES:
                    continue
                relative_path = plot_path.relative_to(self.session_dir).as_posix()
                plots.append({
                    "name": plot_path.stem,
                    "stage": plot_path.parent.relative_to(self.session_dir).as_posix(),
                    "path": relative_path,
                })
        return {"plots": plots}

    def resolve_plot(self, plot_relative_path: str) -> Path:
        """Resolve a plot path safely, rejecting traversal outside the session."""
        candidate = (self.session_dir / plot_relative_path).resolve()
        session_root = self.session_dir.resolve()
        # Reject any path that escapes the session directory (path traversal).
        if session_root not in candidate.parents and candidate != session_root:
            raise HTTPException(status_code=400, detail="Invalid plot path.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Plot not found.")
        if candidate.suffix.lower() not in PLOT_FILE_SUFFIXES:
            raise HTTPException(status_code=400, detail="Unsupported plot type.")
        return candidate

    def _default_shap_model(self) -> str | None:
        judge_decision = self.judge_decision()
        if judge_decision and judge_decision.get("selected_model"):
            return judge_decision["selected_model"]
        shap_root = self.evaluation_dir / "shap"
        if not shap_root.is_dir():
            return None
        for model_dir in sorted(shap_root.iterdir()):
            if (model_dir / "csv" / SHAP_IMPORTANCE_FILENAME).is_file():
                return model_dir.name
        return None

    @staticmethod
    def _index_training_metrics(
        training_summary: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if not training_summary:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for model_item in training_summary.get("models", []):
            if model_item.get("status") != "completed":
                continue
            index[model_item.get("model_name")] = {
                "validation_score": model_item.get("validation_score"),
                "metrics": model_item.get("metrics", {}),
            }
        return index

    @staticmethod
    def _leaderboard_from_metrics_only(
        metrics_by_model: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            metrics_by_model.items(),
            key=lambda item: item[1].get("validation_score") or float("-inf"),
            reverse=True,
        )
        return [
            {
                "rank": rank_index + 1,
                "model_name": model_name,
                "score": training_record.get("validation_score"),
                "verdict": "pending",
                "reasons": [],
                "validation_score": training_record.get("validation_score"),
                "metrics": training_record.get("metrics", {}),
                "winner": False,
            }
            for rank_index, (model_name, training_record) in enumerate(ranked)
        ]

    @staticmethod
    def _read_shap_csv(csv_path: Path) -> list[dict[str, Any]]:
        features: list[dict[str, Any]] = []
        with csv_path.open(encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                features.append({
                    "feature": row.get("feature_name"),
                    "importance": float(row.get("mean_absolute_shap_value", 0.0) or 0.0),
                })
        return features

    @staticmethod
    def _read_json_or_none(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _build_reader(
    session_id: str,
    session_manager: SessionManager,
) -> EvaluationArtifactReader:
    session_dir = session_manager.get_session_path(session_id=session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    return EvaluationArtifactReader(session_dir=session_dir)


@router.get("/{session_id}/leaderboard")
def get_leaderboard(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.build_leaderboard()}


@router.get("/{session_id}/verdict")
def get_verdict(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.verdict()}


@router.get("/{session_id}/shap")
def get_shap(
    session_id: str,
    model_name: str | None = Query(default=None),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.shap_importance(model_name=model_name)}


@router.get("/{session_id}/tokens")
def get_tokens(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    token_usage = reader.token_usage()
    if token_usage is None:
        return {"session_id": session_id, "status": "pending", "agents": {}}
    return {"session_id": session_id, "status": "complete", **token_usage}


@router.get("/{session_id}/plots")
def list_plots(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.list_plots()}


@router.get("/{session_id}/plots/{plot_path:path}")
def get_plot(
    session_id: str,
    plot_path: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileResponse:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    resolved_plot = reader.resolve_plot(plot_relative_path=plot_path)
    return FileResponse(path=resolved_plot)
