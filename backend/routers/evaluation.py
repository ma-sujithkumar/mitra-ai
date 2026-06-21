from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_session_manager
from backend.services.feature_status import FeatureEngineeringStatusReader
from backend.session import SessionManager
from backend.orchestration.plotting import PipelinePlotGenerator


router = APIRouter(prefix="/api/runs", tags=["evaluation"])

# Artifact file names written by the orchestration pipeline. Kept as module
# constants (mirrors the path segments already hardcoded in runs.py) so the
# router resolves them in one place rather than scattering literals.
JUDGE_DECISION_FILENAME = "judge_decision.json"
TRAINING_SUMMARY_FILENAME = "training_summary.json"
TOKEN_USAGE_FILENAME = "token_usage.json"
SHAP_IMPORTANCE_FILENAME = "global_feature_importance.csv"
DATASET_PRIOR_FILENAME = "dataset_prior.json"
VALIDATION_REPORT_FILENAME = "validation_report.json"
MODEL_CONFIG_FILENAME = "model_config.json"
OVERFITTING_ANALYSIS_FILENAME = "overfitting_analysis.json"

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

    def shap_status(self) -> dict[str, Any]:
        status_path = self.evaluation_dir / "shap_status.json"
        data = self._read_json_or_none(status_path)
        if data is None:
            return {"status": "pending", "progress": 0, "message": "Awaiting SHAP analysis..."}
        return data

    def overfitting_status(self) -> dict[str, Any]:
        status_path = self.evaluation_dir / "overfitting_status.json"
        data = self._read_json_or_none(status_path)
        if data is None:
            return {"status": "pending", "progress": 0, "message": "Awaiting Overfitting analysis..."}
        return data

    def judge_status(self) -> dict[str, Any]:
        status_path = self.reports_dir / "judge_status.json"
        data = self._read_json_or_none(status_path)
        if data is None:
            return {"status": "pending", "progress": 0, "message": "Awaiting Judge Agent..."}
        return data

    def build_leaderboard(self) -> dict[str, Any]:
        """Merge judge ranking with training metrics, overfitting, and HPT results."""
        judge_decision = self.judge_decision()
        training_summary = self.training_summary()

        if judge_decision is None and training_summary is None:
            return {"status": "pending", "selected_model": None, "models": [], "decision_trace": None}

        # Index training metrics, overfitting, and HPT results by model name for O(1) merge.
        metrics_by_model = self._index_training_metrics(training_summary)
        overfitting_by_model = self._index_overfitting()
        hpt_by_model = self._index_hpt_results()
        selected_model = (judge_decision or {}).get("selected_model")
        decision_trace = (judge_decision or {}).get("decision_trace")
        comparison_explanation = (judge_decision or {}).get("comparison_explanation")

        leaderboard_rows: list[dict[str, Any]] = []
        ranked_models = (judge_decision or {}).get("ranked_models") or []
        for ranked_model in ranked_models:
            model_name = ranked_model.get("model_name")
            training_record = metrics_by_model.get(model_name, {})
            hpt_record = hpt_by_model.get(model_name, {})
            leaderboard_rows.append({
                "rank": ranked_model.get("rank"),
                "model_name": model_name,
                "score": ranked_model.get("score"),
                "verdict": ranked_model.get("verdict"),
                "reasons": ranked_model.get("reasons", []),
                "llm_flags": ranked_model.get("llm_flags", []),
                # Governance-dashboard fields (structured Judge findings + decision).
                "decision": ranked_model.get("decision"),
                "findings": ranked_model.get("findings", []),
                "ranking_explanation": ranked_model.get("ranking_explanation"),
                "validation_score": training_record.get("validation_score"),
                "metrics": training_record.get("metrics", {}),
                "overfitting": overfitting_by_model.get(model_name),
                "winner": model_name == selected_model,
                # HPT fields: populated after on-demand tuning of the top-1 model
                "hpt_best_score": hpt_record.get("best_score"),
                "hpt_best_params": hpt_record.get("best_hyperparameters"),
                "hpt_n_trials": hpt_record.get("n_trials"),
                "hpt_primary_metric": hpt_record.get("primary_metric"),
            })

        # If the judge has not run yet, fall back to a metrics-only leaderboard
        # so the UI can still render trained-model results.
        if not leaderboard_rows and metrics_by_model:
            leaderboard_rows = self._leaderboard_from_metrics_only(metrics_by_model, overfitting_by_model)

        status = "complete" if judge_decision is not None else "training_only"
        return {
            "status": status,
            "selected_model": selected_model,
            "decision_trace": decision_trace,
            "comparison_explanation": comparison_explanation,
            "models": leaderboard_rows,
        }

    def verdict(self) -> dict[str, Any]:
        """Return the raw judge decision (selected model + reasoning trace)."""
        judge_decision = self.judge_decision()
        if judge_decision is None:
            return {"status": "pending", "selected_model": None}

        # If the judge is still running or pending, the verdict is not complete.
        status_data = self.judge_status()
        judge_status_str = status_data.get("status", "pending")
        if judge_status_str not in ("all_completed", "completed", "failed"):
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

    def feature_engineering_status(self) -> dict[str, Any]:
        """Return feature engineering step status and agent reasoning."""
        status_reader = FeatureEngineeringStatusReader(self.session_dir)
        return status_reader.read()

    def d2v_prior(self) -> dict[str, Any]:
        """Return Dataset2Vec warm-start prior if available."""
        prior = self._read_json_or_none(self.reports_dir / DATASET_PRIOR_FILENAME)
        if prior is None:
            return {"status": "pending"}
        return {"status": "complete", **prior}

    def validation_report(self) -> dict[str, Any]:
        """Return validation report (upload-time column checks)."""
        report = self._read_json_or_none(self.reports_dir / VALIDATION_REPORT_FILENAME)
        if report is None:
            return {"status": "pending"}
        return {"status": "complete", **report}

    def hpt_results(self) -> dict[str, Any]:
        """Return HPT / Optuna results if available."""
        hpt_path = self.evaluation_dir / "hpt" / "hpt_results.json"
        if not hpt_path.is_file():
            return {"status": "pending", "hpt_results": []}
        try:
            return {"status": "complete", **json.loads(hpt_path.read_text(encoding="utf-8"))}
        except Exception:
            return {"status": "pending", "hpt_results": []}

    def model_config(self) -> dict[str, Any]:
        """Return model config (selected families, task type, etc.).

        model_config.json can live at the session root (fallback artifact
        builder) or under reports/ (PipelinePrep / model selection agent), and
        its top-level shape is a list of model entries rather than a dict.
        """
        for candidate_path in (self.session_dir / MODEL_CONFIG_FILENAME, self.reports_dir / MODEL_CONFIG_FILENAME):
            if candidate_path.is_file():
                raw_data = json.loads(candidate_path.read_text(encoding="utf-8"))
                models = raw_data if isinstance(raw_data, list) else raw_data.get("models", [])
                return {"status": "complete", "models": models}
        return {"status": "pending", "models": []}

    def resolve_model_path(self, model_name: str) -> Path:
        """Resolve a trained model file path safely within the session directory."""
        training_summary = self.training_summary()
        if training_summary is None:
            raise HTTPException(status_code=404, detail="Training summary not found.")

        # Find the model entry and get its artifact path.
        model_path_str: str | None = None
        for model_item in training_summary.get("models", []):
            if model_item.get("model_name") == model_name:
                model_path_str = model_item.get("model_path")
                break

        if model_path_str is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in training summary.")

        candidate = Path(model_path_str)
        if not candidate.is_absolute():
            candidate = self.session_dir / candidate

        resolved = candidate.resolve()
        session_root = self.session_dir.resolve()
        if session_root not in resolved.parents and resolved != session_root:
            raise HTTPException(status_code=400, detail="Model path is outside the session directory.")
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"Model artifact not found on disk: {resolved}")
        return resolved

    def all_model_paths(self) -> list[tuple[str, Path]]:
        """Return list of (model_name, path) for all completed models."""
        training_summary = self.training_summary()
        if not training_summary:
            return []
        results: list[tuple[str, Path]] = []
        for model_item in training_summary.get("models", []):
            if model_item.get("status") != "completed":
                continue
            model_name = model_item.get("model_name", "")
            model_path_str = model_item.get("model_path")
            if not model_path_str:
                continue
            candidate = Path(model_path_str)
            if not candidate.is_absolute():
                candidate = self.session_dir / candidate
            if candidate.resolve().is_file():
                results.append((model_name, candidate.resolve()))
        return results

    def _index_overfitting(self) -> dict[str, dict[str, Any]]:
        """Build a per-model overfitting summary from per-model analysis files."""
        overfitting_root = self.evaluation_dir / "overfitting"
        result: dict[str, dict[str, Any]] = {}
        if not overfitting_root.is_dir():
            return result
        for model_dir in overfitting_root.iterdir():
            if not model_dir.is_dir():
                continue
            analysis_path = model_dir / OVERFITTING_ANALYSIS_FILENAME
            analysis = self._read_json_or_none(analysis_path)
            if analysis is None:
                continue
            primary_metric = analysis.get("primary_metric")
            gaps = analysis.get("gaps", {})
            result[model_dir.name] = {
                "is_overfitted": analysis.get("is_overfitted"),
                "primary_metric": primary_metric,
                "gap": gaps.get(primary_metric) if primary_metric else None,
                "gap_threshold": analysis.get("gap_threshold"),
                "gaps": gaps,
                "train_metrics": analysis.get("train_metrics"),
                "test_metrics": analysis.get("test_metrics"),
                "cv_results": analysis.get("k_fold_cross_validation_results"),
            }
        return result

    def _index_hpt_results(self) -> dict[str, dict[str, Any]]:
        """Index HPT best params and score by model name from hpt_results.json.

        The file written by training_service._execute_hpt has this shape::

            {"hpt_results": [{"name": "...", "best_hyperparameters": {...},
                              "val_metrics": {...}, "n_trials": 5, ...}]}
        """
        hpt_path = self.evaluation_dir / "hpt" / "hpt_results.json"
        if not hpt_path.is_file():
            return {}
        try:
            raw_data = json.loads(hpt_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for entry in raw_data.get("hpt_results", []):
            model_name = entry.get("name") or entry.get("model_name")
            if not model_name:
                continue
            # Derive a single scalar best_score from val_metrics
            val_metrics = entry.get("val_metrics") or {}
            best_score = (
                val_metrics.get("accuracy")
                or val_metrics.get("r2")
                or val_metrics.get("f1")
                or next(iter(val_metrics.values()), None)
            )
            result[model_name] = {
                "best_hyperparameters": entry.get("best_hyperparameters"),
                "best_score": best_score,
                "val_metrics": val_metrics,
                "n_trials": entry.get("n_trials"),
                "primary_metric": entry.get("primary_metric"),
                "tuning_time_seconds": entry.get("tuning_time_seconds"),
            }
        return result

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
        """Index per-model validation metrics by model name.

        model_item["metrics"] nests per-split values under "train"/"validation"
        (see backend/agents/training/metrics.py). The leaderboard only ever
        displays validation-set metrics, so flatten that nesting here rather
        than pushing the train/validation split into every frontend caller.
        """
        if not training_summary:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for model_item in training_summary.get("models", []):
            if model_item.get("status") != "completed":
                continue
            raw_metrics = model_item.get("metrics", {}) or {}
            flat_metrics = raw_metrics.get("validation") or raw_metrics
            index[model_item.get("model_name")] = {
                "validation_score": model_item.get("validation_score"),
                "metrics": flat_metrics,
            }
        return index

    @staticmethod
    def _leaderboard_from_metrics_only(
        metrics_by_model: dict[str, dict[str, Any]],
        overfitting_by_model: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        overfitting_by_model = overfitting_by_model or {}
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
                "llm_flags": [],
                # Judge has not run yet => no structured findings/decision yet.
                "decision": "PENDING",
                "findings": [],
                "ranking_explanation": None,
                "validation_score": training_record.get("validation_score"),
                "metrics": training_record.get("metrics", {}),
                "overfitting": overfitting_by_model.get(model_name),
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


@router.post("/{session_id}/hpt/run")
def run_hpt_endpoint(
    session_id: str,
    request: Request,
) -> dict[str, str]:
    from backend.services.dependencies import get_training_service
    training_service = get_training_service(request)
    training_service.run_hpt(session_id)
    return {"session_id": session_id, "status": "running"}


@router.get("/{session_id}/verdict")
def get_verdict(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.verdict()}


@router.get("/{session_id}/evaluation/shap/status")
def get_shap_status(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.shap_status()}


@router.get("/{session_id}/evaluation/overfitting/status")
def get_overfitting_status(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.overfitting_status()}


@router.get("/{session_id}/evaluation/judge/status")
def get_judge_status(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.judge_status()}


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


@router.get("/{session_id}/feature-engineering")
def get_feature_engineering(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.feature_engineering_status()}


@router.get("/{session_id}/d2v-prior")
def get_d2v_prior(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.d2v_prior()}


@router.get("/{session_id}/validation")
def get_validation(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.validation_report()}


@router.get("/{session_id}/model-config")
def get_model_config(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.model_config()}


@router.get("/{session_id}/hpt")
def get_hpt_results(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, **reader.hpt_results()}


@router.get("/{session_id}/models/download-all")
def download_all_models(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    """Stream a zip archive of all completed model artifacts."""
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    model_paths = reader.all_model_paths()
    if not model_paths:
        raise HTTPException(status_code=404, detail="No trained model artifacts found.")

    def zip_generator():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_archive:
            for model_name, model_path in model_paths:
                archive_name = f"{model_name}/{model_path.name}"
                zip_archive.write(model_path, arcname=archive_name)
        zip_buffer.seek(0)
        yield zip_buffer.read()

    return StreamingResponse(
        zip_generator(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=models_{session_id}.zip"},
    )


@router.get("/{session_id}/models/{model_name}/download")
def download_model(
    session_id: str,
    model_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileResponse:
    """Download a single trained model artifact by name."""
    reader = _build_reader(session_id=session_id, session_manager=session_manager)
    model_path = reader.resolve_model_path(model_name=model_name)
    return FileResponse(
        path=model_path,
        filename=model_path.name,
        media_type="application/octet-stream",
    )


@router.post("/{session_id}/plots/generate")
def generate_plots(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Trigger plot generation for the given session ID."""
    session_dir = session_manager.get_session_path(session_id=session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    try:
        plot_generator = PipelinePlotGenerator(session_dir=session_dir)
        results = plot_generator.generate_all()
        return {
            "status": "success",
            "message": "Visualizations generated successfully",
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

