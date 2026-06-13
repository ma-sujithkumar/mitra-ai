import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from config_loader import ConfigLoader

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_workspace_root() -> Path:
    workspace_root = ConfigLoader.get_str("paths", "WORKSPACE_ROOT")
    return Path(__file__).parent.parent.parent / workspace_root


def _load_run_summary(session_dir: Path) -> Optional[dict]:
    report_path = session_dir / "reports" / "validation_report.json"
    metadata_path = session_dir / "reports" / "metadata.json"

    if not report_path.exists():
        return None

    with open(report_path) as report_file:
        report = json.load(report_file)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)

    stat = session_dir.stat()
    return {
        "id": session_dir.name[:8],
        "session_id": session_dir.name,
        "dataset": metadata.get("statistics", {}) and "data.csv",
        "task": metadata.get("problem_type", "unknown"),
        "best": metadata.get("target_col", "—"),
        "acc": None,
        "status": "done" if report.get("passed") else "review",
        "drift": "stable",
        "created_at": stat.st_mtime,
    }


@router.get("/runs")
async def list_runs(limit: int = Query(default=5, ge=1, le=50)) -> JSONResponse:
    workspace = _get_workspace_root()
    if not workspace.exists():
        return JSONResponse(content={"runs": []})

    run_summaries = []
    for session_dir in sorted(workspace.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not session_dir.is_dir() or session_dir.name == "logs":
            continue
        summary = _load_run_summary(session_dir)
        if summary:
            run_summaries.append(summary)
        if len(run_summaries) >= limit:
            break

    return JSONResponse(content={"runs": run_summaries})


@router.get("/runs/stats")
async def get_run_stats() -> JSONResponse:
    workspace = _get_workspace_root()
    total_runs = 0
    models_trained = 0

    if workspace.exists():
        for session_dir in workspace.iterdir():
            if not session_dir.is_dir() or session_dir.name == "logs":
                continue
            if (session_dir / "reports" / "validation_report.json").exists():
                total_runs += 1

    return JSONResponse(content={
        "total_runs": total_runs,
        "models_trained": models_trained,
        "best_accuracy": None,
        "avg_run_time_min": None,
    })
