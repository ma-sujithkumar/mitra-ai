from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from backend.services.dependencies import get_training_service
from backend.schemas.training import TrainingCancelResponse
from backend.schemas.training import TrainingStartRequest
from backend.schemas.training import TrainingStartResponse
from backend.schemas.training import TrainingStatusResponse
from backend.services.training_service import TrainingArtifactError
from backend.services.training_service import TrainingCancellationError
from backend.services.training_service import TrainingRunConflictError
from backend.services.training_service import TrainingRunNotFoundError
from backend.services.training_service import TrainingService
from backend.services.training_service import TrainingSessionNotFoundError

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", response_model=TrainingStartResponse, status_code=202)
def start_training(
    training_request: TrainingStartRequest,
    training_service: TrainingService = Depends(get_training_service),
) -> TrainingStartResponse:
    try:
        state = training_service.start(training_request)
    except TrainingSessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "SESSION_NOT_FOUND", "message": str(exc)},
        ) from exc
    except TrainingArtifactError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "TRAINING_ARTIFACTS_INVALID",
                "message": str(exc),
                "missing_paths": exc.missing_paths,
            },
        ) from exc
    except TrainingRunConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "TRAINING_ALREADY_EXISTS", "message": str(exc)},
        ) from exc

    return TrainingStartResponse(
        session_id=state.session_id,
        status=state.status,
        execution_mode=state.execution_mode,
        status_url=f"/api/training/status/{state.session_id}",
        events_url=f"/api/training/events?session_id={state.session_id}",
    )


@router.get("/status/{session_id}", response_model=TrainingStatusResponse)
def get_training_status(
    session_id: str,
    training_service: TrainingService = Depends(get_training_service),
) -> TrainingStatusResponse:
    try:
        return training_service.get_status(session_id)
    except TrainingRunNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "TRAINING_NOT_FOUND", "message": str(exc)},
        ) from exc


@router.post("/cancel/{session_id}", response_model=TrainingCancelResponse)
def cancel_training(
    session_id: str,
    training_service: TrainingService = Depends(get_training_service),
) -> TrainingCancelResponse:
    try:
        state = training_service.cancel(session_id)
    except TrainingRunNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "TRAINING_NOT_FOUND", "message": str(exc)},
        ) from exc
    except TrainingCancellationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "TRAINING_NOT_CANCELLABLE", "message": str(exc)},
        ) from exc

    return TrainingCancelResponse(
        session_id=state.session_id,
        status=state.status,
        cancellation_requested=state.cancellation_requested,
        cancelled_jobs=state.cancelled_jobs,
    )
