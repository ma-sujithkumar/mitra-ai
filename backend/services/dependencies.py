from __future__ import annotations

from fastapi import Request

from backend.services.training_service import TrainingService


def get_training_service(request: Request) -> TrainingService:
    return request.app.state.training_service
