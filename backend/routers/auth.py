from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from backend.auth.schemas import AuthResponse
from backend.auth.schemas import LoginRequest
from backend.auth.schemas import SignupRequest
from backend.auth.service import AuthError
from backend.auth.service import AuthService
from backend.dependencies import get_auth_service


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse, status_code=201)
def signup(
    payload: SignupRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    try:
        result = auth_service.signup(
            name=payload.name,
            username=payload.username,
            password=payload.password,
        )
    except AuthError as auth_error:
        raise HTTPException(
            status_code=auth_error.status_code,
            detail={"message": auth_error.message},
        )
    return AuthResponse(**result)


@router.post("/login", response_model=AuthResponse)
def login(
    payload: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    try:
        result = auth_service.login(
            username=payload.username,
            password=payload.password,
        )
    except AuthError as auth_error:
        raise HTTPException(
            status_code=auth_error.status_code,
            detail={"message": auth_error.message},
        )
    return AuthResponse(**result)
