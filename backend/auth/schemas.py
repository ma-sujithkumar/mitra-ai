from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class SignupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=255)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=255)


class AuthResponse(BaseModel):
    user_id: str
    username: str
    name: str
    workspace_path: str
