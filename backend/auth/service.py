from __future__ import annotations

import re
from pathlib import Path

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.db import AuthDatabase
from backend.auth.models import User
from backend.config_loader import AuthDbConfig


# Allowed characters for a username when used as a directory name. Anything
# outside this set is rejected to prevent path traversal in mitra/<user_id>.
SAFE_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._@-]+$")


class AuthError(Exception):
    """Base class for auth failures, carrying an HTTP status code."""

    status_code: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DuplicateUserError(AuthError):
    status_code = 409


class InvalidCredentialsError(AuthError):
    status_code = 401


class WeakPasswordError(AuthError):
    status_code = 422


class InvalidUsernameError(AuthError):
    status_code = 422


class AuthService:
    def __init__(
        self,
        database: AuthDatabase,
        authdb_config: AuthDbConfig,
    ) -> None:
        self.database = database
        self.authdb_config = authdb_config
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def signup(self, name: str, username: str, password: str) -> dict[str, str]:
        normalized_username = self._validate_username(username)
        self._validate_password(password)
        password_hash = self.password_context.hash(password)
        session: Session = self.database.create_session()
        try:
            existing_user = session.scalar(
                select(User).where(User.username == normalized_username)
            )
            if existing_user is not None:
                raise DuplicateUserError("user id already exists")
            new_user = User(
                name=name.strip(),
                username=normalized_username,
                password_hash=password_hash,
            )
            session.add(new_user)
            session.commit()
            session.refresh(new_user)
            workspace_path = self._ensure_user_workspace(normalized_username)
            return {
                "user_id": normalized_username,
                "username": normalized_username,
                "name": new_user.name,
                "workspace_path": str(workspace_path),
            }
        finally:
            session.close()

    def login(self, username: str, password: str) -> dict[str, str]:
        normalized_username = self._validate_username(username)
        session: Session = self.database.create_session()
        try:
            user = session.scalar(
                select(User).where(User.username == normalized_username)
            )
            if user is None or not self.password_context.verify(
                password, user.password_hash
            ):
                raise InvalidCredentialsError("invalid username or password")
            workspace_path = self._ensure_user_workspace(normalized_username)
            return {
                "user_id": normalized_username,
                "username": normalized_username,
                "name": user.name,
                "workspace_path": str(workspace_path),
            }
        finally:
            session.close()

    def _validate_username(self, username: str) -> str:
        normalized_username = username.strip()
        if not SAFE_USERNAME_PATTERN.match(normalized_username):
            raise InvalidUsernameError(
                "username may only contain letters, digits, and . _ - @"
            )
        return normalized_username

    def _validate_password(self, password: str) -> None:
        minimum_length = self.authdb_config.password_min_length
        has_letter = any(character.isalpha() for character in password)
        has_digit = any(character.isdigit() for character in password)
        if len(password) < minimum_length or not has_letter or not has_digit:
            raise WeakPasswordError(
                "password must be at least "
                f"{minimum_length} characters and include a letter and a digit"
            )

    def _ensure_user_workspace(self, user_id: str) -> Path:
        # mkdir -p mitra/<user_id> on first successful auth.
        workspace_path = self.authdb_config.user_workspace_root / user_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        return workspace_path
