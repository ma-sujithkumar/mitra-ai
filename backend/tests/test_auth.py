from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.models import User
from backend.config_loader import ConfigLoader
from backend.main import create_app


def test_auth_signup_creates_user_and_workspace(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    payload = {
        "name": "Jane Doe",
        "username": "jane_doe",
        "password": "Password123",
    }
    response = client.post("/api/auth/signup", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["username"] == "jane_doe"
    assert data["name"] == "Jane Doe"
    assert "user_id" in data
    assert "workspace_path" in data

    # Verify workspace directory is created
    workspace_dir = Path(data["workspace_path"])
    assert workspace_dir.exists()
    assert workspace_dir.is_dir()
    assert workspace_dir.name == "jane_doe"


def test_auth_signup_rejects_duplicate_username(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    payload = {
        "name": "Jane Doe",
        "username": "jane_doe",
        "password": "Password123",
    }
    response1 = client.post("/api/auth/signup", json=payload)
    assert response1.status_code == 201

    # Attempt signup again with the same username
    response2 = client.post("/api/auth/signup", json=payload)
    assert response2.status_code == 409
    assert response2.json()["detail"]["message"] == "user id already exists"


def test_auth_login_success_and_failure(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    signup_payload = {
        "name": "John Doe",
        "username": "john_doe",
        "password": "Password123",
    }
    signup_response = client.post("/api/auth/signup", json=signup_payload)
    assert signup_response.status_code == 201

    # Login with correct credentials
    login_payload = {
        "username": "john_doe",
        "password": "Password123",
    }
    login_response = client.post("/api/auth/login", json=login_payload)
    assert login_response.status_code == 200
    assert login_response.json()["username"] == "john_doe"

    # Login with incorrect password
    invalid_password_payload = {
        "username": "john_doe",
        "password": "WrongPassword123",
    }
    invalid_password_response = client.post("/api/auth/login", json=invalid_password_payload)
    assert invalid_password_response.status_code == 401
    assert invalid_password_response.json()["detail"]["message"] == "invalid username or password"

    # Login with non-existent user
    non_existent_payload = {
        "username": "non_existent",
        "password": "Password123",
    }
    non_existent_response = client.post("/api/auth/login", json=non_existent_payload)
    assert non_existent_response.status_code == 401
    assert non_existent_response.json()["detail"]["message"] == "invalid username or password"


def test_password_is_hashed_in_database(
    test_config_loader: ConfigLoader,
) -> None:
    app = create_app(config_loader=test_config_loader)
    client = TestClient(app)

    signup_payload = {
        "name": "Alice Smith",
        "username": "alice_smith",
        "password": "MySecurePassword123",
    }
    signup_response = client.post("/api/auth/signup", json=signup_payload)
    assert signup_response.status_code == 201

    # Verify that password is not stored in plaintext in the database
    auth_service = app.state.auth_service
    session: Session = auth_service.database.create_session()
    try:
        db_user = session.scalar(
            select(User).where(User.username == "alice_smith")
        )
        assert db_user is not None
        assert db_user.password_hash != "MySecurePassword123"
        assert auth_service.password_context.verify("MySecurePassword123", db_user.password_hash)
    finally:
        session.close()
