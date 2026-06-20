from __future__ import annotations

import os

from dotenv import dotenv_values
from sqlalchemy import Engine
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from backend.auth.models import Base
from backend.config_loader import AuthDbConfig


class AuthDatabase:
    """Lazily builds a PostgreSQL engine from config + environment values.

    The engine and tables are created on first use so the main application can
    still start when PostgreSQL is unavailable; only auth endpoints fail then.
    """

    def __init__(self, authdb_config: AuthDbConfig, repo_root) -> None:
        self.authdb_config = authdb_config
        self.repo_root = repo_root
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    def _resolve_env(self) -> dict[str, str]:
        # Merge .env file values with process environment, preferring the
        # process environment when both define the same key.
        env_file_values = dotenv_values(self.repo_root / ".env")
        merged: dict[str, str] = {
            key: value for key, value in env_file_values.items() if value is not None
        }
        merged.update(
            {key: value for key, value in os.environ.items() if value is not None}
        )
        return merged

    def _build_url(self) -> URL:
        config = self.authdb_config
        env_values = self._resolve_env()
        host = env_values.get(config.db_host_env, config.db_host_default)
        port = env_values.get(config.db_port_env, config.db_port_default)
        database = env_values.get(config.db_name_env, config.db_name_default)
        username = env_values.get(config.db_user_env, config.db_user_default)
        password = env_values.get(config.db_password_env, "")
        return URL.create(
            drivername="postgresql+psycopg2",
            username=username,
            password=password,
            host=host,
            port=int(port),
            database=database,
        )

    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self._build_url(), pool_pre_ping=True)
            Base.metadata.create_all(self._engine)
        return self._engine

    def session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine(), expire_on_commit=False
            )
        return self._session_factory

    def create_session(self) -> Session:
        return self.session_factory()()
