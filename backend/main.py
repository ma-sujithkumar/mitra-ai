import logging
from contextlib import asynccontextmanager
from time import time
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.activity_log import configure_file_logging
from backend.agents.llm_smoke_test import LlmSmokeTester
from backend.agents.metadata_gen_agent import MetadataAgentRunner
from backend.auth.db import AuthDatabase
from backend.auth.service import AuthService
from backend.config_loader import ConfigLoader
from backend.config_loader import LoggingConfig
from backend.jobs import JobRegistry
from backend.routers import auth
from backend.routers import config
from backend.routers import evaluation
from backend.routers import feature_engineering
from backend.routers import health
from backend.routers import llm
from backend.routers import logs
from backend.routers import metadata
from backend.routers import runs
from backend.routers import upload
from backend.routers import training
from backend.routers import training_events
from backend.routers import validate
from backend.session import SessionManager
from backend.services.training_service import TrainingService
from backend.orchestration.events import TrainingEventBus


def _configure_mitra_logging(logging_config: LoggingConfig) -> None:
    # Ensure the application's own loggers emit to the console at INFO, since
    # uvicorn only attaches handlers to its own logger namespaces. A rotating
    # file handler is added so all backend activity is also persisted to disk.
    mitra_logger = logging.getLogger("mitra")
    mitra_logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in mitra_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        mitra_logger.addHandler(handler)
        mitra_logger.propagate = False
    configure_file_logging(
        log_file=logging_config.log_file,
        level=logging_config.log_level,
        max_bytes=logging_config.log_max_bytes,
        backup_count=logging_config.log_backup_count,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Replaces the removed Starlette add_event_handler("shutdown", ...) API.
    # Runs the training service shutdown when the server stops.
    yield
    app.state.training_service.shutdown()


def create_app(config_loader: ConfigLoader | None = None) -> FastAPI:
    resolved_config_loader = config_loader or ConfigLoader()
    _configure_mitra_logging(logging_config=resolved_config_loader.logging)
    app = FastAPI(title="MITRA Epic 1 API", lifespan=_lifespan)
    app.state.started_at_epoch = time()
    app.state.config_loader = resolved_config_loader
    app.state.session_manager = SessionManager(
        workspace_root=resolved_config_loader.paths.workspace_root
    )
    app.state.job_registry = JobRegistry()
    app.state.training_event_bus = TrainingEventBus()
    app.state.training_service = TrainingService(
        config_loader=resolved_config_loader,
        session_manager=app.state.session_manager,
        event_bus=app.state.training_event_bus,
    )
    app.state.metadata_agent_runner = MetadataAgentRunner()
    app.state.llm_smoke_tester = LlmSmokeTester()
    # Auth database is lazy: the engine/tables are created on first request so
    # the app still starts when PostgreSQL is unavailable; only auth endpoints fail then.
    auth_database = AuthDatabase(
        authdb_config=resolved_config_loader.authdb,
        repo_root=resolved_config_loader.repo_root,
    )
    app.state.auth_service = AuthService(
        database=auth_database,
        authdb_config=resolved_config_loader.authdb,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth.router)
    app.include_router(upload.router)
    app.include_router(validate.router)
    app.include_router(metadata.router)
    app.include_router(health.router)
    app.include_router(config.router)
    app.include_router(runs.router)
    app.include_router(llm.router)
    app.include_router(logs.router)
    app.include_router(training_events.router)
    app.include_router(training.router)
    app.include_router(evaluation.router)
    app.include_router(feature_engineering.router)

    return app


app = create_app()
