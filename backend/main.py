import logging
from contextlib import asynccontextmanager
from time import time
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.llm_smoke_test import LlmSmokeTester
from backend.agents.metadata_gen_agent import MetadataAgentRunner
from backend.config_loader import ConfigLoader
from backend.jobs import JobRegistry
from backend.routers import config
from backend.routers import health
from backend.routers import llm
from backend.routers import metadata
from backend.routers import runs
from backend.routers import upload
from backend.routers import training
from backend.routers import training_events
from backend.routers import validate
from backend.session import SessionManager
from backend.services.training_service import TrainingService
from epic_3.events import TrainingEventBus


def _configure_mitra_logging() -> None:
    # Ensure the application's own loggers emit to the console at INFO, since
    # uvicorn only attaches handlers to its own logger namespaces.
    mitra_logger = logging.getLogger("mitra")
    mitra_logger.setLevel(logging.INFO)
    if not mitra_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        mitra_logger.addHandler(handler)
        mitra_logger.propagate = False


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Replaces the removed Starlette add_event_handler("shutdown", ...) API.
    # Runs the training service shutdown when the server stops.
    yield
    app.state.training_service.shutdown()


def create_app(config_loader: ConfigLoader | None = None) -> FastAPI:
    _configure_mitra_logging()
    resolved_config_loader = config_loader or ConfigLoader()
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(upload.router)
    app.include_router(validate.router)
    app.include_router(metadata.router)
    app.include_router(health.router)
    app.include_router(config.router)
    app.include_router(runs.router)
    app.include_router(llm.router)
    app.include_router(training_events.router)
    app.include_router(training.router)

    return app


app = create_app()
