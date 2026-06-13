from time import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.metadata_gen_agent import MetadataAgentRunner
from backend.config_loader import ConfigLoader
from backend.jobs import JobRegistry
from backend.routers import config
from backend.routers import health
from backend.routers import metadata
from backend.routers import runs
from backend.routers import upload
from backend.routers import validate
from backend.session import SessionManager


def create_app(config_loader: ConfigLoader | None = None) -> FastAPI:
    resolved_config_loader = config_loader or ConfigLoader()
    app = FastAPI(title="MITRA Epic 1 API")
    app.state.started_at_epoch = time()
    app.state.config_loader = resolved_config_loader
    app.state.session_manager = SessionManager(
        workspace_root=resolved_config_loader.paths.workspace_root
    )
    app.state.job_registry = JobRegistry()
    app.state.metadata_agent_runner = MetadataAgentRunner()
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

    return app


app = create_app()
