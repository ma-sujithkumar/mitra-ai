import uuid
from pathlib import Path

from config_loader import ConfigLoader


class SessionManager:

    @staticmethod
    def _workspace_root() -> Path:
        workspace_root = ConfigLoader.get_str("paths", "WORKSPACE_ROOT")
        repo_root = Path(__file__).parent.parent
        return repo_root / workspace_root

    @classmethod
    def create_session(cls) -> str:
        session_id = str(uuid.uuid4())
        workspace = cls._workspace_root()
        session_dir = workspace / session_id

        (session_dir / "data").mkdir(parents=True, exist_ok=True)
        (session_dir / "reports").mkdir(parents=True, exist_ok=True)

        log_dir = workspace / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        return session_id

    @classmethod
    def get_session_path(cls, session_id: str, subpath: str = "") -> Path:
        base = cls._workspace_root() / session_id
        if subpath:
            return base / subpath
        return base

    @classmethod
    def session_exists(cls, session_id: str) -> bool:
        return (cls._workspace_root() / session_id).exists()
