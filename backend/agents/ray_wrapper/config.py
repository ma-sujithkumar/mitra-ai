"""Central project configuration for Epic-3 Ray execution."""

from __future__ import annotations

import configparser
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RaySettings(BaseModel):
    """Ray settings loaded from the project's single ``config.ini`` file."""

    model_config = ConfigDict(extra="forbid")

    address: str | None
    namespace: str = Field(min_length=1)
    local_num_cpus: int = Field(ge=0)
    include_dashboard: bool
    job_timeout_sec: float = Field(gt=0.0)
    default_cpus_per_job: float = Field(gt=0.0)
    image_cpus_per_job: float = Field(gt=0.0)
    image_gpus_per_job: float = Field(ge=0.0)
    default_memory_gb: float = Field(ge=0.0)

    @classmethod
    def from_project_config(
        cls,
        config_path: str | Path | None = None,
    ) -> "RaySettings":
        """Load the ``[ray]`` section from the root project config."""

        # config.py -> ray_wrapper -> agents -> backend -> repo root
        project_root = Path(__file__).resolve().parents[3]
        source = Path(config_path or project_root / "config.ini").expanduser().resolve()
        parser = configparser.ConfigParser()
        loaded_files = parser.read(source)
        if not loaded_files:
            raise FileNotFoundError(f"Config file not found: {source}")
        if not parser.has_section("ray"):
            raise ValueError(f"Missing [ray] section in config: {source}")

        raw_address = parser.get("ray", "ADDRESS").strip()
        address = None if raw_address.lower() in {"", "none", "null"} else raw_address
        return cls(
            address=address,
            namespace=parser.get("ray", "NAMESPACE").strip(),
            local_num_cpus=parser.getint("ray", "LOCAL_NUM_CPUS"),
            include_dashboard=parser.getboolean("ray", "INCLUDE_DASHBOARD"),
            job_timeout_sec=parser.getfloat("ray", "JOB_TIMEOUT_SEC"),
            default_cpus_per_job=parser.getfloat(
                "ray", "DEFAULT_CPUS_PER_JOB"
            ),
            image_cpus_per_job=parser.getfloat(
                "ray", "IMAGE_CPUS_PER_JOB"
            ),
            image_gpus_per_job=parser.getfloat(
                "ray", "IMAGE_GPUS_PER_JOB"
            ),
            default_memory_gb=parser.getfloat("ray", "DEFAULT_MEMORY_GB"),
        )

    def resolved_local_num_cpus(self) -> int:
        """Return configured CPU count, or auto-size when config uses zero."""

        if self.local_num_cpus > 0:
            return self.local_num_cpus
        return max(1, (os.cpu_count() or 2) - 1)
