from __future__ import annotations

from pathlib import Path

from epic_3.ray_wrapper.config import RaySettings


def test_loads_ray_settings_from_project_config() -> None:
    settings = RaySettings.from_project_config()

    assert settings.address == "auto"
    assert settings.namespace == "mitra-epic3"
    assert settings.job_timeout_sec == 300.0
    assert settings.default_cpus_per_job == 1.0
    assert settings.image_gpus_per_job == 1.0
    assert settings.resolved_local_num_cpus() >= 1


def test_none_address_disables_external_connection(tmp_path: Path) -> None:
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[ray]\n"
        "ADDRESS=none\n"
        "NAMESPACE=test\n"
        "LOCAL_NUM_CPUS=2\n"
        "INCLUDE_DASHBOARD=false\n"
        "JOB_TIMEOUT_SEC=10\n"
        "DEFAULT_CPUS_PER_JOB=1\n"
        "IMAGE_CPUS_PER_JOB=2\n"
        "IMAGE_GPUS_PER_JOB=1\n"
        "DEFAULT_MEMORY_GB=0\n",
        encoding="utf-8",
    )

    settings = RaySettings.from_project_config(config_path)

    assert settings.address is None
    assert settings.resolved_local_num_cpus() == 2
