import configparser
import os
from typing import Any, Dict

import yaml


def _resolve_ini_path() -> str:
    """Locate config.ini relative to this file."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "config", "config.ini"))


def _resolve_yaml_path() -> str:
    """Read config.ini to find config.yaml, resolving relative to judge_agent root."""
    ini_path = _resolve_ini_path()
    parser = configparser.ConfigParser()
    parser.read(ini_path)

    relative_yaml = parser.get("paths", "config_yaml")
    agent_root = os.path.normpath(os.path.dirname(__file__))
    return os.path.join(agent_root, relative_yaml)


def load_judge_config() -> Dict[str, Any]:
    """Load and return the full judge agent config.yaml as a dict.

    Raises:
        FileNotFoundError: If config.yaml cannot be located.
    """
    yaml_path = _resolve_yaml_path()
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"config.yaml not found at resolved path: {yaml_path}. "
            "Ensure config/config.ini [paths] config_yaml points to the right location."
        )
    with open(yaml_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def get_python_binary() -> str:
    """Return the Python binary path from config.ini [python] PYTHON."""
    ini_path = _resolve_ini_path()
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    return parser.get("python", "PYTHON")
