import configparser
import json
from pathlib import Path


# Config.ini lives one level above mitra-backend/ (repo root)
CONFIG_PATH = Path(__file__).parent.parent / "config.ini"


class ConfigLoader:
    _config: configparser.ConfigParser = None

    @classmethod
    def _load(cls) -> configparser.ConfigParser:
        if cls._config is None:
            cls._config = configparser.ConfigParser()
            read_paths = cls._config.read(CONFIG_PATH)
            if not read_paths:
                raise FileNotFoundError(f"config.ini not found at {CONFIG_PATH}")
        return cls._config

    @classmethod
    def get_str(cls, section: str, key: str) -> str:
        return cls._load().get(section, key)

    @classmethod
    def get_int(cls, section: str, key: str) -> int:
        return cls._load().getint(section, key)

    @classmethod
    def get_float(cls, section: str, key: str) -> float:
        return cls._load().getfloat(section, key)

    @classmethod
    def get_json_list(cls, section: str, key: str) -> list:
        raw = cls._load().get(section, key)
        return json.loads(raw)
