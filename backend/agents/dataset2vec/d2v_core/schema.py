import configparser
import os
from typing import Optional

import yaml
from pydantic import BaseModel


def load_ini(ini_path: str) -> configparser.ConfigParser:
    """Read config.ini at the given path. Raises if the path does not exist."""
    if not os.path.isfile(ini_path):
        raise FileNotFoundError(f"=> config.ini not found at '{ini_path}'.")
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    return parser


def load_yaml_config(ini_path: str, top_level_key: Optional[str] = None) -> dict:
    """Resolve config.yaml via config.ini [paths] config_yaml, relative to the tool
    root (parent directory of the config/ folder holding config.ini), then return
    either the full parsed dict or one top-level section of it."""
    ini_parser = load_ini(ini_path)
    config_yaml_relative = ini_parser.get("paths", "config_yaml")
    tool_root = os.path.normpath(os.path.join(os.path.dirname(ini_path), ".."))
    config_yaml_path = os.path.join(tool_root, config_yaml_relative)
    with open(config_yaml_path, "r") as yaml_file:
        raw_config = yaml.safe_load(yaml_file)
    if top_level_key is None:
        return raw_config
    if top_level_key not in raw_config:
        raise KeyError(f"=> config.yaml has no top-level key '{top_level_key}'.")
    return raw_config[top_level_key]


def load_search_spaces(ini_path: str) -> dict:
    """Resolve search_spaces.json via config.ini [paths] search_spaces_json."""
    ini_parser = load_ini(ini_path)
    search_spaces_relative = ini_parser.get("paths", "search_spaces_json")
    tool_root = os.path.normpath(os.path.join(os.path.dirname(ini_path), ".."))
    search_spaces_path = os.path.join(tool_root, search_spaces_relative)
    import json

    with open(search_spaces_path, "r") as json_file:
        return json.load(json_file)


def resolve_store_dir(ini_path: str) -> str:
    """Resolve config.ini [paths] store_dir to an absolute path under the tool root."""
    ini_parser = load_ini(ini_path)
    store_dir_relative = ini_parser.get("paths", "store_dir")
    tool_root = os.path.normpath(os.path.join(os.path.dirname(ini_path), ".."))
    return os.path.join(tool_root, store_dir_relative)


def resolve_model_library_root(ini_path: str) -> str:
    """Resolve config.ini [paths] model_library_root to an absolute path."""
    ini_parser = load_ini(ini_path)
    return ini_parser.get("paths", "model_library_root")


class LeaderboardEntry(BaseModel):
    rank: int
    model_name: str
    hyperparameters: dict
    metrics: dict
    n_trials: int


class LeaderboardRecord(BaseModel):
    dataset_id: str
    encoder_version: str
    # None when this record is written by Phase 2 alone (sweep runs before/
    # independently of Phase 1); the real embedding is attached at join time
    # by MetaKnowledgeStore.build_meta_kb() via train_embeddings.parquet.
    embedding: Optional[list[float]] = None
    task_type: str
    n_rows: int
    n_cols: int
    target_cardinality: int
    primary_metric: str
    leaderboard: list[LeaderboardEntry]
    best_model: str
    created_at: str


class NeighborResult(BaseModel):
    dataset_id: str
    similarity: float
    best_model: str
    recommended_hyperparameters: dict
    metrics: dict


class VerificationResult(BaseModel):
    trained: bool
    achieved_metric: Optional[float] = None
    achieved_full: Optional[dict] = None
    delta_vs_expected: Optional[float] = None
    within_tolerance: Optional[bool] = None


class RankedModelEntry(BaseModel):
    model_name: str
    score: float
    recommended_hyperparameters: dict
    expected_metric: float
    verification: Optional[VerificationResult] = None


class VerificationSummary(BaseModel):
    tolerance: float
    n_verified: int
    n_within_tolerance: int
    best_achieved: Optional[dict] = None
    mean_abs_delta: Optional[float] = None


class DatasetPrior(BaseModel):
    query_dataset_id: str
    encoder_version: str
    top_k: int
    primary_metric: str
    neighbors: list[NeighborResult]
    ranked_models: list[RankedModelEntry]
    verification_summary: Optional[VerificationSummary] = None
    cold_start: bool
    caveats: list[str]
