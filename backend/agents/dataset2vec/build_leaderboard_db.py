import argparse
import glob
import logging
import os

# Importing d2v_core.sweep triggers its module-level bootstrap (model_library_root
# inserted into sys.path AND into os.environ["PYTHONPATH"] so Ray worker
# subprocesses can resolve `core.*` imports too). core.config_loader is only
# importable after that bootstrap runs, so this import must stay first.
from d2v_core.sweep import LeaderboardSweep, MemoryJanitor
from d2v_core.schema import load_search_spaces, load_yaml_config, resolve_store_dir
from d2v_core.store import MetaKnowledgeStore

from core.config_loader import EXPECTED_MODELS

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PHASE 2: run the Optuna leaderboard sweep over the training corpus."
    )
    parser.add_argument("-c", "--config", required=True, type=str, help="path to config.ini")
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="comma-separated dataset_ids to restrict the sweep to (default: all *.npz in sweep.corpus_dir)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="acknowledges that already-completed (dataset_id, model_name) units will be skipped "
        "(LeaderboardSweep.run always skips them via store.completed_units(), regardless of this flag)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()


def discover_dataset_ids(corpus_dir: str) -> list[str]:
    """Globs corpus_dir for *.npz files and strips the extension, mirroring
    d2v_core/sampling.py::CorpusSampler._load_corpus's glob pattern."""
    npz_paths = sorted(glob.glob(os.path.join(corpus_dir, "*.npz")))
    dataset_ids = [os.path.splitext(os.path.basename(npz_path))[0] for npz_path in npz_paths]
    if len(dataset_ids) == 0:
        raise ValueError(f"=> no *.npz datasets found under corpus_dir '{corpus_dir}'.")
    return dataset_ids


def resolve_model_names(models_config: object) -> list[str]:
    """sweep.models in config.yaml is either the literal string 'all' (use
    EXPECTED_MODELS from model_library) or an explicit list of model names."""
    if models_config == "all":
        return list(EXPECTED_MODELS)
    return list(models_config)


def resolve_optuna_storage(optuna_storage_relative: str, tool_root: str) -> str:
    """config.yaml writes optuna_storage as 'sqlite:///store/optuna.db', relative
    to the tool root (store_dir resolves to '<tool_root>/store'). Strip the
    sqlite:/// prefix, join against tool_root, and re-absolutize so the sweep's
    SQLite file lives at exactly '<store_dir>/optuna.db' -- the same hardcoded
    path that MetaKnowledgeStore.completed_units() reads from."""
    sqlite_prefix = "sqlite:///"
    if not optuna_storage_relative.startswith(sqlite_prefix):
        raise ValueError(
            f"=> sweep.optuna_storage '{optuna_storage_relative}' does not start with '{sqlite_prefix}'."
        )
    relative_db_path = optuna_storage_relative[len(sqlite_prefix):]
    absolute_db_path = os.path.normpath(os.path.join(tool_root, relative_db_path))
    return f"{sqlite_prefix}{absolute_db_path}"


def resolve_scratch_dir(scratch_dir_relative: str, tool_root: str) -> str:
    """scratch_dir in config.yaml ('store/scratch') is relative to the tool root,
    same convention as optuna_storage above."""
    return os.path.normpath(os.path.join(tool_root, scratch_dir_relative))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sweep_config = load_yaml_config(args.config, "sweep")
    store_config = load_yaml_config(args.config, "store")

    corpus_dir = sweep_config["corpus_dir"]
    if corpus_dir is None:
        raise ValueError(
            "=> sweep.corpus_dir is null in config.yaml. Set it to a directory of "
            "*.npz datasets before running build_leaderboard_db.py."
        )
    if sweep_config["n_parallel"] is None:
        raise ValueError(
            "=> sweep.n_parallel is null in config.yaml. Set it to the number of "
            "Ray CPU workers before running build_leaderboard_db.py."
        )

    tool_root = os.path.normpath(os.path.join(os.path.dirname(args.config), ".."))
    sweep_config["optuna_storage"] = resolve_optuna_storage(sweep_config["optuna_storage"], tool_root)
    sweep_config["scratch_dir"] = resolve_scratch_dir(sweep_config["scratch_dir"], tool_root)

    logger.info(
        "=> --resume=%s passed; LeaderboardSweep.run always skips already-completed "
        "(dataset_id, model_name) units via store.completed_units(), regardless of this flag.",
        args.resume,
    )

    dataset_ids = (
        args.datasets.split(",") if args.datasets is not None else discover_dataset_ids(corpus_dir)
    )
    model_names = resolve_model_names(sweep_config["models"])

    search_spaces = load_search_spaces(args.config)
    store_dir = resolve_store_dir(args.config)
    store = MetaKnowledgeStore(
        store_dir=store_dir,
        faiss_metric=store_config["faiss_metric"],
        normalize_embeddings=store_config["normalize_embeddings"],
    )

    janitor = MemoryJanitor(
        scratch_dir=sweep_config["scratch_dir"],
        cleanup_interval_seconds=sweep_config["cleanup_interval_seconds"],
    )
    janitor.start()

    sweep = LeaderboardSweep(store=store, search_spaces=search_spaces, sweep_config=sweep_config)
    n_units_dispatched = sweep.run(corpus_dir, dataset_ids, model_names)
    logger.info(
        "=> sweep finished: dispatched %d unit(s) across %d dataset(s) x %d model(s).",
        n_units_dispatched, len(dataset_ids), len(model_names),
    )

    janitor.stop()


if __name__ == "__main__":
    main()
