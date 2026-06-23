# Dataset2Vec Meta-Knowledge Base -- How to Run

## 0. One-time setup
```bash
# install the three packages this tool needs that model_library doesn't have yet
/home/sujithma/venv/bin/pip install optuna ray faiss-cpu

# fill in config/config.ini (one global ini, see [paths] section) and
# config/config.yaml: set training.corpus_dir and sweep.corpus_dir to a directory
# of *.npz files (each with X_train, y_train, X_test, y_test arrays). Both are
# REQUIRED -- the tool errors out if left null, it will not silently default.
# Also set sweep.n_parallel to the number of Ray CPU workers for your machine.

mkdir -p claude_outputs store/scratch store/encoder/checkpoints
```

## 1. PHASE 1 -- train the Dataset2Vec encoder (GPU)
```bash
PYBIN=/home/sujithma/venv/bin/python
CUDA_VISIBLE_DEVICES=0 nohup $PYBIN train_encoder.py \
    -c config/config.ini --resume -v \
    > claude_outputs/phase1_encoder.log 2>&1 &
echo "encoder PID $!"
tail -f claude_outputs/phase1_encoder.log
```
Produces `store/encoder/encoder.pt` + `encoder_version.json` and
`store/train_embeddings.parquet` (one embedding per training dataset). Safe to
kill and restart with `--resume` -- it continues from the latest checkpoint.

## 2. PHASE 2 -- parallel Optuna leaderboard sweep
GPU trials run ONE AT A TIME (`sweep.gpu_fraction_pytorch: 1.0` in config.yaml,
set this way because the GPU only has 4GB VRAM); sklearn/xgboost trials run in
parallel on CPU via Ray. Set `sweep.n_parallel` to fit your CPU core count.

```bash
# smoke test on 2 datasets first
$PYBIN build_leaderboard_db.py -c config/config.ini \
    --datasets <dataset_id_1>,<dataset_id_2> --resume -v

# full overnight run (sweeps every *.npz under sweep.corpus_dir if --datasets
# is omitted, against every model in sweep.models -- "all" means MITRA's full
# 60-model EXPECTED_MODELS list)
CUDA_VISIBLE_DEVICES=0 nohup $PYBIN build_leaderboard_db.py \
    -c config/config.ini --resume -v \
    > claude_outputs/phase2_leaderboard.log 2>&1 &
echo "leaderboard PID $!"

# monitor progress and confirm memory stays flat (the hourly janitor should
# keep both flat across a multi-hour run)
tail -f claude_outputs/phase2_leaderboard.log
watch -n 30 nvidia-smi
watch -n 30 free -g
```
Produces `store/leaderboards.parquet`, `store/meta_kb.parquet` (the
embedding -> leaderboard join, an inner join on dataset_id so a dataset only
appears once it has BOTH an embedding from Phase 1 and a leaderboard from
Phase 2), and `store/index.faiss`. Safe to kill and restart with `--resume` --
already-completed `(dataset_id, model_name)` Optuna studies are skipped via
`store.completed_units()`, regardless of whether `--resume` is passed (the
underlying skip-logic always applies; the flag is informational).

## 3. PHASE 3 -- warm-start query + verify on a NEW test dataset
```bash
$PYBIN query.py -c config/config.ini \
    -i path/to/test_dataset.npz -o claude_outputs/prior -k 5 --verify -v
cat claude_outputs/prior/dataset_prior.json
```
`dataset_prior.json` contains:
- `neighbors`: the top-k most similar training datasets (FAISS cosine
  similarity over L2-normalized Dataset2Vec embeddings), each with its
  best_model/hyperparameters/metrics.
- `ranked_models`: the top suggested models + the hyperparameters that won on
  similar training datasets, ranked by a similarity-weighted vote across
  every neighbor's full leaderboard (not just each neighbor's single best
  model), with an `expected_metric`.
- `verification` (per ranked model, only populated with `--verify`): the
  result of ACTUALLY TRAINING that model on your test dataset --
  `achieved_metric`, `delta_vs_expected`, `within_tolerance` (default
  tolerance `retrieval.verify_tolerance` = 0.05).
- `verification_summary` (only populated with `--verify`): rollup across all
  suggestions (`n_verified`, `n_within_tolerance`, `mean_abs_delta`,
  `best_achieved`).
- `cold_start: true` if the store is empty or has no same-task neighbors yet
  -- in that case `neighbors`/`ranked_models` are empty and no training
  happens, regardless of `--verify`.

CLI flags: `-c/--config` (required), `-i/--input` (required, .npz with
X_train/y_train/X_test/y_test), `-o/--output` (required, directory --
created if missing), `-k/--top-k` (optional, overrides `retrieval.top_k`
for the neighbor search only), `--verify` (optional, trains every
recommended model on the test dataset), `-v/--verbose`.

## 4. Running the test suite (fast, no GPU required)
```bash
$PYBIN tests/test_meta_kb.py -v
```
Runs all 3 phases end-to-end on 5 tiny toy/synthetic datasets in an isolated
temp store (not your real `store/`), CPU-only. Finishes in well under a
minute (~35s) -- explicitly separate from the real overnight 60-model corpus
run described above.

## Config reference (`config/config.yaml`)

| Section | Key | Meaning |
|---|---|---|
| `encoder` | `embedding_dim` | Fixed-length output dimension of the Dataset2Vec encoder. |
| `encoder` | `f_block`/`g_block`/`h_block` | Hidden size/layers/residual-unit count for each of the three sub-networks. |
| `encoder` | `encoder_version` | Tag written into `encoder_version.json` and every embedding row. |
| `training` | `corpus_dir` | REQUIRED. Directory of `*.npz` training datasets for Phase 1. |
| `training` | `device` | `cuda` or `cpu`. |
| `training` | `n_instances_sample`/`n_features_sample`/`n_classes_sample` | Per-patch sampling sizes (fixed input dimension trick). |
| `training` | `epochs`/`es_patience`/`es_min_delta`/`checkpoint_every` | Training loop / early-stopping / checkpoint cadence. |
| `sweep` | `corpus_dir` | REQUIRED. Directory of `*.npz` datasets for Phase 2 (can differ from `training.corpus_dir`). |
| `sweep` | `models` | `"all"` (MITRA's 60 `EXPECTED_MODELS`) or an explicit list. |
| `sweep` | `n_parallel` | REQUIRED. Ray CPU worker count. |
| `sweep` | `n_trials_per_model` | Optuna trials per `(dataset_id, model_name)` unit. |
| `sweep` | `primary_metric` | Metric Optuna optimizes and leaderboards rank by. |
| `sweep` | `gpu_fraction_pytorch` | GPU resource request per PyTorch trial (1.0 = one at a time on a 4GB GPU). |
| `sweep` | `optuna_storage`/`scratch_dir`/`cleanup_interval_seconds` | SQLite path, janitor scratch dir, janitor cadence. |
| `store` | `faiss_metric`/`normalize_embeddings` | FAISS index type and whether embeddings are L2-normalized before indexing. |
| `retrieval` | `top_k`/`n_recommended_models`/`same_task_only`/`min_similarity` | Phase 3 neighbor search and ranking knobs. |
| `retrieval` | `model_vote` | Ranking strategy (`similarity_weighted` is the only one implemented). |
| `retrieval` | `verify_tolerance` | `|achieved - expected|` threshold for `within_tolerance`. |

`store_dir` itself is NOT in config.yaml -- it lives in `config/config.ini`'s
`[paths] store_dir`, since paths are never duplicated between the two files.
