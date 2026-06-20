# SPEC: Dataset2Vec Meta-Knowledge Base (Self-Improving AutoML Memory)

## 1. GOAL
Build a meta-learning knowledge base in three explicit, GPU-driven phases:

PHASE 1 (encoder): Train a Dataset2Vec encoder on the GPU using the available
corpus of datasets. The encoder projects ANY tabular dataset into a fixed
low-dimensional embedding that is invariant to the number of instances, features,
and classes. After training, generate and persist embeddings for the training
datasets.

PHASE 2 (leaderboard DB): For every training dataset, run a heavily parallelized
AutoML sweep (all applicable ML models, hyperparameters tuned with Optuna) to
produce a ranked leaderboard, and store an `embedding_vector -> leaderboard`
mapping in a single DB. This is the prior knowledge that powers warm-start.

PHASE 3 (query + VERIFY): Given a NEW test dataset, embed it with the frozen
Phase-1 encoder, retrieve the most similar training datasets from the DB, return a
ranked list of ML models + hyperparameters as a warm-start prior, then ACTUALLY
TRAIN those suggested models on the test dataset and match the achieved accuracy
against the metric the warm-start predicted.

The Dataset2Vec encoder gives us SIMILARITY across datasets; the Phase-2 leaderboard
DB gives us WHAT WORKED. The warm-start answer ("for this test dataset, here are the
ML models + hyperparameters to start from") is the join of the two. Phase 3 then
CLOSES THE LOOP by training the suggestions on the real test data and comparing
achieved-vs-expected accuracy, so the prior is validated, not just asserted.

Reference architecture: https://github.com/hadijomaa/dataset2vec
(Jomaa, Schmidt-Thieme, Grabocka. "Dataset2Vec: learning dataset meta-features",
Data Mining and Knowledge Discovery, 2021.) The original is TensorFlow; this spec
re-implements the encoder in PyTorch so the whole MITRA stack stays on one framework.

## 2. KEY DESIGN DECISIONS (read first)
1. The Dataset2Vec encoder is SELF-SUPERVISED. Phase 1 needs ONLY raw datasets
   (X, y). It does NOT need leaderboards. The contrastive objective only needs to
   know which patches came from the same source dataset.
2. The leaderboard prior is NOT produced by the encoder or by the paper. It is
   MANUFACTURED in Phase 2 by actually running ML models on each training dataset
   once. There is no public leaderboard dataset that ships with Dataset2Vec.
3. Phase 1 and Phase 2 are INDEPENDENT and can run concurrently. They join only on
   `dataset_id`. The only ordering constraint is that the embedding column in the
   leaderboard DB needs the trained encoder; leaderboards can be generated first and
   embeddings backfilled.
4. GPU is used in three places: (a) training the Dataset2Vec encoder (Phase 1),
   (b) training the PyTorch model family inside the Optuna sweep (Phase 2), and
   (c) training the suggested models on the test dataset in the Phase-3 verify step.
   All other models (sklearn, xgboost) run on CPU workers in parallel.
5. MEMORY DISCIPLINE (Phase 2 and Phase-3 verify): trained ML models are NEVER
   persisted. After each Optuna trial / verify run only
   `(model_name, hyperparameters, metrics)` is recorded; the model object is freed
   immediately and CUDA cache is cleared. A janitor sweep runs on a configurable
   interval (default hourly) to delete stray temp artifacts, run `gc.collect()`,
   and `torch.cuda.empty_cache()`.

## 3. REUSE EXISTING MITRA COMPONENTS (do NOT reimplement)
Verified against the repo (paths and signatures are real):
- `model_library/ml_kit.py::MLKit(model_name, data, training_mode).train()/.test()/.save()`
  => single entry point for all 60 registered models; pickle-serializable for Ray
  worker dispatch. This is the model runner inside the Phase-2 sweep AND the
  Phase-3 verify step.
- `model_library/core/data_bundle.py::CommonData(X_train, y_train, X_test, y_test)`
  and `DataBundle(common=..., hyperparameters={})`
  => the dataset container. The encoder reads `common.X_train`, `common.y_train`.
  The `hyperparameters` dict is the INJECTION POINT for Optuna (Phase 2) and for the
  recommended hyperparameters (Phase-3 verify): pass
  `DataBundle(common=..., hyperparameters=recommended_params)`.
- `model_library/metrics/evaluators.py::compute_metrics(y_true, y_pred, task_type, model_name)`
  => returns `MetricsResult` (accuracy, f1_macro, f1_weighted for classification;
  rmse, r2 for regression). Stored verbatim in leaderboards and used for the
  Phase-3 achieved-vs-expected comparison.
- `model_library/core/validators.py::validate_model_name` and `EXPECTED_MODELS`
  => the 60-name canonical vocabulary (30 classifiers + 30 regressors). The Phase-2
  model set and the Phase-3 `ranked_models` are restricted to these names.
- Import bootstrap mirrors `model_library/tests/test_*.py`: a config-driven
  `sys.path.insert(0, model_library_root)` before MITRA imports.

## 4. PACKAGES
- pytorch (CUDA build)   Dataset2Vec encoder + PyTorch model family on GPU
- numpy
- optuna                 hyperparameter tuning in Phase 2 (TPE sampler + pruner)
- ray                    process-level parallelism across (dataset, model) units
- faiss-cpu              similarity index over embeddings (Phase 3 retrieval)
- pandas + pyarrow       parquet leaderboard DB + embedding store
- pyyaml                 controllables
- pydantic               record / leaderboard / prior schema validation
- MITRA model_library    MLKit, CommonData, DataBundle, MetricsResult, validate_model_name

No try-except around imports (CLAUDE.md #8); if a package is missing the tool errors
out and asks the user to install it. All imports at the top of each file (CLAUDE.md #1).

================================================================================
## PHASE 1 - DATASET2VEC ENCODER (train on GPU, emit embeddings)
================================================================================

### 5.1 Input representation
A dataset patch is a predictor matrix X of shape (n, m) and a target matrix Y of
shape (n, t). Y is one-hot for classification (t = number of classes in the patch)
and a single column for regression (t = 1). Per-column standardization (mean 0,
std 1) is applied before encoding, controlled by `encoder.standardize_inputs`.

### 5.2 Hierarchical set network (f, g, h blocks)
A permutation-invariant set function built from three residual-MLP sub-networks. It
produces a fixed vector regardless of n, m, t:
1. For every feature column j and target column c, form the per-instance interaction
   vector [x(i,j), y(i,c)] in R^2 for i = 1..n.
2. Block f: residual MLP on each [x(i,j), y(i,c)], then MEAN-POOL over the n
   instances => one vector per (j, c) pair. Invariant to the number of rows n.
3. Block g: residual MLP on each pooled (j, c) vector, then MEAN-POOL over all m*t
   feature-target pairs => one fixed-length vector. Invariant to m and t.
4. Block h: final residual MLP mapping the pooled vector to embedding phi(D) in R^d
   (default d = 64).
All layer widths, residual-block counts, and d are config controllables (CLAUDE.md #5).

### 5.3 GPU training objective (Siamese / contrastive)
Two patches sampled from the SAME source dataset must land close together; patches
from DIFFERENT datasets must land far apart.
- Corpus: a pool of labeled datasets (`training.corpus_dir`); each is a "source".
- Sampling (`core/sampling.py`): each batch draws patches by subsampling rows, a
  subset of feature columns, and (for classification) a subset of classes. Pairs are
  labeled 1 if both patches come from the same source dataset, else 0.
- Similarity: s(D1, D2) = exp(-gamma * || phi(D1) - phi(D2) ||_2), gamma in config.
- Loss: binary cross-entropy between s(D1, D2) and the pair label.
- Optimizer: Adam; gradient clipping. Tensors and model are moved to
  `training.device` (`cuda` if available, else `cpu`). Optional AMP mixed precision
  via `training.use_amp` for larger batches on the GPU.
- Early stopping on validation pair-AUC (no improvement > `training.es_min_delta`
  for `training.es_patience` checks), mirroring the reference `d2v.py`.
- Checkpoints saved every `training.checkpoint_every` steps to
  `store/encoder/checkpoints/` so an overnight run is resumable.

### 5.4 Outputs of Phase 1
1. `store/encoder/encoder.pt` + `store/encoder/encoder_version.json`
   (frozen weights + version string, e.g. `d2v-v1`).
2. `store/embeddings/train_embeddings.parquet`: one row per training dataset
   `{dataset_id, encoder_version, embedding (length d), n_rows, n_cols,
   task_type, target_cardinality, created_at}`. This is the "generate embeddings
   for some train datasets" deliverable and the left side of the Phase-2 join.

### 5.5 Phase-1 entrypoint
`train_encoder.py -c <config> [--resume] [-v]`
- Trains on `training.corpus_dir`, saves the versioned encoder, then embeds every
  corpus dataset and writes `train_embeddings.parquet`.
- `--resume` continues from the latest checkpoint (for overnight runs).
- No hardcoded arg defaults => None + error (CLAUDE.md #27). All output dirs created
  with `mkdir -p` (CLAUDE.md #17).

================================================================================
## PHASE 2 - PARALLEL OPTUNA LEADERBOARD DB (embedding -> leaderboard)
================================================================================

### 6.1 What it produces
For every training dataset, a ranked leaderboard of `(model_name, best
hyperparameters, metrics)` and a single mapping DB keyed by dataset that carries the
Phase-1 embedding alongside its leaderboard.

### 6.2 The sweep unit and parallelism
The atomic unit of work is a `(dataset_id, model_name)` pair => one Optuna study.
- A dataset with `same_task_only` produces units only for models matching its task
  type (classifiers for classification, regressors for regression).
- Units are embarrassingly parallel across both datasets and models and are
  dispatched as Ray tasks across a worker pool sized by `sweep.n_parallel`.
- Resource-aware scheduling via a hash-map dispatch (CLAUDE.md #4/#23):
  `model_family_resources = {"pytorch": {"num_gpus": frac}, "sklearn": {"num_cpus": 1},
  "xgboost": {"num_cpus": k}}`. PyTorch models request a GPU fraction; sklearn/xgboost
  run on CPU workers so the GPU is reserved for the model family that needs it.
- Per study: Optuna `TPESampler` + `MedianPruner` run `sweep.n_trials_per_model`
  trials. Each trial:
  1. `trial_params = suggest_fn[model_name](trial)`  (search space from config, see 6.3)
  2. `data = DataBundle(common=CommonData(...), hyperparameters=trial_params)`
  3. `kit = MLKit(model_name, data, training_mode="full_train"); kit.train()`
  4. `y_pred = kit.test()`
  5. `result = compute_metrics(y_test, y_pred, task_type, model_name)`
  6. record `(trial_params, result)`; return the primary metric to Optuna.
  7. MEMORY: `del kit; gc.collect(); torch.cuda.empty_cache()` (CLAUDE.md, point 5).
- The study's best trial becomes one leaderboard row for that dataset. Rows per
  dataset are ranked by `sweep.primary_metric`; top `sweep.leaderboard_top_n` kept.

### 6.3 Hyperparameter search spaces (config-driven, no if-else ladder)
Search spaces live in `config/search_spaces.json`, one entry per model name mapping
to a list of typed param specs (e.g. `{"name": "n_estimators", "type": "int",
"low": 100, "high": 800}`). A single `build_suggestions(trial, space)` reads the
JSON and calls the matching `trial.suggest_*` via a type->function dispatch dict, so
adding a model is a config edit, not new code (CLAUDE.md #4/#5/#23). Models with no
entry run a single default-config trial.

### 6.4 Memory + cleanup discipline (must take care of memory)
- Trained models are NEVER saved (`kit.save()` is not called in Phase 2).
- After every trial: free the model, `gc.collect()`, and clear CUDA cache.
- A background JANITOR (`sweep.cleanup_interval_seconds`, default 3600 = hourly):
  deletes any temp files under `sweep.scratch_dir`, runs `gc.collect()`, and clears
  the CUDA cache on every GPU. Implemented as a periodic Ray actor / daemon thread,
  not an if-else ladder.
- `sweep.n_parallel` and the per-family resource map bound peak RAM/VRAM; set them
  to fit the machine (see overnight commands).
- Optuna uses a SQLite RDB backend (`sweep.optuna_storage`) so studies are durable
  and an overnight crash resumes instead of restarting. Only trial metrics live in
  SQLite; no model bytes.

### 6.5 The leaderboard DB (single mapping store)
```
store/
  encoder/                     # Phase 1: encoder.pt + encoder_version.json + checkpoints/
  embeddings/
    train_embeddings.parquet   # Phase 1: dataset_id -> embedding
  leaderboards.parquet         # Phase 2: dataset_id -> ranked leaderboard rows
  meta_kb.parquet              # JOIN view: embedding_vector + leaderboard per dataset
  index.faiss                  # Phase 3: FAISS over L2-normalized embeddings
  optuna.db                    # Phase 2: durable Optuna RDB (trial metrics only)
```
`meta_kb.parquet` is the `embedding_vector -> leaderboard` mapping you asked for; it
is the inner join of `train_embeddings.parquet` and `leaderboards.parquet` on
`dataset_id`, and is what Phase 3 retrieves against.

### 6.6 Leaderboard record schema (pydantic, one block per dataset)
```json
{
  "dataset_id": "uci_adult",
  "encoder_version": "d2v-v1",
  "embedding": [0.013, -0.221, "..."],
  "task_type": "classification",
  "n_rows": 32561,
  "n_cols": 14,
  "target_cardinality": 2,
  "primary_metric": "f1_macro",
  "leaderboard": [
    {
      "rank": 1,
      "model_name": "XGBClassifier",
      "hyperparameters": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.07},
      "metrics": {"accuracy": 0.91, "f1_macro": 0.88, "f1_weighted": 0.90},
      "n_trials": 50
    }
  ],
  "best_model": "XGBClassifier",
  "created_at": "2026-06-16T22:50:00Z"
}
```

### 6.7 Phase-2 entrypoint
`build_leaderboard_db.py -c <config> [--datasets <id,id,...>] [--resume] [-v]`
- Enumerates `(dataset_id, model_name)` units for the configured corpus, dispatches
  them across Ray, runs Optuna per unit, writes `leaderboards.parquet`, then joins
  with `train_embeddings.parquet` to (re)build `meta_kb.parquet` + `index.faiss`.
- `--resume` skips `(dataset, model)` units already present in `optuna.db`.
- `--datasets` restricts to a subset for smoke tests.

================================================================================
## PHASE 3 - WARM-START QUERY + VERIFY (test dataset -> models, then train & match)
================================================================================

### 7.1 Step A - retrieve the prior
`query.py -i <dataset.npz> -o <output_dir> [-k <top_k>] [--verify] [-v]`
loads `X_train,y_train,X_test,y_test` into `CommonData`/`DataBundle`, embeds with the
frozen encoder, searches `index.faiss` for the top-k nearest TRAINING datasets
(`same_task_only` so classification never mixes into regression), and builds the
ranked warm-start list. Each `ranked_models` entry carries the
`recommended_hyperparameters` that won on the nearest training datasets AND an
`expected_metric` (the similarity-weighted metric those neighbors achieved).

### 7.2 Step B - VERIFY on the test dataset (train and match accuracy)
When `--verify` is set, for each entry in `ranked_models` the query:
1. builds `DataBundle(common=CommonData(X_train,y_train,X_test,y_test),
   hyperparameters=recommended_hyperparameters)`,
2. `kit = MLKit(model_name, data, training_mode="full_train"); kit.train()`,
3. `y_pred = kit.test()`,
4. `achieved = compute_metrics(y_test, y_pred, task_type, model_name)`,
5. compares `achieved[primary_metric]` against the prior's `expected_metric` and
   records the gap,
6. MEMORY: `del kit; gc.collect(); torch.cuda.empty_cache()` (models are NOT saved).
The verify models run on the SAME Ray pool / GPU-fraction scheme as Phase 2 so the
suggested models train in parallel. PyTorch suggestions use the GPU; sklearn/xgboost
use CPU workers. This is the "train the model and match the accuracy with suggestion"
requirement.

### 7.3 Output `dataset_prior.json`
```json
{
  "query_dataset_id": "test_fraud",
  "encoder_version": "d2v-v1",
  "top_k": 5,
  "primary_metric": "f1_macro",
  "neighbors": [
    {"dataset_id": "uci_adult", "similarity": 0.93, "best_model": "XGBClassifier",
     "recommended_hyperparameters": {"...": "..."}, "metrics": {"f1_macro": 0.88}}
  ],
  "ranked_models": [
    {
      "model_name": "XGBClassifier",
      "score": 0.94,
      "recommended_hyperparameters": {"n_estimators": 600, "max_depth": 6},
      "expected_metric": 0.88,
      "verification": {
        "trained": true,
        "achieved_metric": 0.87,
        "achieved_full": {"accuracy": 0.90, "f1_macro": 0.87, "f1_weighted": 0.89},
        "delta_vs_expected": -0.01,
        "within_tolerance": true
      }
    }
  ],
  "verification_summary": {
    "tolerance": 0.05,
    "n_verified": 5,
    "n_within_tolerance": 4,
    "best_achieved": {"model_name": "XGBClassifier", "metric": 0.87},
    "mean_abs_delta": 0.018
  },
  "cold_start": false,
  "caveats": ["Embedding similarity does not guarantee identical column semantics."]
}
```
- `ranked_models` is the warm-start answer: top-5 to top-10 models AND the
  hyperparameters that worked on the nearest training datasets, produced by a
  similarity-weighted vote over neighbor leaderboards, restricted to
  `validate_model_name`. Count = `retrieval.n_recommended_models` (default 10, floor 5).
- `verification.*` is the achieved-vs-expected match from Step B. `within_tolerance`
  is `|achieved - expected| <= retrieval.verify_tolerance` (default 0.05).
- `verification_summary` rolls up the match across all suggestions so a caller can
  see at a glance whether the warm-start held on the real test data.
- `cold_start: true` when the store is empty or has no same-task neighbors; then
  `neighbors` / `ranked_models` are empty, verification is skipped, and the caller
  degrades gracefully.

## 8. CONTROLLABLES (config/config.yaml)
```yaml
encoder:
  embedding_dim: 64
  f_block: { hidden: 32, layers: 3, residual_blocks: 2 }
  g_block: { hidden: 32, layers: 3, residual_blocks: 2 }
  h_block: { hidden: 32, layers: 3 }
  standardize_inputs: true
  encoder_version: d2v-v1
training:                            # PHASE 1
  corpus_dir: null                   # REQUIRED; error if null
  device: cuda                       # cuda | cpu
  use_amp: true                      # mixed precision on GPU
  n_instances_sample: 256
  n_features_sample: 16
  n_classes_sample: 3
  pairs_per_batch: 64
  gamma: 1.0
  learning_rate: 0.001
  epochs: 10000
  es_patience: 16
  es_min_delta: 0.001
  checkpoint_every: 50
  random_state: 42
sweep:                               # PHASE 2
  corpus_dir: null                   # REQUIRED; error if null
  models: all                        # 'all' or explicit subset of EXPECTED_MODELS
  n_parallel: null                   # REQUIRED; Ray workers (size to the machine)
  n_trials_per_model: 50
  primary_metric: f1_macro           # classification; rmse/r2 for regression corpora
  leaderboard_top_n: 10
  same_task_only: true
  gpu_fraction_pytorch: 0.25         # GPU slice per concurrent PyTorch model
  optuna_sampler: tpe                # tpe | random  (hash-map dispatch)
  optuna_pruner: median             # median | none (hash-map dispatch)
  optuna_storage: sqlite:///store/optuna.db
  scratch_dir: store/scratch
  cleanup_interval_seconds: 3600     # hourly janitor: gc + empty_cache + temp delete
store:
  store_dir: null                    # REQUIRED; error if null
  faiss_metric: inner_product
  normalize_embeddings: true
retrieval:                           # PHASE 3
  top_k: 10
  n_recommended_models: 10           # floor 5
  same_task_only: true
  min_similarity: 0.0
  model_vote: similarity_weighted    # hash-map dispatch
  verify: true                       # train suggestions on the test set and match
  verify_tolerance: 0.05             # |achieved - expected| within-tolerance band
```
- `model_vote`, `optuna_sampler`, `optuna_pruner`, and per-family resources are all
  hash-map dispatch keys, never if-else ladders (CLAUDE.md #4/#23).
- Python binary + paths live in `config/config.ini` (`[python] PYTHON=...`,
  `[paths] model_library_root, config_yaml, store_dir, search_spaces_json`),
  one config.ini per project (CLAUDE.md #6/#7/#22). No path hardcoded in code.

## 9. PUBLIC API + CLI
Core class `MetaKnowledgeBase` (OOP, fully typed, imports at top):
```python
mkb = MetaKnowledgeBase(config_path="config/config.ini")
mkb.train_encoder() -> str                                   # PHASE 1, returns encoder_version
vector = mkb.encode(data: DataBundle) -> np.ndarray          # fixed-dim embedding
mkb.build_leaderboard_db(dataset_ids: list[str] | None) -> int   # PHASE 2, rows written
prior = mkb.query(data: DataBundle, top_k: int | None = None,
                  verify: bool | None = None) -> dict         # PHASE 3 (+ verify)
```
CLI scripts (argparse, `-v` verbose, no hardcoded arg defaults => None + error):
- `train_encoder.py        -c <config> [--resume] [-v]`        # PHASE 1
- `build_leaderboard_db.py -c <config> [--datasets ...] [--resume] [-v]`   # PHASE 2
- `query.py                -i <dataset.npz> -o <output_dir> [-k <top_k>] [--verify] [-v]`  # PHASE 3

## 10. PROJECT STRUCTURE
```
epic_4/dataset2Vec/
  config/
    config.ini             # python binary + paths
    config.yaml            # all controllables (Section 8)
    search_spaces.json     # per-model Optuna search spaces (Section 6.3)
  core/
    encoder.py             # Dataset2Vec f/g/h network (PyTorch, GPU), encode()
    sampling.py            # patch + pair sampling for Phase-1 training
    sweep.py               # Ray + Optuna leaderboard sweep, memory janitor (Phase 2)
    verify.py              # Phase-3 train-and-match runner (reuses sweep Ray pool)
    store.py               # parquet leaderboard DB + FAISS index + join view
    schema.py              # pydantic Record / Leaderboard / Prior models
  meta_knowledge_base.py   # MetaKnowledgeBase orchestrator
  train_encoder.py         # PHASE 1 entrypoint
  build_leaderboard_db.py  # PHASE 2 entrypoint
  query.py                 # PHASE 3 entrypoint (+ --verify)
  claude_scripts/          # standalone/test scripts (gitignored, CLAUDE.md #13)
  tests/
    test_meta_kb.py        # end-to-end runner (no pytest)
  docs/
    README.md              # git-tracked; other docs gitignored (CLAUDE.md #10)
  SPEC.md
```

## 11. OVERNIGHT RUN COMMANDS
Python binary comes from `config/config.ini [python] PYTHON`; below uses `$PYBIN` as a
stand-in. Logs go to `claude_outputs/` (CLAUDE.md #24); dirs created with `mkdir -p`.

```bash
# 0. one-time: create output/scratch dirs
mkdir -p claude_outputs store/scratch store/encoder/checkpoints

# 1. PHASE 1 - train the Dataset2Vec encoder on the GPU (resumable)
#    pick the GPU with: nvidia-smi
CUDA_VISIBLE_DEVICES=0 nohup $PYBIN train_encoder.py \
    -c config/config.ini --resume -v \
    > claude_outputs/phase1_encoder.log 2>&1 &
echo "encoder PID $!"

# watch progress
tail -f claude_outputs/phase1_encoder.log

# 2. PHASE 2 - parallel Optuna leaderboard DB (resumable, memory-bounded)
#    set sweep.n_parallel in config.yaml to fit the box, e.g.:
#      n_parallel = (physical CPU cores - 2); gpu_fraction_pytorch so that
#      n_parallel * gpu_fraction_pytorch <= number of GPUs.
#    Phase 2 can start as soon as encoder.pt exists (or run fully independently
#    and backfill embeddings).
CUDA_VISIBLE_DEVICES=0 nohup $PYBIN build_leaderboard_db.py \
    -c config/config.ini --resume -v \
    > claude_outputs/phase2_leaderboard.log 2>&1 &
echo "leaderboard PID $!"

# monitor: leaderboard progress + memory headroom
tail -f claude_outputs/phase2_leaderboard.log
watch -n 30 nvidia-smi          # GPU VRAM (should stay flat - models are freed)
watch -n 30 free -g             # system RAM

# smoke test on 2 datasets before committing to the full overnight run
$PYBIN build_leaderboard_db.py -c config/config.ini \
    --datasets uci_adult,uci_wine --resume -v

# 3. PHASE 3 - warm-start + VERIFY for a held-out TEST dataset:
#    retrieves the prior, then trains each suggested model on the test set and
#    matches achieved accuracy against the expected metric.
$PYBIN query.py -i path/to/test_dataset.npz -o claude_outputs/prior -k 5 --verify -v
cat claude_outputs/prior/dataset_prior.json   # ranked_models + verification_summary
```
Resumability: kill/restart is safe. Phase 1 resumes from the latest checkpoint;
Phase 2 skips `(dataset, model)` units already recorded in `store/optuna.db`. The
hourly janitor keeps VRAM/RAM flat across the overnight run and the Phase-3 verify.

## 12. ACCEPTANCE CRITERIA
1. PHASE 1: `train_encoder.py` trains on the GPU and saves a versioned encoder; two
   patches of the SAME source dataset have higher cosine similarity than patches of
   DIFFERENT datasets (the contrastive objective is actually learned).
2. PHASE 1: the encoder returns the SAME fixed length d for datasets of different
   (n, m, t); `train_embeddings.parquet` has one embedding per training dataset.
3. PHASE 2: `build_leaderboard_db.py` runs the model set under Optuna across Ray
   workers and writes `leaderboards.parquet`; every `model_name` is a valid
   `validate_model_name` entry; leaderboards are ranked by the primary metric.
4. PHASE 2 MEMORY: no model files are written by the sweep or the verify step; VRAM
   and RAM stay bounded across a multi-hour run; the hourly janitor clears CUDA cache
   and temp files.
5. PHASE 2: `--resume` does not recompute `(dataset, model)` units already in
   `optuna.db`; `meta_kb.parquet` is the correct join of embeddings and leaderboards.
6. PHASE 3 RETRIEVE: `query.py` on a held-out test dataset returns a schema-valid
   `dataset_prior.json` whose neighbors are sorted by descending similarity and whose
   `ranked_models` (models + hyperparameters) are all valid model names.
7. PHASE 3 VERIFY: with `--verify`, each suggested model is trained on the test
   dataset and its achieved metric is recorded with `delta_vs_expected` and
   `within_tolerance`; `verification_summary` reports `n_within_tolerance` and
   `mean_abs_delta`. The achieved metric of the top suggestion is within
   `verify_tolerance` of the expected metric on the acceptance corpus.
8. Cold start: querying an empty store returns `cold_start: true`, skips verification,
   and does not raise.
9. No MITRA container/metric/model-name logic is reimplemented; MLKit, DataBundle,
   MetricsResult, and validate_model_name are reused.

## 13. OPEN ITEMS / RISKS
1. Encoder versioning: embeddings are comparable only within one `encoder_version`.
   Every record stores its version; retraining the encoder requires re-embedding the
   store and re-running the join (migration script is a v2 item).
2. Phase-2 cost: running 60 models * N trials over the corpus is the expensive step;
   `sweep.models`, `n_trials_per_model`, and `leaderboard_top_n` bound it. The
   overnight run + `--resume` are the intended workflow.
3. GPU contention: PyTorch models share the GPU via `gpu_fraction_pytorch`; set it so
   concurrent GPU studies fit VRAM, else trials OOM. sklearn/xgboost stay on CPU.
4. Verify cost vs expected mismatch: the Phase-3 verify trains up to
   `n_recommended_models` models on the test set; a large `delta_vs_expected` is a
   SIGNAL (similar distribution != identical column semantics), reported as a caveat,
   not an error. Tolerance is configurable (`verify_tolerance`).
5. Input scaling: per-column standardization (`standardize_inputs`) so embeddings are
   not dominated by raw feature scale.
6. Task-type mixing: `same_task_only` keeps classification and regression separate in
   the sweep, retrieval, and verify.
7. Large datasets: encoding uses subsampled patches (`n_instances_sample`) so encoder
   memory is bounded; Phase-2/verify training on very large datasets is bounded by the
   model wrappers themselves (streaming ingestion is a v2 item).
