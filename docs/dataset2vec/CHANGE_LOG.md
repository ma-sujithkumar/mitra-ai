# Dataset2Vec Change Log

---

## 2026-06-17

### Fix: Phase 2 OOM — chunked dataset dispatch in LeaderboardSweep.run()

**File:** `d2v_core/sweep.py` — `LeaderboardSweep.run()`

**Problem:** All 2420 units (121 datasets x 20 models) were dispatched as Ray futures before any results were collected. This placed all 121 `CommonData` arrays into Ray's plasma object store simultaneously, consuming ~13 GB and killing the node with OOM.

**Fix:** Process one dataset at a time.
- Group pending `(dataset_id, model_name)` units by `dataset_id`.
- For each dataset: call `ray.put(common)` once — all 20 model workers share the same plasma ref instead of getting N independent copies.
- Dispatch that dataset's 20 model tasks, call `ray.get()` to collect all results, write the leaderboard record, then `del common_ref` to free the plasma entry.
- Move to the next dataset. Peak plasma usage is now bounded to 1 dataset at a time.

---

### Fix: Individual trial failures crash the entire sweep (NuSVC infeasible nu)

**File:** `d2v_core/sweep.py` — `run_optuna_study()`, `_run_sweep_unit_remote()`, `LeaderboardSweep.run()`

**Problem:** `NuSVC` raises `ValueError: specified nu is infeasible` when the `nu` hyperparameter violates the class-distribution constraint for a given dataset. Optuna let the exception propagate through `study.optimize()`, crashing the Ray task and surfacing as `RayTaskError` in the driver, stopping the entire sweep.

**Fix (3 parts):**

1. `run_optuna_study`: pass `catch=(Exception,)` to `study.optimize()` so any per-trial sklearn/torch error (infeasible params, singular matrix, convergence failure) marks that trial as FAIL and Optuna continues to the next trial.

2. `run_optuna_study`: count ALL trial states (COMPLETE + FAIL + PRUNED) toward the `n_trials` budget — not just COMPLETE — so models that always fail do not loop forever trying to accumulate `n_trials` successful results.

3. `run_optuna_study`: after `study.optimize()`, check if any COMPLETE trials exist. If not, log a warning and return `None` instead of crashing on `study.best_trial`.

4. `LeaderboardSweep.run()`: filter out `None` results (models with zero successful trials) with a warning instead of crashing. If every model for a dataset returns `None`, log a warning and skip writing a leaderboard record for that dataset.

---

### Optimisation: Skip O(n^2) models on large datasets

**Files:** `config/config.yaml`, `d2v_core/sweep.py` — `LeaderboardSweep.run()`

**Problem:** Kernel SVMs (`SVC`, `NuSVC`) use libsvm which is O(n^2) to O(n^3) in training. On datasets with >10K rows (e.g. miniboone at 130K rows), a single trial can take 30–120 minutes, making 50 trials per model infeasible.

**Fix:** Added 3 new config knobs under `sweep:` in `config/config.yaml`:

```yaml
large_dataset_row_threshold: 10000
large_dataset_skip_models:
  - SVC
  - NuSVC
study_timeout_seconds: 600
```

In `LeaderboardSweep.run()`, before dispatching model tasks for a dataset:
- Compute `n_rows_total = X_train rows + X_test rows`.
- If `n_rows_total >= large_dataset_row_threshold`, log a warning and remove any model in `large_dataset_skip_models` from that dataset's dispatch list.

Log output when triggered:
```
=> dataset_id='miniboone' has 130064 rows (>= 10000 threshold): skipping O(n^2) models ['SVC', 'NuSVC'].
```

---

### Optimisation: Per-study wall-time timeout

**Files:** `config/config.yaml`, `d2v_core/sweep.py` — `run_optuna_study()`, `_run_sweep_unit_remote()`

**Problem:** Slow models (GradientBoosting, MLP, tree ensembles) on large datasets could run for hours on a single `(dataset, model)` study even after the O(n^2) skip, blocking the sweep.

**Fix:** Pass `timeout=study_timeout_seconds` to `study.optimize()`. After the wall-time cap, Optuna stops issuing new trials and returns whatever COMPLETE trials it has. If zero COMPLETE trials exist after timeout, `run_optuna_study` returns `None` (handled by the filter above).

`study_timeout_seconds` flows from `config.yaml -> sweep_config -> LeaderboardSweep.run() -> _run_sweep_unit_remote -> run_optuna_study`.

---

## Estimated Phase 2 Runtime (post-fixes)

| Segment | Estimate |
|---|---|
| 112 small datasets (<=10K rows, n_parallel=15) | 1.5 – 3 hours |
| 9 large datasets (>10K rows, SVC/NuSVC skipped, 600s timeout per study) | 3 – 8 hours |
| **Total** | **~5 – 11 hours** |
