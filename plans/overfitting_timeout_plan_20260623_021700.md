# Plan: Overfitting Analysis Timeout Implementation

## Context
When executing multi-turn Judge evaluations, training of new candidate models completes, but the evaluation phase blocks inside `EvalRunner.run` if one of the model's overfitting checks hangs or takes very long (e.g. `PyTorchFCNNClassifier` training and 5-fold cross-validation). This prevents the Judge from moving forward even though validation scores are already computed and available.

To resolve this, we will introduce a configurable overfitting timeout and run the overfitting evaluations with a non-blocking timeout.

## Implementation Steps

### 1. Add Configuration Setting in `config.ini`
Add `OVERFITTING_TIMEOUT_SEC=120` to the `[pipeline]` section of the global [config.ini](file:///home/sujithma/mitra/config.ini).

### 2. Update `config_loader.py`
Modify [config_loader.py](file:///home/sujithma/mitra/backend/config_loader.py):
* Add `overfitting_timeout_sec: int` to `PipelineConfig`.
* Parse `OVERFITTING_TIMEOUT_SEC` from the `[pipeline]` section of `config.ini` with a default fallback of `120` seconds.

### 3. Update `EvalRunner` and `OverfittingRunner` in `eval_runner.py`
Modify [eval_runner.py](file:///home/sujithma/mitra/backend/orchestration/eval_runner.py):
* Update `EvalRunner.__init__` to accept `overfitting_timeout_sec: int = 120` and store it as `self.overfitting_timeout_sec`.
* Pass `timeout_sec=self.overfitting_timeout_sec` to `overfit_runner.run(...)`.
* Update `OverfittingRunner.run` signature to accept `timeout_sec: Optional[int] = None`.
* In `OverfittingRunner.run`, replace the standard `with ProcessPoolExecutor() as pool:` block with an unmanaged `ProcessPoolExecutor` instance, wrapping the execution in a `try...finally` block that shuts down the pool cleanly without blocking via `pool.shutdown(wait=False, cancel_futures=True)`.
* Use `as_completed` with `timeout=timeout_sec` to poll model overfitting futures. If a model times out:
  * Cancel its future.
  * Write a status message stating it timed out.
  * Return `None` for its overfitting directory so the Judge fallback is used.

### 4. Update Instantiations of `EvalRunner`
Update all places where `EvalRunner` is instantiated to pass `overfitting_timeout_sec`:
* In [run_pipeline.py](file:///home/sujithma/mitra/backend/orchestration/run_pipeline.py#L261).
* In [training_service.py](file:///home/sujithma/mitra/backend/services/training_service.py#L918).

## Verification
* Ensure all files compile and run successfully.
* Verify overfitting checks terminate gracefully when exceeding the timeout limit, allowing the pipeline/Judge to proceed.
