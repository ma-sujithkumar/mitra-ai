# Unsupervised feature engineering (no target column)

Timestamp: 20260621_190426
Branch: dev

## Goal
When a run's metadata is `problem_type: unsupervised` (no target column), feature
engineering should still run — executing only the steps that do NOT involve a
target column — instead of failing with `TARGET_REQUIRED`.

## Why it currently fails
`/api/feature-engineering` -> `_resolve_target_column()` finds no target (request,
metadata.target_col/target_column, output_cols all empty for unsupervised) and
raises 422 TARGET_REQUIRED. The FE orchestrator also assumes a target
(`df[target_column]`, `df.drop(columns=[target_column])`, task inference) and two
pipeline steps consume the target.

## Target-dependent surface (everything else is target-free)
- `orchestrator.run()` start: `target = df[target_column]`, `features = df.drop(target)`, `_infer_task`.
- Pipeline steps: `compute_feature_stats` (feature_stats.py:131 `state.target`) and
  `select_features` (selector.py:116 `state.target`). All other steps (profile,
  infer_types, missing, outliers, encode, create, scale, validate, report) are
  feature-only.
- `pipeline_prep`: `_split_dataset(target_column)` and `_run_model_selection` are
  supervised-only; `_run_d2v_query` already skips non-classification.

## Changes
1. **state.py** — `target: pd.Series | None = None`, `target_column: str | None = None`.
2. **orchestrator __init__** — accept `task == "unsupervised"`; allow `target_column=None`.
3. **orchestrator.run()** — if unsupervised (task=="unsupervised" or target_column is
   None): `target=None`, `features=df` (keep all columns), `resolved_task="unsupervised"`;
   build PipelineState with target=None. Set `state.selected_columns = all feature
   columns`, `selection_method="unsupervised_all"`.
4. **orchestrator._run_pipeline** — when `state.task == "unsupervised"`, drop the
   `compute_feature_stats` and `select_features` steps from the ordered list.
5. **pipeline_prep.run() / _run_feature_engineering** — accept `target_column:
   str | None`; pass through. When resolved_task == "unsupervised": write
   feature_selection.json (all columns), skip split + model selection, return the
   feature artifact path (no model_config.json for unsupervised). d2v already skips.
6. **feature_engineering.py starter** — read `problem_type` from metadata.json;
   `is_unsupervised = problem_type == "unsupervised"`. Only raise TARGET_REQUIRED
   when not unsupervised. Pass `target_column` (possibly None) through
   `_run_pipeline_prep` (signature -> `str | None`).

## Out of scope
Unsupervised model selection / training. This change makes FE produce
`engineered_dataset.csv` + `feature_artifact.json` for unsupervised runs and stops
there (per user: "only the feature engineering steps which do not involve target").

## Tests
- Orchestrator unsupervised run on a small CSV with no target -> produces
  engineered_dataset.csv, feature_artifact.json with task=="unsupervised",
  selected_columns == all input columns, no crash; target steps skipped.
- FE starter does not 422 when metadata problem_type==unsupervised and no target.
