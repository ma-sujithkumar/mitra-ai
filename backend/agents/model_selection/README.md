# MITRA Model Selection

This implementation selects models **only** from the existing MLKit registry:

`model_library/ml_kit.py::MODEL_REGISTRY`

It does not maintain a second model-name allowlist and does not import PyTorch,
XGBoost, or scikit-learn merely to discover names. The catalog agent parses the
registry with `ast`, confirms every registry entry has defaults in
`model_library/config/config.yaml`, and classifies it from the wrapper import
module (`models.classifiers.*` or `models.regressors.*`).

## Agents

1. **ModelLibraryCatalogAgent** — discovers exact MLKit names and defaults.
2. **DatasetProfileAgent** — profiles metadata/features without raw-data access.
3. **LLMModelRankingAgent** — optional one-shot ranker; prompt contains the exact
   catalog and the response is treated as untrusted.
4. **DeterministicRankingAgent** — network-free fallback.
5. **ModelSelectionValidationAgent** — rejects unknown/wrong-task names.
6. **ModelSelectionOrchestratorAgent** — coordinates and atomically writes output.

The model library currently exposes 30 classifiers and 30 regressors. It exposes
no KMeans, DBSCAN, IsolationForest, or other unsupervised estimator, so an
`unsupervised` metadata request fails explicitly rather than inventing a model
outside the library.

## CLI

Run from the repository root:

```bash
python -m epic_3.model_selection.cli \
  --metadata epic_3/model_selection/fixtures/iris_metadata.json \
  --feature-selection epic_3/model_selection/fixtures/iris_feature_selection.json \
  --mini-data epic_3/model_selection/fixtures/iris_mini_data.csv \
  --model-library-root model_library \
  --output /tmp/model_config.json \
  --report /tmp/model_selection_report.json \
  --max-models 5
```

## Python API

```python
from epic_3.model_selection import select_models

selected = select_models(
    metadata_path=".mitra/session/metadata.json",
    feature_selection_path=".mitra/session/feature_selection.json",
    mini_data_path=".mitra/session/mini_data.csv",
    model_library_root="model_library",
    output_path=".mitra/session/model_config.json",
    report_path=".mitra/session/model_selection_report.json",
    max_models=5,
)
```

Each `model_config.json` entry includes the exact `model_name` needed by
`MLKit(model_name=...)`, its default configuration copied from the library,
priority, rationale, and provenance. `hp_space` is intentionally empty here:
Epic-4's HPT agent should own search-space construction instead of model
selection inventing ranges not present in the current model library.

## Optional LiteLLM integration

Pass any adapter with `complete(prompt: str) -> str` to `select_models` as
`llm_client`. The LLM receives the exact allowed registry names once. Unknown
names are discarded and deterministic candidates fill the remaining slots.

## Tests

```bash
pytest -q epic_3/model_selection/tests
```
