# MLKit Interface Schema

This document is the authoritative reference for all MLKit interfaces, arguments, and model-specific configuration keys. It is the primary contract for LLM agents using this library.

---

## 1. DataBundle

The single input container passed to `MLKit`.

```python
from core.data_bundle import CommonData, DataBundle

common = CommonData(
    X_train=X_train,   # np.ndarray, shape (n_train, n_features), dtype float32
    y_train=y_train,   # np.ndarray, shape (n_train,)
    X_test=X_test,     # np.ndarray, shape (n_test, n_features), dtype float32
    y_test=y_test,     # np.ndarray, shape (n_test,)
)
data = DataBundle(common=common, hyperparameters={})
```

### CommonData fields

| Field | Type | Description |
|-------|------|-------------|
| X_train | np.ndarray (2D) | Training features |
| y_train | np.ndarray (1D) | Training labels/targets |
| X_test | np.ndarray (2D) | Test features (same n_features as X_train) |
| y_test | np.ndarray (1D) | Test labels/targets |

### DataBundle.hyperparameters

Optional dict of per-call overrides. Keys must match the model's `config.yaml` parameter names exactly. These are merged on top of `config.yaml` defaults at model instantiation.

**Example — override XGBClassifier learning rate:**
```python
DataBundle(common=common, hyperparameters={"learning_rate": 0.05, "n_estimators": 200})
```

---

## 2. MLKit Constructor

```python
MLKit(
    model_name: str,
    data: DataBundle,
    preloaded_model: Optional[str] = None,
    training_mode: str = "fine_tune",
)
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| model_name | str | Yes | Exact name from the 60-model registry (see Section 4) |
| data | DataBundle | Yes | Input data and optional hyperparameter overrides |
| preloaded_model | str | No | Path to a .pkl file to load before training |
| training_mode | str | No | `"fine_tune"` (default) or `"full_train"` |

### training_mode behavior

| Value | Behavior when preloaded_model is set |
|-------|--------------------------------------|
| `"fine_tune"` | Loads the model, then calls train() which continues from loaded weights |
| `"full_train"` | Loads the model, then reinitializes all weights before fitting from scratch |

If `preloaded_model` is None, `training_mode` has no effect (always trains from scratch).

---

## 3. MLKit Methods

### train()
```python
kit.train() -> None
```
Fits the model on `data.common.X_train` / `data.common.y_train`. Shape handling (2D enforcement, PyTorch tensor conversion, CNN reshaping) is done internally.

### test()
```python
y_pred = kit.test() -> np.ndarray
```
Runs inference on `data.common.X_test`. Returns raw predictions:
- Classification: integer class labels, shape (n_test,)
- Regression: float values, shape (n_test,)

Pass `y_pred` to `metrics.evaluators.compute_metrics()` to get a `MetricsResult`.

### save(path)
```python
kit.save("path/to/model.pkl") -> None
```
Pickles the trained model. PyTorch models are moved to CPU before pickling (Ray-safe).

### MLKit.load_from(model_name, path, data)
```python
kit = MLKit.load_from(
    model_name="XGBClassifier",
    path="model.pkl",
    data=data_bundle,
)
```
Convenience class method. Equivalent to `MLKit(model_name, data, preloaded_model=path)`.

---

## 4. Model Registry (60 Models)

### Classifiers (30)

| Model Name | Source | Notes |
|------------|--------|-------|
| LogisticRegression | sklearn | |
| RidgeClassifier | sklearn | |
| SGDClassifier | sklearn | |
| SVC | sklearn | probability=True enabled |
| LinearSVC | sklearn | |
| NuSVC | sklearn | probability=True enabled |
| DecisionTreeClassifier | sklearn | |
| RandomForestClassifier | sklearn | |
| ExtraTreesClassifier | sklearn | |
| GradientBoostingClassifier | sklearn | |
| AdaBoostClassifier | sklearn | |
| HistGradientBoostingClassifier | sklearn | |
| KNeighborsClassifier | sklearn | |
| RadiusNeighborsClassifier | sklearn | Falls back to majority class if no neighbor within radius |
| GaussianNB | sklearn | |
| MultinomialNB | sklearn | Negative inputs clipped to 0 automatically |
| ComplementNB | sklearn | Negative inputs clipped to 0 automatically |
| BernoulliNB | sklearn | |
| CategoricalNB | sklearn | Negative inputs clipped to 0 and cast to int automatically |
| MLPClassifier | sklearn | |
| PassiveAggressiveClassifier | sklearn | |
| QuadraticDiscriminantAnalysis | sklearn | |
| LinearDiscriminantAnalysis | sklearn | |
| BaggingClassifier | sklearn | base_estimator configurable in config.yaml |
| DummyClassifier | sklearn | |
| NearestCentroid | sklearn | |
| CalibratedClassifierCV | sklearn | base_estimator configurable in config.yaml |
| XGBClassifier | xgboost | |
| PyTorchFCNNClassifier | pytorch | Fully-connected; architecture from config.yaml |
| PyTorchCNNClassifier | pytorch | CNN; conv_dim=2 for images, conv_dim=1 for 1D signals |

### Regressors (30)

| Model Name | Source | Notes |
|------------|--------|-------|
| LinearRegression | sklearn | |
| Ridge | sklearn | |
| Lasso | sklearn | |
| ElasticNet | sklearn | |
| Lars | sklearn | |
| LassoLars | sklearn | |
| OrthogonalMatchingPursuit | sklearn | |
| BayesianRidge | sklearn | |
| ARDRegression | sklearn | |
| SGDRegressor | sklearn | |
| PassiveAggressiveRegressor | sklearn | |
| SVR | sklearn | |
| NuSVR | sklearn | |
| LinearSVR | sklearn | |
| KNeighborsRegressor | sklearn | |
| RadiusNeighborsRegressor | sklearn | Falls back to training mean if no neighbor within radius |
| DecisionTreeRegressor | sklearn | |
| RandomForestRegressor | sklearn | |
| ExtraTreesRegressor | sklearn | |
| GradientBoostingRegressor | sklearn | |
| AdaBoostRegressor | sklearn | |
| HistGradientBoostingRegressor | sklearn | |
| MLPRegressor | sklearn | |
| BaggingRegressor | sklearn | base_estimator configurable in config.yaml |
| XGBRegressor | xgboost | |
| PyTorchFCNNRegressor | pytorch | Fully-connected; architecture from config.yaml |
| PyTorchCNNRegressor | pytorch | CNN; conv_dim=1 for tabular, conv_dim=2 for images |
| DummyRegressor | sklearn | |
| HuberRegressor | sklearn | |
| TheilSenRegressor | sklearn | n_subsamples capped to 50 in config.yaml to prevent O(n^2) hangs |

---

## 5. config.yaml Keys Per Model Type

### PyTorch FCNN (Classifier and Regressor)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| hidden_layers | list[int] | [512,256,128] / [128,64,32] | Neuron counts per hidden layer |
| activation | str | relu | Activation: relu, tanh, sigmoid, leaky_relu, elu |
| learning_rate | float | 0.001 | Adam optimizer LR |
| epochs | int | 20 / 30 | Training epochs |
| batch_size | int | 64 / 32 | DataLoader batch size |
| dropout | float | 0.3 / 0.2 | Dropout probability |
| num_classes | int | 10 | Classifier only: number of output classes |
| weight_decay | float | 1e-4 | Adam weight decay |

### PyTorch CNN (Classifier and Regressor)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| conv_channels | list[int] | [32,64] / [16,32] | Output channels per conv layer |
| kernel_size | int | 3 | Conv kernel size (padding=1 applied) |
| fc_layers | list[int] | [256,128] / [64,32] | FC layer sizes after conv |
| conv_dim | int | 2 / 1 | 2 = Conv2d (images), 1 = Conv1d (tabular/1D signal) |
| learning_rate | float | 0.001 | Adam LR |
| epochs | int | 10 / 20 | Training epochs |
| batch_size | int | 64 / 32 | DataLoader batch size |
| dropout | float | 0.3 / 0.2 | Dropout after FC layers |
| num_classes | int | 10 | Classifier only |
| weight_decay | float | 1e-4 | Adam weight decay |

**conv_dim notes:**
- `conv_dim=2`: Input is reshaped from (N, features) to (N, 1, H, W) where H = W = sqrt(features). Features are zero-padded if not a perfect square.
- `conv_dim=1`: Input is reshaped from (N, features) to (N, 1, features).

### Meta-estimators (BaggingClassifier, BaggingRegressor, CalibratedClassifierCV)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| base_estimator | str | "DecisionTreeClassifier" / "LinearSVC" | String name of the base estimator class |

Valid `base_estimator` values for classifiers: `DecisionTreeClassifier`, `LinearSVC`, `SVC`, `LogisticRegression`, `SGDClassifier`

Valid `base_estimator` values for regressors: `DecisionTreeRegressor`, `SVR`, `LinearSVR`, `Ridge`, `LinearRegression`

---

## 6. MetricsResult

Returned by `metrics.evaluators.compute_metrics()`.

```python
from metrics.evaluators import compute_metrics, MetricsResult

result: MetricsResult = compute_metrics(
    y_true=y_test,
    y_pred=y_pred,
    task_type="classification",  # or "regression"
    model_name="XGBClassifier",
)
print(result)
```

### Classification fields

| Field | Description |
|-------|-------------|
| accuracy | Overall accuracy |
| f1_macro | F1 score (macro average) |
| f1_weighted | F1 score (weighted average) |
| precision_macro | Precision (macro average) |
| recall_macro | Recall (macro average) |

### Regression fields

| Field | Description |
|-------|-------------|
| mse | Mean Squared Error |
| rmse | Root Mean Squared Error |
| mae | Mean Absolute Error |
| r2 | R-squared coefficient of determination |

---

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid model_name | `ValueError` with "Did you mean: [...]?" suggestions |
| X_train is not 2D | `ValueError` with shape details |
| X_train / y_train sample count mismatch | `ValueError` with counts |
| config.yaml missing a model key | `ConfigValidationError` with missing key list |
| RadiusNeighbors no neighbors in radius | Warning logged; fallback to majority class / training mean |
| MultinomialNB / CategoricalNB negative input | Warning logged; input clipped to 0 |
