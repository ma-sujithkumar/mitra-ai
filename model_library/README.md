# MLKit — Universal ML/DL Wrapper Library

A unified Python library wrapping 60 ML/DL models (30 classifiers + 30 regressors) behind a single `MLKit` class. Designed for LLM agent use: instantiate by model name, pass a `DataBundle`, call `.train()` and `.test()`.

Full interface documentation: [docs/schema.md](docs/schema.md)

---

## Quick Start

```python
from ml_kit import MLKit
from core.data_bundle import CommonData, DataBundle
from metrics.evaluators import compute_metrics
import numpy as np

common = CommonData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
data = DataBundle(common=common)

kit = MLKit(model_name="XGBClassifier", data=data)
kit.train()
y_pred = kit.test()

result = compute_metrics(y_true=y_test, y_pred=y_pred, task_type="classification", model_name="XGBClassifier")
print(result)
```

---

## Installation

```bash
pip install scikit-learn torch torchvision xgboost pyyaml ray pydantic
```

---

## All 60 Model Names

Pass any of these strings as `model_name` to `MLKit`.

### Classifiers (30)

```
LogisticRegression        RidgeClassifier           SGDClassifier
SVC                       LinearSVC                 NuSVC
DecisionTreeClassifier    RandomForestClassifier    ExtraTreesClassifier
GradientBoostingClassifier AdaBoostClassifier       HistGradientBoostingClassifier
KNeighborsClassifier      RadiusNeighborsClassifier GaussianNB
MultinomialNB             ComplementNB              BernoulliNB
CategoricalNB             MLPClassifier             PassiveAggressiveClassifier
QuadraticDiscriminantAnalysis LinearDiscriminantAnalysis BaggingClassifier
DummyClassifier           NearestCentroid           CalibratedClassifierCV
XGBClassifier             PyTorchFCNNClassifier     PyTorchCNNClassifier
```

### Regressors (30)

```
LinearRegression          Ridge                     Lasso
ElasticNet                Lars                      LassoLars
OrthogonalMatchingPursuit BayesianRidge             ARDRegression
SGDRegressor              PassiveAggressiveRegressor SVR
NuSVR                     LinearSVR                 KNeighborsRegressor
RadiusNeighborsRegressor  DecisionTreeRegressor     RandomForestRegressor
ExtraTreesRegressor       GradientBoostingRegressor AdaBoostRegressor
HistGradientBoostingRegressor MLPRegressor          BaggingRegressor
XGBRegressor              PyTorchFCNNRegressor      PyTorchCNNRegressor
DummyRegressor            HuberRegressor            TheilSenRegressor
```

---

## Project Structure

```
model_library/
+-- config/
|   +-- config.ini          Python path + config.yaml path
|   +-- config.yaml         All 60 model hyperparameter dicts
+-- core/
|   +-- data_bundle.py      DataBundle and CommonData dataclasses
|   +-- validators.py       Pydantic input validation
|   +-- config_loader.py    YAML loader + schema validator
+-- models/
|   +-- base.py             Abstract BaseModel
|   +-- classifiers/        30 classifier wrappers
|   +-- regressors/         30 regressor wrappers
+-- metrics/
|   +-- evaluators.py       MetricsResult + compute_metrics()
+-- tests/
|   +-- test_mnist.py       Run all 30 classifiers on MNIST
|   +-- test_house_price.py Run all 30 regressors on California Housing
+-- docs/
|   +-- schema.md           Full interface and args documentation
+-- ml_kit.py               Main MLKit class
+-- README.md               This file
```

---

## Ray Usage

`MLKit` instances are fully pickle-serializable. PyTorch models are moved to CPU before pickling.

```python
import ray
from ml_kit import MLKit
from core.data_bundle import CommonData, DataBundle

@ray.remote
def train_and_test(model_name, X_train, y_train, X_test, y_test):
    common = CommonData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    data = DataBundle(common=common)
    kit = MLKit(model_name=model_name, data=data)
    kit.train()
    return kit.test()

ray.init()
futures = [train_and_test.remote(name, X_train, y_train, X_test, y_test) for name in model_names]
results = ray.get(futures)
```

---

## Running Tests

```bash
# All 30 classifiers on MNIST
python tests/test_mnist.py

# All 30 regressors on California Housing
python tests/test_house_price.py

# With verbose / debug logging
python tests/test_mnist.py -v
python tests/test_house_price.py -v
```

---

See [docs/schema.md](docs/schema.md) for the full interface specification including all config.yaml keys, DataBundle structure, and error handling behavior.
