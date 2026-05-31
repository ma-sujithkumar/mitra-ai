# SPEC: Universal ML/DL Wrapper Library (MLKit)

## 1. GOAL
To create a unified, robust, and highly modular Python library/codebase that encapsulates the implementation of standard machine learning (ML) and deep learning (DL) models. This library provides a single point of entry for instantiating, training, and testing models without internal data processing overhead.

## 2. PACKAGES TO BE USED
- scikit-learn
- pytorch
- xgboost
- pyyaml (for configurations)
- ray (for distributed training)

## 3. LIST OF ML MODELS TO BE IMPLEMENTED
-(Total: 60 Models - 30 Classifiers, 30 Regressors)-

### Classification Models (30)
1.  LogisticRegression
2.  RidgeClassifier
3.  SGDClassifier
4.  SVC
5.  LinearSVC
6.  NuSVC
7.  DecisionTreeClassifier
8.  RandomForestClassifier
9.  ExtraTreesClassifier
10. GradientBoostingClassifier
11. AdaBoostClassifier
12. HistGradientBoostingClassifier
13. KNeighborsClassifier
14. RadiusNeighborsClassifier
15. GaussianNB
16. MultinomialNB
17. ComplementNB
18. BernoulliNB
19. CategoricalNB
20. MLPClassifier
21. PassiveAggressiveClassifier
22. QuadraticDiscriminantAnalysis
23. LinearDiscriminantAnalysis
24. BaggingClassifier
25. XGBClassifier
26. PyTorchFCNNClassifier
27. PyTorchCNNClassifier
28. DummyClassifier
29. NearestCentroid
30. CalibratedClassifierCV

### Regression Models (30)
31. LinearRegression
32. Ridge
33. Lasso
34. ElasticNet
35. Lars
36. LassoLars
37. OrthogonalMatchingPursuit
38. BayesianRidge
39. ARDRegression
40. SGDRegressor
41. PassiveAggressiveRegressor
42. SVR
43. NuSVR
44. LinearSVR
45. KNeighborsRegressor
46. RadiusNeighborsRegressor
47. DecisionTreeRegressor
48. RandomForestRegressor
49. ExtraTreesRegressor
50. GradientBoostingRegressor
51. AdaBoostRegressor
52. HistGradientBoostingRegressor
53. MLPRegressor
54. BaggingRegressor
55. XGBRegressor
56. PyTorchFCNNRegressor
57. PyTorchCNNRegressor
58. DummyRegressor
59. HuberRegressor
60. TheilSenRegressor

## 4. APPLICATION
1.  **Agentic Instantiation:** This library is going to be used by an LLM Agent which will instantiate any ML model from this library by name and pass the prepared data to .train() and .test().
2.  **Ray Integration:** The end-application will use RAY to run the training using this code, meaning the main class should be serializable and adaptable to parallel worker execution.

## 5. TEMPLATE
Give me one final class - MLKit()

**Example usage:**

from ml_kit import MLKit

# data contains X_train, y_train, X_test, y_test
model = MLKit(model_name="XGBClassifier", data=data, preloaded_model="model.pkl")
model.train()
metrics = model.test()



## 6. CONSTRAINTS

1. **Python Environment:** Must use #!/home/sujithma/venv/bin/python.
2. **Hyperparameters:**
- Keep the hyperparameters of all the ML/DL models to their default values.
- If you do not find any obvious hyperparameter (e.g., PyTorch architecture specifics), assume a sensible baseline but make sure to parameterize them.
- Put all the default/parameterized values under config/config.yaml. **No hardcoding is allowed in the code.**


3. **Scope of Operations:** Need not do any data splitting, feature selection, or scaling. The task is purely to implement the wrappers for the ML models using the already available libraries.

## 7. OUTPUTS & PROJECT STRUCTURE

- config/config.yaml: Contains hyperparameter dictionaries for all 60 models.
- models/classifiers/:
- sklearn_classifiers.py
- pytorch_classifiers.py
- xgboost_classifiers.py


- models/regressors/:
- sklearn_regressors.py
- pytorch_regressors.py
- xgboost_regressors.py


- metrics/evaluators.py: Computes metrics for classification and regression.
- tests/test_mnist.py: Tests to run using MNIST for classification.
- tests/test_house_price.py: Tests to run using House Price Prediction for regression.
- ml_kit.py: The main script which orchestrates the classes, gives access to each model, and runs train() and test().
- README.md -> a visiting card for any agent to understand the library and harness it.

## 8. ACCEPTANCE CRITERIA

1. Run all 60 models sequentially (or distributed) using MNIST for classification and House Price Prediction for regression, and report all standard classification/regression metrics.
2. It should run end-to-end without any errors. Shape mismatches (e.g., for 1D vs 2D inputs in PyTorch vs Sklearn) must be handled internally by the wrappers before fitting.


------------------------------------------------------------------------------------------------------------------------------------------------------------

## Additional Improvements:
1. Enforce the inputs schema, write a schema.md which will explain the interface and args clearly and link it in README.md
2. In every .py, Put this at the top
```
@Authored by <model_name>, Co-Authored by Sujith M A, Created <>, Last Updated <>'
```


------------------------------------------------------------------------------------------------------------------------------------------------------------
## AI Review and Sujith M A's answer

 Ambiguities \u2014 Need Your Answers Before Planning
  
  1. data parameter contract \u2014 what type/shape?
  The spec shows data=data but never defines its structure. Is it a plain dict with keys X_train, y_train, X_test, y_test? A dataclass? A NamedTuple? What dtypes \u2014 numpy arrays, pandas
  DataFrames, or torch Tensors? This is the most critical ambiguity since every model depends on it.
Ans: it's a dictionary which has all the args and hyperparameters, keep nested dict, common: , hyperparameters: 

  2. preloaded_model behavior \u2014 does train() still run?
  If a path to model.pkl is given, should .train() skip training entirely, fine-tune the loaded model, or raise? The spec is silent. This has silent failure potential.
Ans: take a type argument as full_train, fine_tune, default it to fine_tune
 

  3. What does test() return exactly?
  The acceptance criteria says "report all standard metrics" but never lists them. Should test() return a dict? What keys \u2014 accuracy, f1, auc, mse, r2? Are multi-class F1 scores
  macro/micro/weighted? This affects the evaluators.py design completely.
Ans: test() should run the inference and return y - that's it. Keep the metrics in metrics.py
  
  4. PyTorchCNNClassifier / PyTorchCNNRegressor \u2014 what input shape is expected?
  CNNs expect spatial/image input. PyTorchCNNClassifier on MNIST (28×28) is reasonable. But PyTorchCNNRegressor is listed for house price prediction \u2014 tabular data has no spatial
  structure. Should the CNN treat the 1D feature vector as a 1D signal (Conv1D)? Or is a CNN for regression intentionally included even if architecturally unusual?
Ans: Take whether to use Conv2D, or 1D as an arg. All the model arguments must be well-documented in schema.md and linked in README.md


  5. Dataset sources for test files
  test_mnist.py and test_house_price.py need data. Where do these come from \u2014 torchvision.datasets, sklearn.datasets.fetch_openml, a local file, or auto-downloaded? The spec says "no
  data splitting" is the library's job, but the test files must split it themselves \u2014 is that correct?
Ans: torchvision.datasets, sklearn.datasets.fetch_openml. test_suite can split


  6. Are the test files pytest tests or standalone runner scripts?
  The word "tests" and the .py convention imply pytest, but the acceptance criteria says "run all 60 models sequentially (or distributed) and report metrics" \u2014 that reads more like a
  runner script. Which is intended?
Ans: No pytest. 


  7. <model_name> in the authorship header \u2014 which model?
  @Authored by <model_name> \u2014 is <model_name> the AI model that generates the code (e.g., Claude Sonnet 4.6)? The Created <> and Last Updated <> are blank placeholders \u2014 should these
  be auto-filled at generation time?
Ans: yes

  8. Meta-estimators \u2014 what base estimator?
  CalibratedClassifierCV, BaggingClassifier, and BaggingRegressor wrap other estimators. What base estimator should they default to? The spec says "keep defaults," but sklearn's
  default for BaggingClassifier base is DecisionTreeClassifier \u2014 is that acceptable, and should this be configurable in config.yaml?
Ans: Should be given in config.yaml. Also, All the model arguments must be well-documented in schema.md and linked in README.md
  

  ---
  Risks \u2014 Will Break Without Mitigation

  1. PyTorchCNNRegressor on tabular data
  A 2D CNN on house price features (e.g., 13 flat features) will crash. The shape-mismatch handling requirement in section 8 covers sklearn vs. PyTorch, but doesn't address CNN vs.
  tabular input. Needs an explicit architectural decision (Conv1D? Treat as FCN?).
Ans: Should be given as an arg. Should support all the dimensions. All the model arguments must be well-documented in schema.md and linked in README.md
  

  2. RadiusNeighborsClassifier / RadiusNeighborsRegressor
  These raise ValueError at inference time if no training samples fall within the radius for a given test point. This is a known sklearn pitfall that will break the end-to-end run
  requirement.
Ans: Add try-except and log the error properly in tests 

  3. MultinomialNB, ComplementNB, CategoricalNB
  Require strictly non-negative input. MNIST pixels (0\u2013255) are fine, but if the test suite ever normalizes features to zero-mean, these will crash silently.
Ans: Add try-except and log the error properly in tests   

  4. TheilSenRegressor
  Has O(n² ·  n_subsamples) complexity. On even a modest house price dataset, this can time out unless n_subsamples is capped in config.yaml.
Ans: Cap it. default it to very small size even in config.yaml

  5. Ray serialization of PyTorch models
  PyTorch models that use GPU tensors (CUDA) cannot be trivially pickled. The spec says "should be serializable," but if a worker uses a different CUDA device than the main process,
  deserialization will fail. Needs a CPU-safe serialization path.
Ans: Yes, should be picke-able

  6. No error handling specified for invalid model_name
  An LLM agent is the primary caller (section 4.1). If the agent passes a misspelled model name, there is no spec for the error response \u2014 silent None, ValueError, or a typed exception
  with a "did you mean X?" message.
Ans: Yes, Also, Create a input validation layer using Pydantic. A list of ML models should be available in README.md

  7. config.yaml as the sole source of truth \u2014 fragile bootstrap
  The constraint "no hardcoding" means PyTorch training loop parameters (learning rate, epochs, batch size, hidden layer sizes) must all live in YAML. If config.yaml is missing a key,
  every model using that key will throw a KeyError. A schema validator for the config is not in scope but is a real risk.
Ans: Go for it. implement it.

  ---
  Suggested Improvements (Non-blocking) 

  1. Define a DataBundle dataclass \u2014 makes the data contract explicit and enables IDE/agent autocomplete.
  2. Return a typed MetricsResult from test() instead of a raw dict \u2014 prevents key errors downstream in the agent.
  3. Add a model.save(path) / MLKit.load(path) API \u2014 the preloaded_model param implies this need but doesn't expose a clean save path.
  4. README.md as machine-readable YAML front-matter \u2014 since the primary consumer is an LLM agent, a structured header (model names, input schema, output schema) is more useful than
  prose.
Ans: ALL APPROVED except 4. Apply it for schema.md and link it in README.md

------

