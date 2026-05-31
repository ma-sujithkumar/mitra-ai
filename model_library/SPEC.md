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


