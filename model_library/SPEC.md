## GOAL:
To create a library/kit/codebase which contains the implementation of all the machine learning models and deep learning models using standard libariries

## PACKAGES TO BE USED:
1. sklearn
2. pytorch

## LIST OF ML MODELS TO BE IMPLEMENTED
Note: All classification and regression models.
1.
2.
3.
4.
5.
6.
..
50. 

## APPLICATION
1. This library is going to be used by an LLM Agent which should instantiate any ML model from this library and pass the data to train(), test()
2. The end-application will use RAY to run the training using this code. 

## TEMPLATE
Give me one final class - MLkit()
E.g. usage
model = MLKit(model_name, data, preloaded_model=<pkl> file)
model.train()
model.test()

## CONSTRAINTS
1. python - /home/sujithma/venv/bin/python
2. Keep the hyperparameters of all the ML models to their default values.
2.1. If you do not find any hyperparameter, assume it to any value but make sure to parametrize them.
2.2. Put all the default values/parameterized values under config/config.yaml. No hardcoding is allowed in the code.
3. Need not do any split, feature selection and so on. Task is just to implement the ML models, that too using the already available libraries.

## OUTPUTS
config/config.yaml
models/<model_name.py> (Need classifier, regressors seperately)
ml_kit.py (Script which orchestrates, gives access to each model and runs train, test)
tests/ (Tests to run using MNIST for classification )

## ACCEPTANCE CRITERIA
1. Run all the models using MNIST 



