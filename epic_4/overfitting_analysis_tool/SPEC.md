## GOAL

The goal is to write python scripts which will get the regression and classification metrics and figure out if the ML model has overfitted or not.

## Application context
1. Should work in an AutoML pipeline.


## Requirements (These things should be there for sure)
1. Calculate overfitting gap = training accuracy - test accuracy
2. Perform K-fold cross validation
3. Need to be interfacable with another training class through which the script can run the training - keep a placeholder for now

## Output Format:
A json which has fields like
```
{
	overfitting_gap: <float>,
	k_fold_cross_validation_results: {}
}

## Input Format:
```json
{
	"model_type": "classification|regression",
	"train_metrics":{},
	"test_metrics":{},
}

## CLI Args:
-o, <output_dir>
-i, <input_json>


## Development outputs
1. config/config.yaml - capturing all the necessary controllables
2. overfitting_analysis.py - 
3. tests/ -> test_suite to create a simple training class and with simple dataset and verify if it works.


## Acceptance criteria:
1. Run the script successfully with the test_suite and report the outputs

