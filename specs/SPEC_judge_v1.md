## GOAL

To develop a judge agent or augmented LLM (LLM as a judge) in an ML pipeline and decide what to do next (orchestrate and drive)

## CONSTRAINTS
1. Only use Google ADK to develop this agent (https://github.com/google/adk-python).
Do not write any code from scratch. Agent development kit has functions which you can directly use.
2. No prompts directly in the code. use jinja2


## INPUTS
1. From an overfitting analysis tool - which will give a set of metrics (classification metrics / regression metrics), the gap between the training accuracy and inference accuracy. a json of these metrics.
2. Inference metrics from 5-10 ML models (Assume the input structure yourself and output in input_format_requirement.md)
3. SHAP Explainability metrics of 5-10 ML models (Another script which will give you these in a json) which you can use for model debugging - `https://www.geeksforgeeks.org/machine-learning/shap-a-comprehensive-guide-to-shapley-additive-explanations/` (You will get only numbers/text - no images/plots, go through this page to understand what can be used to do model debugging)
4. Minidata.csv - A csv containing pd.describe() output
5. metadata.csv - User metadata - format is not fixed yet.
6. Hyperparameter tuning sensitivity metrics

## OUTPUTS

1. You should suggest whether this model can be selected for the leaderboard (as the top performing ML model for this dataset) considering these factors (ordered by their weightage):
    (i). Accuracy/R2 score or any other performance metrics
    (ii). How much the model has overfitted to the available data (We should avoid overfitting)
    (iii). Complexity of the model.
2. Decide the output format yourself.
3. The judge agent should be interfacable with an orchestrator which will give feedback to the orchestrator agent (whether to select the model or not)

## JUDGE AGENT THOUGHT PROCESS
1. Prefer simple models when performance differs by 1%
2. Prefer Low overfitting on train data - high accuracy on test data
3. Reject models with test acccuracy below a certain threshold (parameterize it)