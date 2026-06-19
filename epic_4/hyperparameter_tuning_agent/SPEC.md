# SPEC: Epic-4 - Hyperparameter Tuning Agent

## 1. PROBLEM STATEMENT

I need to find the best hyperparameters for machine learning models automatically. If I manually try different values then it is:

a. Time-consuming (takes hours/days)
b. Not systematic (I might miss good combinations)
c. Not reproducible (different people get different results)

Example: For a Random Forest, I need to decide:
a. How many trees? (10/100/1000?)
b. How deep can each tree grow? (3/5/10?)
c. How many features to consider at each split?

## 2. Current Understanding

### 2.1 What are Hyperparameters?
Hyperparameters are settings I choose before training starts:
a. They control HOW the model learns
b. They are NOT learned from the data
c. Different values give different results

### 2.2 What is Hyperparameter Tuning?
It's the process of:
1. Trying different combinations of hyperparameters
2. Training the model with each combination
3. Evaluating performance (accuracy, F1, etc.)
4. Finding the combination that gives the best performance

### 2.3 Solution for tuning Hyperparameters

Existing problem with manually choosing hyperparameters :
- How do I know which hyperparameters to tune for each model?
- How many trials should I run?
- How do I know if the model is overfitting?
- How can I run multiple trials in parallel to save time?

Solution : Use Optuna because of below features
- TPESampler : Optuna uses Bayesian optimization (intelligent search)
- MedianPruner:  Optuna supports pruning (stop bad trials early)
- suggest_int(), suggest_float(), suggest_categorical() :  Defines search spaces
- Optuna handles reproducibility with seeds
- Optuna can run parallel trials

### 2.4 How Optuna Works
1. I define a search space (what hyperparameters to try)
2. Optuna suggests values intelligently
3. I train the model with those values
4. I return the accuracy (or loss)
5. Optuna learns which values work best
6. Repeat for N trials

### 2.5 Dependencies
1. Metrics : Use accuracy for classification and R² for regression as defaults, but make it configurable.
2. from the train.csv file from epic2 - split train.csv into train and validation (Use 80/20 split by default, configurable via config.ini.)
3. Overfitting: (train_score - val_score) > some threshold. For now lets keep 10% and Configurable via config.ini - OVERFITTING_GAP_THRESHOLD
4. Select least-overfitted trial as fallback, if everything is overfitted

### 2.6 Enhancements 
1. To find the metrics we already have "evaluators.py" which compute_metrics
2.  - Classification - returns accuracy, f1_macro, f1_weighted, precision_macro, recall_macro
    - Regression -	mse, rmse, mae, r2
3. For parallel execution - use RAY

### 2.7 Output format 
- in hpt_results.json
{
  "hpt_results": [
    {
      "name": "xgb_v1",
      "best_hyperparameters": {...},
      "val_metrics": {...},
      "train_metrics": {...},
      "overfitting": {"is_overfitted": false, "gap": 0.03}
    }
  ]
}

### Process flow
1. Read config.ini
2. Read metadata.json (problem_type, target_col)
3. Read model_config.json (models + search spaces)
4. Load train.csv
5. Split into train/val (80/20 with stratification)
6. Create DataBundle
7. For each model in model_config.json:
   a. Create training function
   b. Initialize OptunaWrapper
   c. Run optimization (MAX_HPT_TRIALS trials)
   d. Select best trial (prefer non-overfitted)
   e. Build result entry
   f. Add to results list
8. Write hpt_results.json (atomic write)
9. Update metadata.json with completion info

# Test Datasets
1. Iris (classification, small)
2. Wine (classification, medium)
3. Breast Cancer (classification, medium)
4. Boston Housing (regression, small)