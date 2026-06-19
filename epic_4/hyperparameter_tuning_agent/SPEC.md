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
3. Overfitting: train_score - val_score > some threshold. For now lets keep 10% and later configurable