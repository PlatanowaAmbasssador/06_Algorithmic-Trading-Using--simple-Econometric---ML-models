# Machine Learning in Finance 1 - Project
## Algorithmic investment strategies using simple ML and econometric models
#### Kamil Kashif & Damian Ślusarczyk

## Description

### Aim
This project aims to check which of the used models helps us obtain the best strategy for trading AAPL stock. We predict stock price movement direction, therefore it is a classification problem.

### Data
We use AAPL stock prices between 2013 and 2023 obtained from Yahoo! Finance. As a part of feature engineering, we calculate technical indicators.

### Models
We use the following ML/econometric models:
- linear regression with elastic net (the numerical output is transformed to categorical data),
- logistic regression with elastic net,
- random forest.
All models are implemented in `scikit learn` library.

### Performance metrics
Because our goal is to make the best investment strategy, instead of using ML performance metrics, we use performance metrics for investment strategies. Our main performance metric is **Information ratio\*\***.

### Validation approach
We use walk forward approach for validation and hyperparameter tuning. For details see `02_linear_and_logit_regressions.ipynb` and `03_ML_random_forest_model.ipynb`.

## Project structure

- `01_features_engineering_and_eda.ipynb` - downloading data, feature engineering, EDA,
- `02_linear_and_logit_regressions.ipynb` - fitting linear and logit regression models,
- `03_ML_random_forest_model.ipynb` - fitting random forest model,
- `04_naive_strategies_and_results.ipynb` - analysing the results, calculating performance metrics,
- `useful_functions.py` - functions and classes used throughout the project,
- `input`, `data_objects` - input data files and other objects generated in `01`,
- `output` - model predictions generated in `02` and `03`,
- `Data_nd_WFO.ipynb` - graphical representation of walk forward approach,
- `requirements.txt` - requirements file to run the code.
