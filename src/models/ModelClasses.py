import os
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor

class LinearRegressionWrapper():
    def __init__(self, model_params={'fit_intercept': True}):

        self.model_name = "linear_regression"
        self.search_type = 'grid'
        self.param_grid = {'fit_intercept': [True, False]}
        if model_params is None:
            self.ModelClass = LinearRegression()
        else:
            self.ModelClass = LinearRegression(**model_params)

class LassoWrapper():
    def __init__(self, model_params={'fit_intercept': True, 'max_iter': 10000000}, hyperparameter_space_method="default"):

        if hyperparameter_space_method == "fht":
            # find optimal lambda according to friedman, hastie, and tibshirani (2010)
            ## Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), 1.
            data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'inputs', 'etfs_macro_large.csv'))
            data["date"] = pd.to_datetime(data["date"])
            data = data.set_index("date")

            removed_etfs = ['XLI', 'XLE', 'XLK', 'XLV', 'XLU', 'XLF', 'XLY', 'XLP', 'XLB']
            selected_data = data.drop(removed_etfs, axis=1)

            lambda_max = selected_data.drop(["SPY"], axis=1).mul(selected_data["SPY"], axis=0).sum(axis=0).abs().max() / selected_data.shape[0]
            epsilon = .0001
            K = 100
            alpha_path = np.round(np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * epsilon), K)), 10)
        elif hyperparameter_space_method == "default":
            alpha_path = np.logspace(-4, 1, num=100)

        self.model_name = "lasso"
        self.search_type = 'random'
        self.param_grid = {'alpha': alpha_path}
        if model_params is None:
            self.ModelClass = Lasso()
        else:
            self.ModelClass = Lasso(**model_params)

class RandomForestWrapper():
    def __init__(self, model_params=None):
        self.model_name = "random_forest"
        self.search_type = 'random'
        self.param_grid = {'model__max_features': ['auto', 'sqrt', 'log2'],
                           'model__min_samples_split': sp_randint(2, 31),
                           'model__n_estimators': sp_randint(2, 301),
                           'model__max_depth': sp_randint(2, 20)}
        if model_params is None:
            self.ModelClass = RandomForestRegressor()
        else:
            self.ModelClass = RandomForestRegressor(**model_params)

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        LassoWrapper()