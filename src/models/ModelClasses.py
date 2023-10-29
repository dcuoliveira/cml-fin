import os
import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np

class LassoWrapper():
    def __init__(self, model_params={'fit_intercept': True, 'max_iter': 100000}):

        # find optimal lambda according to hastie and tibshirani (2010)
        data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'inputs', 'etfs_macro_large.csv'))
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")

        removed_etfs = ['XLI', 'XLE', 'XLK', 'XLV', 'XLU', 'XLF', 'XLY', 'XLP', 'XLB']
        selected_data = data.drop(removed_etfs, axis=1)

        lambda_max = selected_data.drop(["SPY"], axis=1).mul(selected_data["SPY"], axis=0).sum(axis=0).abs().max() / selected_data.shape[0]
        epsilon = .0001
        K = 100
        alpha_path = np.round(np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * epsilon), K)), 10)

        self.model_name = "lasso"
        self.search_type = 'random'
        self.param_grid = {'alpha': alpha_path}
        if model_params is None:
            self.ModelClass = Lasso()
        else:
            self.ModelClass = Lasso(**model_params)

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        LassoWrapper()