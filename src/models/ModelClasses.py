from sklearn.linear_model import Lasso
import numpy as np

class LassoWrapper():
    def __init__(self, model_params={'fit_intercept': True, 'max_iter': 100000}):
        self.model_name = "lasso"
        self.search_type = 'random'
        self.param_grid = {'alpha': np.linspace(0.001, 0.05, 100)}
        if model_params is None:
            self.ModelClass = Lasso()
        else:
            self.ModelClass = Lasso(**model_params)