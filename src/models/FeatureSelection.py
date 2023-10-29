import lingam
import pandas as pd
import numpy as np

class FeatureSelection:
    def __init__(self, p: int, beta_threshold: float) -> None:
        self.p = p
        self.beta_threshold = beta_threshold

    def compute_var_lingam_fs(self, yt_train, Xt_train, yt_test, Xt_test):
        
        # run VARLiNGAM
        var_lingam = lingam.VARLiNGAM(lags=p)
        var_lingam_fit = var_lingam.fit(Xt_train)

        labels0 = []
        labels1 = []
        for i in range(p+1):
            for colname in Xt_train.columns:
                if i == 0:
                    labels0.append("{}(t)".format(colname, i))
                else:
                    labels1.append("{}(t-{})".format(colname, i))

        B0 = var_lingam_fit.adjacency_matrices_[0]
        B1 = var_lingam_fit.adjacency_matrices_[1]

        B0_df = pd.DataFrame(B0, columns=labels0, index=labels0)
        B1_df = pd.DataFrame(B1, columns=labels1, index=labels0)

        selected_variables = list(B1_df.loc["{target}(t)".format(target=target)][np.abs(B1_df.loc["{target}(t)".format(target=target)]) > beta_threshold].index)

        # create lags of Xt variables
        for c in Xt_train.columns:
            for lag in range(1, p + 1):
                Xt_train["{}(t-{})".format(c, lag)] = Xt_train[c].shift(lag)
                Xt_test["{}(t-{})".format(c, lag)] = Xt_test[c].shift(lag)
            
            Xt_train.drop(c, axis=1, inplace=True)

    def compute_lasso_fs(self):
        pass

    def compute_granger_causality_fs(self):
        pass

    def compute_neural_granger_causality_fs(self):
        pass