import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import statsmodels.api as sm
import lingam
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

from models.ClusteringModels import ClusteringModels
from models.ModelClasses import LassoWrapper

def cv_opt(X, y, model_wrapper, cv_type, n_splits, n_iter, seed, verbose, n_jobs, scoring):

    # define split type
    if cv_type == 'tscv':
        splits = TimeSeriesSplit(n_splits=n_splits)
    elif cv_type == 'cv':
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # choose search type
    if model_wrapper.search_type == 'random':
        model_search = RandomizedSearchCV(estimator=model_wrapper.ModelClass,
                                        param_distributions=model_wrapper.param_grid,
                                        n_iter=n_iter,
                                        cv=splits,
                                        verbose=verbose,
                                        n_jobs=n_jobs,
                                        scoring=scoring,
                                        random_state=seed)
    elif model_wrapper.search_type == 'grid':
        model_search = GridSearchCV(estimator=model_wrapper.ModelClass,
                                    param_grid=model_wrapper.param_grid,
                                    cv=splits,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    scoring=scoring)
    else:
        raise Exception('search type method not registered')

    # find best model
    model_fit = model_search.fit(X, y)

    return model_fit

def run_forecast(data: pd.DataFrame,
                 target: str,
                 fix_start: bool,
                 estimation_window: int,
                 correl_window: int,
                 p: int,
                 beta_threshold: float,
                 pval_threshold: float,
                 incercept: bool,
                 fs_method: str,
                 cv_type: str):
    
    cm = ClusteringModels()

    predictions = []
    parents_of_target = []
    for step in tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window, desc="rolling {}: {}".format(fs_method, target)):

        if fix_start or (step == 0):
            start = 0
        else:
            start += 1

        train_df = data.iloc[start:(estimation_window + step), :]

        # compute within c1luster correlation
        clusters = cm.compute_clusters(data=data, target=target, clustring_method="kmeans")  
        labelled_clusters = cm.add_cluster_description(clusters=clusters)
        ranks = cm.compute_within_cluster_corr_rank(data=train_df,
                                                    target=target,
                                                    labelled_clusters=labelled_clusters,
                                                    correl_window=correl_window)      

        # select features and time window
        last_row = pd.DataFrame(ranks.iloc[-1])
        selected_columns = list(last_row[last_row == 1].dropna().index)
        train_df = train_df[[target] + selected_columns]
        test_df = data[[target] + selected_columns].iloc[(estimation_window + step - p):(estimation_window + step + 1), :]

        # subset data into train and test
        Xt_train = train_df.drop([target], axis=1)
        yt_train = train_df[[target]]

        Xt_test = test_df.drop([target], axis=1)
        yt_test = test_df[[target]]

        if fs_method == "var-lingam":
            data_train = pd.concat([yt_train, Xt_train], axis=1)
            data_test = pd.concat([yt_test, Xt_test], axis=1)

            # run VAR-LiNGAM
            var_lingam = lingam.VARLiNGAM(lags=p)
            var_lingam_fit = var_lingam.fit(data_train)

            # build labels - ONLY WORKS FOR k=1
            labels = {}
            for i in range(p+1):

                var_names = []
                for colname in data_train.columns:
                        if i == 0:
                            var_names.append(f"{colname}(t)")
                        else:
                            var_names.append(f"{colname}(t-{i})")
                labels[f'labels{i}'] = var_names

                if i != 0:
                    B = var_lingam_fit.adjacency_matrices_[i]
                    B_df = pd.DataFrame(B, columns=labels[f'labels{i}'] , index=labels['labels0'] )
                    selected_variables = list(B_df.loc["{target}(t)".format(target=target)][np.abs(B_df.loc["{target}(t)".format(target=target)]) > beta_threshold].index)

            # create lags of Xt variables
            for c in data_train.columns:
                for lag in range(1, p + 1):
                    data_train["{}(t-{})".format(c, lag)] = data_train[c].shift(lag)
                    data_test["{}(t-{})".format(c, lag)] = data_test[c].shift(lag)
                
                data_train.drop(c, axis=1, inplace=True)
            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()

        elif fs_method == "lasso":
            Xt_train = pd.concat([yt_train, Xt_train], axis=1)
            Xt_test = pd.concat([yt_test, Xt_test], axis=1)

            # create lags of Xt variables
            for c in Xt_train.columns:
                for lag in range(1, p + 1):
                    Xt_train["{}(t-{})".format(c, lag)] = Xt_train[c].shift(lag)
                    Xt_test["{}(t-{})".format(c, lag)] = Xt_test[c].shift(lag)
                
                Xt_train.drop(c, axis=1, inplace=True)

            Xt_train = Xt_train.dropna()
            yt_train = yt_train.loc[Xt_train.index]

            # train lasso model
            model_fit = cv_opt(X=Xt_train,
                               y=yt_train,
                               model_wrapper=LassoWrapper(),
                               cv_type=cv_type,
                               n_splits=2,
                               n_iter=10,
                               seed=2294,
                               verbose=False,
                               n_jobs=1,
                               scoring=make_scorer(mean_squared_error))

            # fit best model
            lasso_best_fit = model_fit.best_estimator_.fit(Xt_train, yt_train)
            B1_df = pd.DataFrame(lasso_best_fit.coef_, index=Xt_train.columns, columns=[f"{target}(t)"]).sort_values(ascending=False, by=f"{target}(t)")

            # select variables with beta > threshold
            selected_variables = list(B1_df[B1_df > beta_threshold].dropna().index)
        elif fs_method == "pairwise-granger":
            data_train = pd.concat([yt_train, Xt_train], axis=1)
            data_test = pd.concat([yt_test, Xt_test], axis=1)

            selected_variables = []
            # run grander causality test for each feature
            for colname in data_train.columns:
                test_result = grangercausalitytests(x=data_train[[target, colname]], maxlag=p, verbose=False, addconst=incercept)

                # select variables with p-value < 0.05
                for lag in test_result.keys():
                    pval = test_result[lag][0]['ssr_ftest'][1]
                    if pval <= pval_threshold:
                        selected_variables += [f"{colname}(t-{lag})"]

            # create lags of Xt variables
            for c in data_train.columns:
                for lag in range(1, p + 1):
                    data_train["{}(t-{})".format(c, lag)] = data_train[c].shift(lag)
                    data_test["{}(t-{})".format(c, lag)] = data_test[c].shift(lag)
                
                data_train.drop(c, axis=1, inplace=True)
            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()
        elif fs_method == "multivariate-granger":
            data_train = pd.concat([yt_train, Xt_train], axis=1)
            data_test = pd.concat([yt_test, Xt_test], axis=1)

            # run grander causality test for each feature
            var_fit = VAR(data_train).fit(maxlags=p)

            # run grander causality test for each feature
            selected_variables = []
            for colname in data_train.columns:
                if colname != target:
                    test_result = var_fit.test_causality(target, [colname], kind='f', signif=pval_threshold)
                    if test_result.pvalue <= pval_threshold:
                        for lag in range(1, p + 1):
                            selected_variables += [f"{colname}(t-{lag})"]               

            # create lags of Xt variables
            for c in data_train.columns:
                for lag in range(1, p + 1):
                    data_train["{}(t-{})".format(c, lag)] = data_train[c].shift(lag)
                    data_test["{}(t-{})".format(c, lag)] = data_test[c].shift(lag)
                
                data_train.drop(c, axis=1, inplace=True)
            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()
        else:
            raise Exception("fs method not registered")

        selected_variables_df = pd.DataFrame(1, index=selected_variables, columns=[Xt_test.index[-1]]).T
        if len(selected_variables) != 0:

            # add clusters to parents
            melted_selected_variables_df = selected_variables_df.reset_index().melt("index").rename(columns={"index": "date"})
            melted_selected_variables_df["fred"] = [varname.split("(")[0] for varname in melted_selected_variables_df["variable"]]
            melted_selected_variables_df = pd.merge(melted_selected_variables_df, labelled_clusters[["fred", "cluster"]], how="left", on="fred")
            parents_of_target.append(melted_selected_variables_df)

            Xt_selected_train = []
            Xt_selected_test = []
            for full_vname in selected_variables:
                Xt_selected_train.append(Xt_train[full_vname])
                Xt_selected_test.append(Xt_test[full_vname])

            Xt_selected_train = pd.concat(Xt_selected_train, axis=1)
            Xt_selected_train = pd.concat([yt_train, Xt_selected_train], axis=1)
            Xt_selected_train = Xt_selected_train.dropna()

            Xt_selected_test = pd.concat(Xt_selected_test, axis=1)
            Xt_selected_test = Xt_selected_test.dropna()

            if incercept:
                Xt_selected_train['const'] = 1
                Xt_selected_test['const'] = 1

            # linear regression estimate and prediction
            model = sm.OLS(endog=Xt_selected_train[target], exog=Xt_selected_train.drop([target], axis=1))
            model_fit = model.fit()
            ypred = model_fit.predict(Xt_selected_test)

            pred = pd.DataFrame([{"date": ypred.index[0], "prediction": ypred[0], "true": yt_test.loc[ypred.index[0]][0]}])
            predictions.append(pred)
        else:
            pred = pd.DataFrame([{"date": yt_test.index[-1], "prediction": 0, "true": yt_test.loc[yt_test.index[-1]][0]}])
            predictions.append(pred)

    # save predictions
    predictions_df = pd.concat(predictions, axis=0)
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    predictions_df.set_index('date', inplace=True)

    # save parents of target
    parents_of_targets_df = pd.concat(parents_of_target, axis=0)

    results = {"predictions": predictions_df,
                "parents_of_target": parents_of_targets_df}

    return results