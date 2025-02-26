import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
import statsmodels.api as sm
import lingam
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.api import OLS
import os
from os.path import join
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV

try:
    from causalnex.structure.dynotears import from_pandas_dynamic
except:
    print("causalnex package not installed. You wont be able to run dynotears model.") 

try:
    from tigramite import data_processing as pp
    from tigramite.independence_tests import cmiknn
    from tigramite.pcmci import PCMCI
except:
    print("tigramite package not installed. You wont be able to run pcmci model.")

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import numpy2ri
    pandas2ri.activate()
except:
    print("rpy2 package not installed. You wont be able to run seqICP model.")

from models.ClusteringModels import ClusteringModels, matchClusters
from models.ModelClasses import LassoWrapper, LinearRegressionWrapper, RandomForestWrapper, SVMWrapper
from utils.parsers import add_and_keep_lags_only

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

def run_lasso_filter(yt_train, Xt_train, yt_test, Xt_test):
    # since var-lingam is based on the VAR model, it cannot deal with collinearity => run lasso first
    alphas = np.linspace(0.0001, 0.2, 100) # we dont want to apply a very strong regularization, but we want var-lingam to do most of the work
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=1000000, alphas=alphas).fit(Xt_train, yt_train.values.ravel())

    # output the selected coefficients
    lasso_coefficients = pd.Series(lasso_cv.coef_, index=Xt_train.columns)
    selected_features = lasso_coefficients[lasso_coefficients != 0].index.tolist()
    
    data_train = pd.concat([yt_train, Xt_train[selected_features]], axis=1)
    data_test = pd.concat([yt_test, Xt_test[selected_features]], axis=1)

    return data_train, data_test

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
                 opt_k_method: str,
                 clustering_method: str,
                 n_clusters: int = 0,
                 cluster_threshold: float = 0.8,
                 intra_cluster_selection: str = "rank",
                 n_iter: int=10,
                 n_jobs: int = -1,
                 seed: int = 19940202,
                 max_p: int=3,
                 apply_lasso: bool=False):
    
    # sanity check causal representation learning method
    if (intra_cluster_selection == "pca") and ((clustering_method == "no") or (n_clusters ==0)):
        raise Exception("Causal Representation Learning Warning: intra_cluster_selection=pca can only be used with clustering_method!=no")
    
    # check if to use clustering or not, and if clustering is rolling or not
    if (clustering_method == "no") or (len(clustering_method.split("_")) == 1):
        rolling_cluster = False
    elif clustering_method.split("_")[0] == "rolling":
        rolling_cluster = True

        # get clustering method
        clustering_method = clustering_method.split("_")[1] if len(clustering_method.split("_")) > 1 else clustering_method
    
    cm = ClusteringModels()

    if rolling_cluster:
        if n_clusters == 0:
            clusters_path = join(os.path.dirname(os.path.dirname(__file__)),
                                "data",
                                "inputs",
                                "clusters",
                                clustering_method,
                                str(n_clusters),
                                opt_k_method)
        else:
            clusters_path = join(os.path.dirname(os.path.dirname(__file__)),
                                "data",
                                "inputs",
                                "clusters",
                                clustering_method,
                                str(n_clusters))
        if not n_clusters:
                clusters_path = join(clusters_path, str(cluster_threshold))
        os.makedirs(clusters_path, exist_ok = True)
        clusters_path = join(clusters_path, "clusters.parquet")

        if os.path.exists(clusters_path):
            clusters_series = pd.read_parquet(clusters_path)
        else:
            clusters_series = []
            for step in tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window, desc="computing clusters {}: {}".format(fs_method, target)):
                
                if fix_start or (step == 0):
                    start = 0
                else:
                    start += 1
                
                train_df = data.iloc[start:(estimation_window + step), :]
                clusters = cm.compute_clusters(data = train_df,
                                               target=target,
                                               n_clusters=n_clusters,
                                               clustering_method=clustering_method,
                                               opt_k_method=opt_k_method)
                labelled_clusters = cm.add_cluster_description(clusters=clusters)
                dfLabels = labelled_clusters.set_index("fred")[["cluster"]].copy()
                dfLabels.columns = [str(step)]
                clusters_series.append(dfLabels)
            # monthly clusters        
            clusters_series = pd.concat(clusters_series, axis = 1)
            matchClusters(clusters_series)
            clusters_series.to_parquet(clusters_path)

    predictions = []
    parents_of_target = []
    dags = {}
    for step in tqdm(range(0, len(data) - estimation_window, 1), total=len(data) - estimation_window, desc="forecasting {}: {}".format(fs_method, target)):

        if fix_start or (step == 0):
            start = 0
        else:
            start += 1

        train_df = data.iloc[start:(estimation_window + step), :]
        test_df = data.iloc[start:(estimation_window + step + 1), :]

        # compute within c1luster correlation
        if clustering_method != "no":
            if rolling_cluster:
                labelled_clusters = clusters_series[[str(step)]]
                labelled_clusters.columns = ["cluster"]
                labelled_clusters.reset_index(inplace = True)
            else:
                clusters = cm.compute_clusters(data=data, target=target, n_clusters=n_clusters, clustering_method=clustering_method)  
                labelled_clusters = cm.add_cluster_description(clusters=clusters)
            
            if intra_cluster_selection == "rank":
                ranks = cm.compute_within_cluster_corr_rank(data=train_df,
                                                            target=target,
                                                            labelled_clusters=labelled_clusters,
                                                            correl_window=correl_window)
                # select features and time window
                last_row = pd.DataFrame(ranks.iloc[-1])
                selected_columns = list(last_row[last_row == 1].dropna().index)
            elif intra_cluster_selection == "pca":
                train_pcs_df = cm.compute_within_cluster_pca(data=train_df,
                                                             labelled_clusters=labelled_clusters,
                                                             n_pcs=1)
                
                test_pcs_df = cm.compute_within_cluster_pca(data=test_df,
                                                            labelled_clusters=labelled_clusters,
                                                            n_pcs=1)

                train_df = pd.concat([train_df, train_pcs_df], axis=1)
                test_df = pd.concat([test_df, test_pcs_df], axis=1)
                selected_columns = list(train_pcs_df.columns)
            else:
                raise Exception(f"intra cluster selection method not registered: {intra_cluster_selection}")
        else:
            labelled_clusters = pd.DataFrame([{"fred": target, "cluster": 1, "description": target}])
            selected_columns = list(train_df.drop([target], axis=1).columns)

        train_df = train_df[[target] + selected_columns]

        # zscore of train data
        mean = train_df.mean()
        std = train_df.std()

        train_df = (train_df - mean) / std

        # select optimal lag
        if p == -1:
            var_select_model = VAR(train_df)
            selected_p = var_select_model.select_order(maxlags=max_p)
            selected_p = selected_p.selected_orders["aic"]
            if selected_p == 0:
                selected_p = 1
        else:
            selected_p = min(p, max_p)

        test_df = test_df[[target] + selected_columns].iloc[(estimation_window + step - selected_p):(estimation_window + step + 1), :]

        # zscore of test data
        test_df = (test_df - mean) / std

        # subset data into train and test
        Xt_train = train_df.drop([target], axis=1)
        yt_train = train_df[[target]]

        Xt_test = test_df.drop([target], axis=1)
        yt_test = test_df[[target]]

        if fs_method == "var-lingam":

            if apply_lasso:
                data_train, data_test = run_lasso_filter(yt_train, Xt_train, yt_test, Xt_test)
            else:
                data_train = pd.concat([yt_train, Xt_train], axis=1)
                data_test = pd.concat([yt_test, Xt_test], axis=1)

            # run VAR-LiNGAM
            var_lingam = lingam.VARLiNGAM(lags=selected_p, criterion="none")
            var_lingam_fit = var_lingam.fit(data_train)

            # build labels
            labels = {}
            selected_variables = []
            for i in range(selected_p):

                var_names = []
                for colname in data_train.columns:
                        if i == 0:
                            var_names.append(f"{colname}(t)")
                        else:
                            var_names.append(f"{colname}(t-{i})")
                labels[f'labels{i}'] = var_names

                B = var_lingam_fit._adjacency_matrices[i]
                B_df = pd.DataFrame(B, columns=labels[f'labels{i}'] , index=labels['labels0'] )
                tmp_selected_variables = list(B_df.loc["{target}(t)".format(target=target)][np.abs(B_df.loc["{target}(t)".format(target=target)]) > beta_threshold].index)

                selected_variables += [name.split("(")[0] for name in tmp_selected_variables]

            selected_variables = list(set(selected_variables))
            selected_variables = [f"{var}(t-{i})" for var in selected_variables for i in range(1, selected_p+1)]

            # save dags
            dict_ = {Xt_train.index[-1].strftime("%Y%m%d"): {
                "dag": var_lingam_fit._adjacency_matrices, 
                "threshold": beta_threshold,
                "labels": labels,
                }
            }
            dags.update(dict_)

            # create lags of Xt variables
            data_train = add_and_keep_lags_only(data=data_train, lags=selected_p)
            data_test = add_and_keep_lags_only(data=data_test, lags=selected_p)
                
            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()
        elif fs_method == "multivariate-granger":

            if apply_lasso:
                data_train, data_test = run_lasso_filter(yt_train, Xt_train, yt_test, Xt_test)
            else:
                data_train = pd.concat([yt_train, Xt_train], axis=1)
                data_test = pd.concat([yt_test, Xt_test], axis=1)

            # run grander causality test for each feature
            var_fit = VAR(data_train).fit(maxlags=selected_p)

            # run grander causality test for each feature
            selected_variables = []
            for colname in data_train.columns:
                if colname != target:
                    test_result = var_fit.test_causality(target, [colname], kind='f', signif=pval_threshold)
                    if test_result.pvalue <= pval_threshold:
                        for lag in range(1, selected_p + 1):
                            selected_variables += [f"{colname}(t-{lag})"]               

            # create lags of Xt variables
            data_train = add_and_keep_lags_only(data=data_train, lags=selected_p)
            data_test = add_and_keep_lags_only(data=data_test, lags=selected_p)

            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()
        elif fs_method == "dynotears":
            if apply_lasso:
                data_train, data_test = run_lasso_filter(yt_train, Xt_train, yt_test, Xt_test)
            else:
                data_train = pd.concat([yt_train, Xt_train], axis=1)
                data_test = pd.concat([yt_test, Xt_test], axis=1)

            dates = data_train.index
            target_data = data_train.reset_index(drop=True)

            # run DYNOTEARS
            dynotears = from_pandas_dynamic(target_data, p=selected_p)

            edges_df = pd.DataFrame(dynotears.edges(), columns=['to', 'from'])[['from', 'to']]
            edges_df["from_lag"] = edges_df["from"].apply(lambda x: int(x.split("_")[1][-1]))
            edges_df["to_lag"] = edges_df["to"].apply(lambda x: int(x.split("_")[1][-1]))

            edges_df["new_from"] = edges_df["from"].apply(lambda x: x.split("_")[0])
            edges_df["new_to"] = edges_df["to"].apply(lambda x: x.split("_")[0])

            # select valid edges
            edges_df = edges_df.loc[(edges_df["from_lag"] == 0) & (edges_df["to_lag"] != 0)]

            # build adjacency matrix
            adj = []
            for from_node in edges_df.loc[edges_df["from_lag"] == 0]["new_from"].unique():
                tmp_from = edges_df.loc[edges_df["new_from"] == from_node]
                col_names = []
                for idx, row in tmp_from.iterrows():
                    col_names.append(f"{row['new_to']}(t-{row['to_lag']})")
                row_name = from_node

                tmp_adj = pd.DataFrame(1, columns=col_names, index=[f"{row_name}(t)"])
                adj.append(tmp_adj)
            adj_df = pd.concat(adj).fillna(0)

            if f"{target}(t)" in adj_df.index:
                selected_variables = list(adj_df[adj_df.index == f"{target}(t)"][adj_df[adj_df.index == f"{target}(t)"] != 0].dropna(axis=1).columns)
            else:
                selected_variables = []

            # # save dags
            # dict_ = {Xt_train.index[-1].strftime("%Y%m%d"): {
            #     "dag": adj_df, 
            #     "threshold": beta_threshold,
            #     "labels": labels,
            #     }
            # }
            # dags.update(dict_)

            # create lags of Xt variables
            data_train = add_and_keep_lags_only(data=data_train, lags=selected_p)
            data_test = add_and_keep_lags_only(data=data_test, lags=selected_p)

            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()
        elif fs_method == "seqICP":

            if apply_lasso:
                data_train, data_test = run_lasso_filter(yt_train, Xt_train, yt_test, Xt_test)
            else:
                data_train = pd.concat([yt_train, Xt_train], axis=1)
                data_test = pd.concat([yt_test, Xt_test], axis=1)
            
            # convert dataframe to R objects
            X_train = data_train.drop(target, axis=1).values
            y_train = data_train[target]

            X_train_r = numpy2ri.numpy2rpy(X_train)
            y_train_r = numpy2ri.numpy2rpy(y_train)
            selected_p_r =  robjects.vectors.IntVector([selected_p])
            pval_threshold_r = robjects.vectors.IntVector([pval_threshold])

            # pass inputs to global variables
            robjects.globalenv['Xmatrix'] = X_train_r
            robjects.globalenv['Y'] = y_train_r
            robjects.globalenv["selected_p"] = selected_p_r
            robjects.globalenv["pval_threshold"] = pval_threshold_r

            robjects.r(f'''
                library(seqICP)

                seqICP_result <- seqICP(Xmatrix,
                                        Y,
                                        test="smooth.variance",
                                        par.test=list(alpha=pval_threshold,B=1000),
                                        model="ar",
                                        par.model=list(pknown=TRUE,p=selected_p),
                                        stopIfEmpty=FALSE,
                                        silent=TRUE)
                seqICP_summary <- summary(seqICP_result)
                parent_set <- seqICP_result$parent.set
                p_values <- seqICP_result$p.values

            ''')

            # retrieve results from seqICP
            p_values = robjects.r['p_values']
            
            selected_variables_df = pd.DataFrame({
                "variables": data_train.drop(target, axis=1).columns,
                "pval": p_values
            })

            selected_variables_df = selected_variables_df.loc[selected_variables_df["pval"] <= pval_threshold]
            
            if selected_variables_df.shape[0] > 0:
                selected_variables = []
                for feature in selected_variables_df["variables"]:
                    for i in range(1, selected_p+1):
                        selected_variables.append(f"{feature}(t-{i})")
            else:
                selected_variables = []
            
            # create lags of Xt variables
            data_train = add_and_keep_lags_only(data=data_train, lags=selected_p)
            data_test = add_and_keep_lags_only(data=data_test, lags=selected_p)
            
            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()
        elif fs_method.startswith("sfscv") or fs_method.startswith("sfstscv"):

            if fs_method.startswith("sfstscv"):
                cv = TimeSeriesSplit(n_splits=5)
            elif fs_method.startswith("sfscv"):
                cv = KFold(n_splits=5)
            else:
                raise Exception(f'Cross-validation Method not Recognized {fs_method}')

            if 'forward' in fs_method:
                direction = 'forward'
            elif 'backward' in fs_method:
                direction = 'backward'
            else:
                raise Exception(f'Feature Selection Direction not recognized: {fs_method}')

            if '-lin' in fs_method:
                model_wrapper = LinearRegressionWrapper(model_params={'fit_intercept': True})
            elif '-rf'in fs_method:
                model_wrapper = RandomForestWrapper()
            elif '-svm' in fs_method:
                model_wrapper = SVMWrapper()
            else:
                raise Exception(f'Feature Selection Model not recognized: {fs_method}')

            sfs = SequentialFeatureSelector(
                model_wrapper.ModelClass, 
                direction=direction, 
                cv=cv, 
                scoring="neg_mean_squared_error"
            )

            pipeline = Pipeline([
                ('feature_selector', sfs),
                ('model', model_wrapper.ModelClass)
            ])

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=model_wrapper.param_grid,
                n_iter=n_iter,
                n_jobs=n_jobs,
                cv=cv,
                scoring="neg_mean_squared_error",
                random_state=seed
            )

            Xt_train = pd.concat([yt_train, Xt_train], axis=1)
            Xt_test = pd.concat([yt_test, Xt_test], axis=1)

            # create lags of Xt variables
            Xt_train = add_and_keep_lags_only(data=Xt_train, lags=selected_p)
            Xt_test = add_and_keep_lags_only(data=Xt_test, lags=selected_p)

            Xt_train = Xt_train.dropna()
            yt_train = yt_train.loc[Xt_train.index]

            search_output = search.fit(Xt_train, yt_train.values.ravel())

            selected_indices = search.best_estimator_.named_steps['feature_selector'].get_support()
            selected_variables = Xt_train.columns[selected_indices]
        elif fs_method == "pcmci":

            if apply_lasso:
                data_train, data_test = run_lasso_filter(yt_train, Xt_train, yt_test, Xt_test)
            else:
                data_train = pd.concat([yt_train, Xt_train], axis=1)
                data_test = pd.concat([yt_test, Xt_test], axis=1)

            data_train_tigramite = pp.DataFrame(data_train.values, var_names=data_train.columns)

            pcmci = PCMCI(dataframe=data_train_tigramite, cond_ind_test=cmiknn.CMIknn(), verbosity=0)
            pcmci.run_pcmci(tau_min=0, tau_max=selected_p, pc_alpha=pval_threshold)

            parents_set = dict()
            for effect in pcmci.all_parents.keys():
                parents_set[pcmci.var_names[effect]] = []
                for cause, t in pcmci.all_parents[effect]:
                    parents_set[pcmci.var_names[effect]].append((pcmci.var_names[cause], t))

            # build labels
            selected_variables = []
            if len(parents_set[target]) != 0:
                for i in range(len(parents_set[target])):
                    cause = parents_set[target][i][0]
                    t = np.abs(parents_set[target][i][1])
                    selected_variables.append(f"{cause}(t-{t})")

            # create lags of Xt variables
            data_train = add_and_keep_lags_only(data=data_train, lags=selected_p)
            data_test = add_and_keep_lags_only(data=data_test, lags=selected_p)
                
            Xt_train = data_train.dropna()
            Xt_test = data_test.dropna()
        else:
            raise Exception(f"fs method not registered: {fs_method}")

        selected_variables_df = pd.DataFrame(1, index=selected_variables, columns=[Xt_test.index[-1]]).T

        yt_test_zscore = yt_test.copy()
        yt_test_zscore.index = pd.to_datetime(yt_test_zscore.index)
        yt_test = yt_test * std[yt_test.columns[0]] + mean[yt_test.columns[0]]
        yt_test.index = pd.to_datetime(yt_test.index)

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
                Xt_selected_train["const"] = 1
                Xt_selected_test["const"] = 1

            # linear regression estimate and prediction
            model = sm.OLS(endog=Xt_selected_train[target], exog=Xt_selected_train.drop([target], axis=1))
            model_fit = model.fit()
            ypred = model_fit.predict(exog=Xt_selected_test)

            pred = pd.DataFrame([{
                "date": ypred.index[0],
                "prediction_zscore": ypred[0],
                "true_zscore": yt_test_zscore.loc[ypred.index[0]][0],
                "prediction": ypred[0] * std[yt_test.columns[0]] + mean[yt_test.columns[0]],
                "true": yt_test.loc[ypred.index[0]][0],
                }])
            predictions.append(pred)
        else:
            pred = pd.DataFrame([{
                "date": yt_test.index[-1],
                "prediction_zscore": 0,
                "true_zscore": yt_test_zscore.iloc[-1][0],
                "prediction": 0,
                "true": yt_test.iloc[-1][0],
                }])
            predictions.append(pred)

    # save predictions
    predictions_df = pd.concat(predictions, axis=0)
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    predictions_df.set_index('date', inplace=True)

    # save parents of target
    if len(parents_of_target) != 0:
        parents_of_targets_df = pd.concat(parents_of_target, axis=0)
    else:
        parents_of_targets_df = pd.DataFrame(columns=["date", "variable", "value", "fred", "cluster"])

    results = {"predictions": predictions_df,
                "parents_of_target": parents_of_targets_df,
                "dags": dags}

    return results
