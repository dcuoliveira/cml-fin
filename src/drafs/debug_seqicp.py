import argparse
import pandas as pd
import os
import statsmodels.api as sm

from forecast.forecast_funcs import run_forecast
from metadata.etfs import etfs_large, etfs_small
from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

from statsmodels.tsa.api import VAR
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
pandas2ri.activate()

from utils.parsers import add_and_keep_lags_only

parser = argparse.ArgumentParser(description="Run forecast.")

estimation_window=12 * 7
p=-1
correl_window=100000
beta_threshold=0.4
pval_threshold=0.05
fix_start=True
incercept=True
fs_method="seqICP"
opt_k_method="no"
clustering_method="no"
n_clusters=0
intra_cluster_selection="no"
data_name="monetary-policy-processed"
inputs_path=os.path.join(os.getcwd(), "data", "inputs")
outputs_path=os.path.join(os.getcwd(), "data", "outputs")
target="ldEXME"


fix_start = str_2_bool(fix_start)
incercept = str_2_bool(incercept)

data = pd.read_csv(os.path.join(inputs_path, f'{data_name}.csv'))

# fix columns
if "Unnamed: 0" in data.columns:
    data = data.drop(["Unnamed: 0"], axis=1)

# fix dates
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")

etfs_large = etfs_large.copy()

target = target

if target != "ldEXME":
    # select etfs to remove
    removed_etfs = [etf for etf in etfs_large if etf != target]

    # delete etfs
    selected_data = data.drop(removed_etfs, axis=1)
else:
    selected_data = data.copy()

data=selected_data
target=target
fix_start=fix_start
estimation_window=estimation_window
correl_window=correl_window
p=p
beta_threshold=beta_threshold
pval_threshold=pval_threshold
incercept=incercept
fs_method=fs_method
opt_k_method=opt_k_method
clustering_method=clustering_method
n_clusters=n_clusters
intra_cluster_selection=intra_cluster_selection

predictions = []
parents_of_target = []
for step in [94, 95, 96]:

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
        selected_p = var_select_model.select_order(maxlags=6)
        selected_p = selected_p.selected_orders["aic"]
        if selected_p == 0:
            selected_p = 1
    else:
        selected_p = p

    test_df = test_df[[target] + selected_columns].iloc[(estimation_window + step - selected_p):(estimation_window + step + 1), :]

    # zscore of test data
    test_df = (test_df - mean) / std

    # subset data into train and test
    Xt_train = train_df.drop([target], axis=1)
    yt_train = train_df[[target]]

    Xt_test = test_df.drop([target], axis=1)
    yt_test = test_df[[target]]

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

    selected_variables_df.to_csv(f'pvalues_step={step}.csv')

    selected_variables_df = selected_variables_df.loc[selected_variables_df["pval"] <= pval_threshold]

    if selected_variables_df.shape[0] > 0:
        selected_variables = []
        for feature in selected_variables_df["variables"]:
            for i in range(1, selected_p+1):
                selected_variables.append(f"{feature}(t-{i})")
    else:
        selected_variables = []

    data_train.to_csv(f'data_train_old_step={step}.csv')

    # create lags of Xt variables
    add_and_keep_lags_only(data=data_train, lags=selected_p)
    add_and_keep_lags_only(data=data_test, lags=selected_p)

    data_train.to_csv(f'data_train_new_step={step}.csv')

    Xt_train = data_train.dropna()
    Xt_test = data_test.dropna()

    selected_variables_df = pd.DataFrame(1, index=selected_variables, columns=[Xt_test.index[-1]]).T

    selected_variables_df.to_csv(f'selected_variables_step={step}.csv')

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

        Xt_train.to_csv(f'Xtrain_step={step}.csv')

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