import builtins
# We define 'init' as a dummy function or object so the library doesn't crash
builtins.init = lambda: None 

import os
import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
import numpy as np
from tqdm import tqdm
import argparse
from PyTimeVar import GAS

from utils.conn_data import save_pickle

# Suppress warnings
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Run GAS model forecasting")
    parser.add_argument("--target", type=str, default="SPY", help="Target variable to forecast")
    parser.add_argument("--pred_method", type=str, default="gas-tstudent", help="Feature selection method")
    parser.add_argument("--estimation_window", type=int, default=12*7, help="Estimation window size")
    parser.add_argument("--fix_start", action='store_true', help="Whether to fix the start of the estimation window")
    parser.add_argument("--p", type=int, default=1, help="Number of lags to include as features (-1 for automatic VAR selection)")
    parser.add_argument("--max_p", type=int, default=3, help="Maximum number of lags for VAR selection")
    parser.add_argument("--inputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "inputs"))
    parser.add_argument("--outputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "outputs"))
    parser.add_argument("--data_name", type=str, default="etfs_macro_large", help="Name of the dataset file")
    parser.add_argument("--correl_window", type=int, default=100000) # all available data
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data = pd.read_csv(os.path.join(args.inputs_path, f'{args.data_name}.csv'))
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date').sort_index()

    max_gas_features = 5  # intercept + up to 4 exog

    # Map fs_method to PyTimeVar's method string
    gas_method_map = {
        'gas-gaussian': 'gaussian',
        'gas-tstudent': 'student',
    }
    selected_gas_method = gas_method_map[args.pred_method]

    # add lags of target
    for lag in range(1, args.p + 1):
        data[f"{args.target}_lag{lag}"] = data[args.target].shift(lag)

    # keep args.target related columns only
    data = data[[col for col in data.columns if col.startswith(args.target)]].copy().dropna()

    predictions = []

    for step in tqdm(
        range(0, len(data) - args.estimation_window, 1),
        total=len(data) - args.estimation_window,
        desc=f"forecasting {args.pred_method}: {args.target}"
    ):
        if args.fix_start or (step == 0):
            start = 0
        else:
            start += 1

        train_end = args.estimation_window + step
        test_start = train_end
        test_end = train_end + 1

        train_df = data.iloc[start:train_end, :].copy()
        test_df = data.iloc[test_start:test_end, :].copy()

        if train_df.empty or test_df.empty:
            continue

        selected_columns = list(train_df.drop([args.target], axis=1).columns)
        train_df = train_df[[args.target] + selected_columns].copy()
        test_df = test_df[[args.target] + selected_columns].copy()

        mean = train_df.mean()
        std = train_df.std().replace(0, 1.0)

        train_df_z = (train_df - mean) / std
        test_df_z = (test_df - mean) / std

        if args.p == -1:
            try:
                var_select_model = VAR(train_df_z[[args.target] + selected_columns])
                selected_p = var_select_model.select_order(maxlags=args.max_p).selected_orders["aic"]
                if selected_p == 0 or selected_p is None:
                    selected_p = 1
            except Exception:
                selected_p = 1
        else:
            selected_p = min(args.p, args.max_p)

        try:
            # --------------------------------------------------
            # PREPARE GAS INPUTS
            # --------------------------------------------------
            vY_train = train_df_z[args.target].values.flatten()
            mX_train = np.hstack([
                np.ones((len(vY_train), 1)),
                train_df_z[selected_columns].values
            ])

            mX_train = np.clip(mX_train, -5, 5)
            vY_train = np.clip(vY_train, -5, 5)

            # --------------------------------------------------
            # FIT GAS
            # --------------------------------------------------
            gas_model = GAS(
                vY=vY_train,
                mX=mX_train,
                method=selected_gas_method,
            )

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_betas, res_params = gas_model.fit()

            if not np.all(np.isfinite(res_params)) or not np.all(np.isfinite(res_betas)):
                print(f"[{args.pred_method} step={step}] Non-finite params/betas, skipping")
                continue

            # --------------------------------------------------
            # EXTRACT LAST FILTERED STATE
            # --------------------------------------------------
            n_betas = res_betas.shape[0]
            beta_T = res_betas[:, -1]

            # --------------------------------------------------
            # FORECAST y_{T+1} = x_{T+1}' · β_T
            # --------------------------------------------------
            next_exog_values = test_df_z[selected_columns].values.flatten()
            next_exog_values = np.clip(next_exog_values, -5, 5)
            mX_next = np.hstack([1.0, next_exog_values])

            if len(mX_next) != len(beta_T):
                min_dim = min(len(mX_next), len(beta_T))
                mX_next = mX_next[:min_dim]
                beta_T = beta_T[:min_dim]

            ypred_zscore = float(np.dot(mX_next, beta_T))

            if not np.isfinite(ypred_zscore):
                print(f"[{args.pred_method} step={step}] Non-finite forecast, skipping")
                continue

            # --------------------------------------------------
            # STORE
            # --------------------------------------------------
            yt_true_zscore = float(test_df_z[args.target].iloc[-1])
            ypred = float(ypred_zscore * std[args.target] + mean[args.target])
            yt_true = float(test_df[args.target].iloc[-1])

            pred = pd.DataFrame([{
                "date": test_df.index[-1],
                "prediction_zscore": ypred_zscore,
                "true_zscore": yt_true_zscore,
                "prediction": ypred,
                "true": yt_true,
                "selected_p": selected_p,
                "n_features_model": n_betas,
                "n_exog_model": n_betas - 1,
            }])

            predictions.append(pred)

        except Exception as e:
            print(f"[{args.pred_method} step={step}] GAS fit/predict failed: {e}")
            continue

    # --------------------------------------------------
    # CONSOLIDATE
    # --------------------------------------------------
    if len(predictions) == 0:
        predictions_df = pd.DataFrame(columns=[
            "prediction_zscore", "true_zscore",
            "prediction", "true",
            "selected_p", "n_features_model", "n_exog_model"
        ])
    else:
        predictions_df = pd.concat(predictions, axis=0)

    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    predictions_df.set_index('date', inplace=True)

    results = {
        "predictions": predictions_df,
    }

    # check if results folder exists
    if not os.path.exists(os.path.join(args.outputs_path, args.pred_method, args.data_name)):
        os.makedirs(os.path.join(args.outputs_path, args.pred_method, args.data_name))
    
    # save results
    save_path = os.path.join(args.outputs_path, args.pred_method, args.data_name, "{}_{}_{}_{}.pickle".format(args.target,
                                                                                                              args.estimation_window,
                                                                                                              args.correl_window,
                                                                                                              args.p))
    save_pickle(path=save_path, obj=results)