import builtins
builtins.init = lambda: None

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
from tqdm import tqdm
import argparse

from utils.conn_data import save_pickle

# Suppress warnings
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', ConvergenceWarning)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_latest_transition_matrix(res, k_regimes):
    P = np.asarray(res.regime_transition)
    if P.ndim == 2:
        return P
    if P.ndim == 3:
        if P.shape[0] == k_regimes and P.shape[1] == k_regimes:
            return P[:, :, -1]
        if P.shape[-2] == k_regimes and P.shape[-1] == k_regimes:
            return P[-1, :, :]
    raise ValueError(f"Unexpected regime_transition shape: {P.shape}")


def get_next_step_probs(current_probs, transition_matrix):
    current_probs = np.asarray(current_probs).reshape(-1)
    P = np.asarray(transition_matrix)
    try:
        probs = current_probs @ P
    except Exception:
        probs = P.T @ current_probs
    probs = np.asarray(probs).reshape(-1)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    return probs


def get_filtered_probs(res):
    for attr in ['filtered_marginal_probabilities', 'filtered_joint_probabilities']:
        if hasattr(res, attr):
            probs = getattr(res, attr)
            if isinstance(probs, pd.DataFrame):
                return probs.iloc[-1].values.reshape(-1)
            elif isinstance(probs, np.ndarray):
                if probs.ndim == 2:
                    return probs[-1].reshape(-1)
                elif probs.ndim == 1:
                    return probs.reshape(-1)

    for attr in ['smoothed_marginal_probabilities']:
        if hasattr(res, attr):
            probs = getattr(res, attr)
            if isinstance(probs, pd.DataFrame):
                return probs.iloc[-1].values.reshape(-1)
            elif isinstance(probs, np.ndarray):
                if probs.ndim == 2:
                    return probs[-1].reshape(-1)
                elif probs.ndim == 1:
                    return probs.reshape(-1)

    raise ValueError("Could not extract regime probabilities from fitted model.")


def regime_specific_forecast(res, x_next, exog_cols, k_regimes):
    params = res.params.copy()

    if not isinstance(params, pd.Series):
        if hasattr(res.model, "param_names"):
            params = pd.Series(params, index=res.model.param_names)
        else:
            raise ValueError("Could not recover parameter names from fitted model.")

    if isinstance(x_next, pd.DataFrame):
        x_next = x_next.iloc[0]
    elif isinstance(x_next, np.ndarray):
        x_next = pd.Series(x_next, index=exog_cols)

    forecasts = []

    for i in range(k_regimes):
        mu_i = 0.0

        for key in [f'const[{i}]', f'intercept[{i}]', 'const', 'intercept']:
            if key in params.index:
                mu_i += float(params[key])
                break

        for j, col in enumerate(exog_cols):
            xval = float(x_next[col])

            regime_keys = [f'x{j+1}[{i}]', f'{col}[{i}]']
            shared_keys = [f'x{j+1}', col]

            coef = None
            for key in regime_keys:
                if key in params.index:
                    coef = float(params[key])
                    break
            if coef is None:
                for key in shared_keys:
                    if key in params.index:
                        coef = float(params[key])
                        break

            if coef is not None:
                mu_i += coef * xval

        forecasts.append(mu_i)

    return np.array(forecasts)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Markov-Switching Dynamic Regression forecasting")
    parser.add_argument("--target", type=str, default="SPY", help="Target variable to forecast")
    parser.add_argument("--pred_method", type=str, default="msdr", help="Prediction method name")
    parser.add_argument("--estimation_window", type=int, default=12*7, help="Estimation window size")
    parser.add_argument("--fix_start", action='store_true', help="Whether to fix the start of the estimation window")
    parser.add_argument("--p", type=int, default=1, help="Number of lags to include as features (-1 for automatic VAR selection)")
    parser.add_argument("--max_p", type=int, default=3, help="Maximum number of lags for VAR selection")
    parser.add_argument("--inputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "inputs"))
    parser.add_argument("--outputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "outputs"))
    parser.add_argument("--data_name", type=str, default="etfs_macro_large", help="Name of the dataset file")
    parser.add_argument("--correl_window", type=int, default=100000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = pd.read_csv(os.path.join(args.inputs_path, f'{args.data_name}.csv'))
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date').sort_index()

    # --------------------------------------------------
    # ADD LAG FEATURES
    # --------------------------------------------------
    for lag in range(1, args.max_p + 1):
        data[f'{args.target}_lag{lag}'] = data[args.target].shift(lag)

    data = data.dropna().copy()

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    predictions = []

    for step in tqdm(
        range(0, len(data) - args.estimation_window, 1),
        total=len(data) - args.estimation_window,
        desc=f"forecasting {args.pred_method}: {args.target}"
    ):
        if args.fix_start or step == 0:
            start = 0
        else:
            start += 1

        train_end = args.estimation_window + step
        test_idx = train_end

        if test_idx >= len(data):
            break

        train_df = data.iloc[start:train_end].copy()
        test_row = data.iloc[test_idx:test_idx + 1].copy()

        if train_df.empty or test_row.empty:
            continue

        # --------------------------------------------------
        # SELECT FEATURES
        # --------------------------------------------------
        lag_columns = [f'{args.target}_lag{i}' for i in range(1, args.max_p + 1)]
        selected_columns = lag_columns.copy()

        train_df = train_df[[args.target] + selected_columns].copy()
        test_row = test_row[[args.target] + selected_columns].copy()

        # --------------------------------------------------
        # ZSCORE NORMALIZATION
        # --------------------------------------------------
        mean = train_df.mean()
        std = train_df.std().replace(0, 1.0)

        train_z = (train_df - mean) / std
        test_z = (test_row - mean) / std

        # --------------------------------------------------
        # LAG ORDER SELECTION
        # --------------------------------------------------
        if args.p == -1:
            try:
                var_select_model = VAR(train_z[[args.target] + lag_columns])
                selected_orders = var_select_model.select_order(maxlags=args.max_p)
                selected_p = selected_orders.selected_orders.get("aic", None)
                if selected_p is None or selected_p <= 0:
                    selected_p = 1
            except Exception:
                selected_p = 1
        else:
            selected_p = min(args.p, args.max_p)

        selected_lag_cols = [f'{args.target}_lag{i}' for i in range(1, selected_p + 1)]

        Xt_train = train_z[selected_lag_cols].copy()
        yt_train = train_z[args.target].copy()

        Xt_test = test_z[selected_lag_cols].copy()
        yt_test = test_row[[args.target]].copy()
        yt_test_zscore = test_z[[args.target]].copy()

        # --------------------------------------------------
        # REGIME SELECTION
        # --------------------------------------------------
        k_regimes_trials = [2, 3]
        regime_models_summary = []

        for k_regimes in k_regimes_trials:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod_trial = sm.tsa.MarkovRegression(
                        yt_train,
                        exog=Xt_train,
                        trend='c',
                        k_regimes=k_regimes,
                        switching_variance=True,
                        switching_exog=True,
                    )
                    res_trial = mod_trial.fit(disp=False, maxiter=200)

                regime_models_summary.append({
                    "k_regimes": k_regimes,
                    "aic": res_trial.aic,
                    "bic": res_trial.bic,
                    "hqic": res_trial.hqic,
                    "result": res_trial,
                })
            except Exception:
                continue

        if len(regime_models_summary) == 0:
            continue

        regime_models_summary_df = pd.DataFrame(regime_models_summary)
        best_row = regime_models_summary_df.sort_values('bic').iloc[0]
        selected_k_regimes = int(best_row['k_regimes'])
        final_res_msdr = best_row['result']

        # --------------------------------------------------
        # FORECAST
        # --------------------------------------------------
        try:
            current_probs = get_filtered_probs(final_res_msdr)

            transition_matrix = get_latest_transition_matrix(final_res_msdr, selected_k_regimes)
            next_step_probs = get_next_step_probs(current_probs, transition_matrix)

            regime_forecasts = regime_specific_forecast(
                final_res_msdr,
                Xt_test.iloc[0],
                selected_lag_cols,
                selected_k_regimes
            )

            ypred_zscore = float((next_step_probs * regime_forecasts).sum())

            if not np.isfinite(ypred_zscore):
                print(f"[step={step}] Non-finite forecast, skipping")
                continue

        except Exception as e:
            print(f"[step={step}] Forecast failed: {e}")
            continue

        # back to original scale
        ypred = ypred_zscore * std[args.target] + mean[args.target]

        pred = pd.DataFrame([{
            "date": yt_test.index[-1],
            "prediction_zscore": ypred_zscore,
            "true_zscore": float(yt_test_zscore.iloc[-1, 0]),
            "prediction": float(ypred),
            "true": float(yt_test.iloc[-1, 0]),
            "selected_p": selected_p,
            "selected_k_regimes": selected_k_regimes,
        }])

        predictions.append(pred)

    # --------------------------------------------------
    # CONSOLIDATE
    # --------------------------------------------------
    if len(predictions) == 0:
        predictions_df = pd.DataFrame(columns=[
            "prediction_zscore", "true_zscore", "prediction", "true",
            "selected_p", "selected_k_regimes"
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
    save_path = os.path.join(
        args.outputs_path, args.pred_method, args.data_name,
        "{}_{}_{}_{}.pickle".format(
            args.target,
            args.estimation_window,
            args.correl_window,
            args.p
        )
    )
    save_pickle(path=save_path, obj=results)