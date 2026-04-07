import builtins
builtins.init = lambda: None

import os
import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from utils.parsers import add_and_keep_lags_only

from utils.conn_data import save_pickle

# Suppress warnings
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', ConvergenceWarning)


def rf_cv_random_search(X_train, y_train, n_candidates=50, n_splits=5, random_state=42):
    """
    Random search over academic-standard RF hyperparameter space
    with TimeSeriesSplit CV.

    Hyperparameter ranges follow Gu, Kelly & Xiu (2020, RFS),
    Bianchi, Büchner & Tamoni (2021), and Coulombe et al. (2022):

    - n_estimators:    {100, 300, 500, 1000}
    - max_depth:       {1, 2, 3, 4, 5, 6, None}
    - max_features:    {1/3, 1/2, sqrt, log2, 1.0}
    - min_samples_split: {2, 5, 10, 20}
    - min_samples_leaf:  {1, 2, 5, 10}
    - bootstrap:       {True}

    Returns the best hyperparameter dict and fitted model is done outside.
    """
    param_grid = {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [1, 2, 3, 4, 5, 6, None],
        'max_features': [1/3, 0.5, 'sqrt', 'log2', 1.0],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
    }

    rng = np.random.RandomState(random_state)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Determine valid max_features given dimensionality
    n_features = X_train.shape[1]
    valid_max_features = []
    for mf in param_grid['max_features']:
        if isinstance(mf, float):
            if int(mf * n_features) >= 1:
                valid_max_features.append(mf)
        elif isinstance(mf, str):
            valid_max_features.append(mf)
    if len(valid_max_features) == 0:
        valid_max_features = [1.0]
    param_grid['max_features'] = valid_max_features

    best_params = None
    best_mse = np.inf

    for _ in range(n_candidates):
        params = {
            'n_estimators': rng.choice(param_grid['n_estimators']),
            'max_depth': rng.choice(param_grid['max_depth']),
            'max_features': rng.choice(param_grid['max_features']),
            'min_samples_split': int(rng.choice(param_grid['min_samples_split'])),
            'min_samples_leaf': int(rng.choice(param_grid['min_samples_leaf'])),
            'bootstrap': True,
            'random_state': random_state,
            'n_jobs': -1,
        }

        # Handle None for max_depth (numpy converts it)
        if params['max_depth'] is not None:
            params['max_depth'] = int(params['max_depth'])

        fold_mses = []
        try:
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = RandomForestRegressor(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                fold_mses.append(mean_squared_error(y_val, y_pred))

            avg_mse = np.mean(fold_mses)

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = params.copy()
        except Exception:
            continue

    if best_params is None:
        # Fallback to conservative defaults from Gu et al. (2020)
        best_params = {
            'n_estimators': 300,
            'max_depth': 3,
            'max_features': 'sqrt',
            'min_samples_split': 5,
            'min_samples_leaf': 5,
            'bootstrap': True,
            'random_state': random_state,
            'n_jobs': -1,
        }

    return best_params


def parse_args():
    parser = argparse.ArgumentParser(description="Run Random Forest forecasting with TimeSeriesSplit CV")
    parser.add_argument("--target", type=str, default="SPY", help="Target variable to forecast")
    parser.add_argument("--pred_method", type=str, default="rf-tscv", help="Prediction method name")
    parser.add_argument("--estimation_window", type=int, default=12*7, help="Estimation window size")
    parser.add_argument("--fix_start", action='store_true', help="Whether to fix the start of the estimation window")
    parser.add_argument("--p", type=int, default=1, help="Number of lags to include as features (-1 for automatic VAR selection)")
    parser.add_argument("--max_p", type=int, default=3, help="Maximum number of lags for VAR selection")
    parser.add_argument("--n_candidates", type=int, default=50, help="Number of random hyperparameter configurations to evaluate")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of TimeSeriesSplit folds")
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

    data = data.drop(['XLI', 'XLE', 'XLK', 'XLV', 'XLU', 'XLF', 'XLY', 'XLP', 'XLB'], axis=1)

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
        test_df = data.iloc[test_start-1:test_end, :].copy()

        if args.p == -1:
            try:
                var_select_model = VAR(train_df[[args.target] + list(train_df.columns)])
                selected_p = var_select_model.select_order(maxlags=args.max_p).selected_orders["aic"]
                if selected_p == 0 or selected_p is None:
                    selected_p = 1
            except Exception:
                selected_p = 1
        else:
            selected_p = min(args.p, args.max_p)

        y_train = train_df[[args.target]].copy()
        y_test = test_df[[args.target]].copy()

        train_df = add_and_keep_lags_only(data=train_df, lags=selected_p)
        test_df = add_and_keep_lags_only(data=test_df, lags=selected_p)

        train_df = pd.concat([y_train, train_df], axis=1).dropna()
        test_df = pd.concat([y_test, test_df], axis=1).dropna()

        if train_df.empty or test_df.empty:
            continue

        selected_columns = list(train_df.drop([args.target], axis=1).columns)
        train_df = train_df[[args.target] + selected_columns].copy()
        test_df = test_df[[args.target] + selected_columns].copy()

        mean = train_df.mean()
        std = train_df.std().replace(0, 1.0)

        train_df_z = (train_df - mean) / std
        test_df_z = (test_df - mean) / std

        try:
            # --------------------------------------------------
            # PREPARE INPUTS
            # --------------------------------------------------
            X_train = train_df_z[selected_columns].values
            y_train_arr = train_df_z[args.target].values

            X_test = test_df_z[selected_columns].values

            # --------------------------------------------------
            # RF CV: random search over academic hyperparameter space
            # --------------------------------------------------
            best_params = rf_cv_random_search(
                X_train, y_train_arr,
                n_candidates=args.n_candidates,
                n_splits=args.n_splits,
                random_state=19940202,
            )

            # --------------------------------------------------
            # FIT FINAL MODEL
            # --------------------------------------------------
            final_model = RandomForestRegressor(**best_params)
            final_model.fit(X_train, y_train_arr)

            ypred_zscore = float(final_model.predict(X_test).flatten()[0])

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
                "best_n_estimators": best_params['n_estimators'],
                "best_max_depth": best_params['max_depth'],
                "best_max_features": str(best_params['max_features']),
                "best_min_samples_split": best_params['min_samples_split'],
                "best_min_samples_leaf": best_params['min_samples_leaf'],
                "n_features": X_train.shape[1],
            }])

            predictions.append(pred)

        except Exception as e:
            print(f"[{args.pred_method} step={step}] RF fit/predict failed: {e}")
            continue

    # --------------------------------------------------
    # CONSOLIDATE
    # --------------------------------------------------
    if len(predictions) == 0:
        predictions_df = pd.DataFrame(columns=[
            "prediction_zscore", "true_zscore",
            "prediction", "true",
            "selected_p", "best_n_estimators", "best_max_depth",
            "best_max_features", "best_min_samples_split",
            "best_min_samples_leaf", "n_features"
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