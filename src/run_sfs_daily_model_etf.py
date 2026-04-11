"""
Daily-frequency prediction using feature sets learned by SFS (PBFS) methods.

Instead of re-running the sequential feature selection from scratch, this script:
  1. Loads an existing PBFS pickle from src/data/outputs/{fs_method}_nocluster/.
  2. Extracts the monthly feature sets from parents_of_target.
  3. For each out-of-sample month, holds the selected features fixed and
     predicts every trading day within that month using a rolling daily OLS
     on the daily-frequency subset of the learned features.

Daily-available features
------------------------
  ETF task   : features in etfs_large (SPY, XLI, …, XLB) from daily_etfs.csv.
               Macro features selected by SFS (FRED-MD) are skipped since they
               are not observable at daily frequency.

  Currency task : only ldEXME → mapped to EURCHF daily log-return.
               All monetary-policy features (dCMR, ldFCIr, …) are skipped.
               Effectively AR(1) unless ldEXME itself was selected.

Supported targets
-----------------
  --target SPY    --data_name etfs_macro_large
  --target EURCHF --data_name monetary-policy-processed

Output saved as:
  <outputs_path>/<fs_method>-daily_nocluster/<data_name>/
      <target>_<window>_<correl_window>_<p>.pickle
  Content: {'predictions': DataFrame (daily), 'parents_of_target': DataFrame,
            'args': args}
"""

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from metadata.etfs import etfs_large
from utils.conn_data import load_pickle, save_pickle
from utils.parsers import str_2_bool

# ---------------------------------------------------------------------------
TRADING_DAYS_PER_MONTH = 21
CURRENCY_TARGETS = {"EURCHF"}
# Mapping: monthly target name → daily column name
MONTHLY_TO_DAILY_TARGET = {"ldEXME": "EURCHF"}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Daily data loaders (identical to run_cdta_model_etf.py)
# ---------------------------------------------------------------------------

def build_daily_data_etf(inputs_path: str) -> pd.DataFrame:
    """Daily log-returns for etfs_large from daily_etfs.csv."""
    path = os.path.join(inputs_path, "daily_etfs.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    cols = [c for c in etfs_large if c in df.columns]
    return np.log(df[cols]).diff().dropna(how="all")


def build_daily_data_currency(eurchf_csv: str, target_col: str = "EURCHF") -> pd.DataFrame:
    """Daily log-returns for EUR/CHF from the yfinance CSV."""
    raw = pd.read_csv(eurchf_csv, header=[0, 1], index_col=0, parse_dates=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [" ".join(c).strip() for c in raw.columns]
    price_col = next(
        (c for c in raw.columns if "adj close" in c.lower()),
        next((c for c in raw.columns if "close" in c.lower()), None),
    )
    if price_col is None:
        raise ValueError(
            f"Close/Adj Close column not found in {eurchf_csv}. "
            f"Columns: {list(raw.columns)}"
        )
    prices = raw[[price_col]].rename(columns={price_col: target_col})
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return np.log(prices.dropna()).diff().dropna()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _monthly_parent_sets(parents_of_target: pd.DataFrame) -> dict:
    """
    Build a {month_timestamp: [fred_col, ...]} dict from parents_of_target.
    Uses the 'fred' column (base variable name without lag suffix).
    """
    parents_of_target = parents_of_target.copy()
    parents_of_target["date"] = pd.to_datetime(parents_of_target["date"])
    result = {}
    for month, grp in parents_of_target.groupby("date"):
        result[month] = grp["fred"].dropna().unique().tolist()
    return result


def _daily_available(monthly_parents: list, daily_returns: pd.DataFrame,
                     monthly_target: str, daily_target: str) -> list:
    """
    Filter a list of monthly-selected variable names to those available
    in daily_returns, remapping the monthly target name to its daily equivalent.

    Returns a list of column names present in daily_returns (always includes
    daily_target as an AR term).
    """
    available = set(daily_returns.columns)
    daily_cols = []
    for col in monthly_parents:
        # Remap monthly target name → daily target name
        mapped = MONTHLY_TO_DAILY_TARGET.get(col, col)
        if mapped in available and mapped != daily_target:
            daily_cols.append(mapped)
    # Always include the daily target itself (AR term)
    if daily_target not in daily_cols:
        daily_cols = [daily_target] + daily_cols
    else:
        daily_cols = [daily_target] + [c for c in daily_cols if c != daily_target]
    return daily_cols


def run_sfs_daily_forecast(
    monthly_predictions: pd.DataFrame,
    monthly_parent_sets: dict,
    daily_returns: pd.DataFrame,
    monthly_target: str,
    daily_target: str,
    daily_ols_window: int,
    intercept: bool,
):
    """
    For each out-of-sample month (rows of monthly_predictions):
      - Look up which features were selected that month.
      - Filter to daily-available features.
      - Predict every trading day in that month via rolling daily OLS.

    Parameters
    ----------
    monthly_predictions : DataFrame
        Original monthly predictions (DatetimeIndex, month-start).
        Used only to enumerate the out-of-sample months.
    monthly_parent_sets : dict
        {month_timestamp: [fred_col, ...]} from parents_of_target.
    daily_returns : DataFrame
        Daily log-returns, must contain daily_target column.
    monthly_target : str
        Target name as used in the monthly pickle (e.g. 'SPY', 'ldEXME').
    daily_target : str
        Target name in daily_returns (e.g. 'SPY', 'EURCHF').
    daily_ols_window : int
        Rolling window of trading days for each daily OLS fit.

    Returns
    -------
    dict with keys 'predictions' (daily DataFrame) and 'parents_of_target'.
    """
    predictions = []
    parents_log = []

    for month in tqdm(monthly_predictions.index, desc=f"SFS daily: {daily_target}"):
        # Retrieve the feature set selected for this month; fall back to AR(1)
        raw_parents = monthly_parent_sets.get(month, [monthly_target])
        feat_cols = _daily_available(raw_parents, daily_returns, monthly_target, daily_target)

        # Trading days within this month in daily_returns
        month_mask = (
            (daily_returns.index.year  == month.year) &
            (daily_returns.index.month == month.month)
        )
        month_days = daily_returns.index[month_mask]

        # Log the monthly parent selection (once per month)
        parents_log.append(pd.DataFrame({
            "date":     [month] * len(feat_cols),
            "variable": feat_cols,
            "value":    1,
        }))

        for day in month_days:
            day_iloc = daily_returns.index.get_loc(day)
            if day_iloc < 2:
                continue

            # Rolling training window: [day_iloc - daily_ols_window, day_iloc)
            t_start = max(0, day_iloc - daily_ols_window)
            window  = daily_returns.iloc[t_start:day_iloc][feat_cols].dropna()

            if len(window) < 5:
                pred_z = pred_raw = 0.0
            else:
                mean_d = window.mean()
                std_d  = window.std().replace(0, 1e-8)
                wz     = (window - mean_d) / std_d

                # Lag-1 OLS: X[:-1] → y[1:]
                X_tr = wz[feat_cols].iloc[:-1].values
                y_tr = wz[daily_target].iloc[1:].values
                X_te = wz[feat_cols].iloc[[-1]].values   # features for `day`

                if intercept:
                    ones = np.ones((len(X_tr), 1))
                    X_tr = np.hstack([ones, X_tr])
                    X_te = np.hstack([[1.0], X_te.ravel()]).reshape(1, -1)

                try:
                    ols      = sm.OLS(y_tr, X_tr).fit()
                    pred_z   = float(ols.predict(X_te)[0])
                    pred_raw = pred_z * float(std_d[daily_target]) + float(mean_d[daily_target])
                except Exception:
                    pred_z = pred_raw = 0.0

            true_raw = float(daily_returns.at[day, daily_target])
            win_tgt  = daily_returns.iloc[t_start:day_iloc][daily_target]
            mu_t, sg_t = float(win_tgt.mean()), float(win_tgt.std())
            sg_t = sg_t if sg_t > 0 else 1e-8
            true_z = (true_raw - mu_t) / sg_t

            predictions.append({
                "date":               day,
                "prediction_zscore":  pred_z,
                "true_zscore":        true_z,
                "prediction":         pred_raw,
                "true":               true_raw,
            })

    predictions_df = pd.DataFrame(predictions)
    if not predictions_df.empty:
        predictions_df["date"] = pd.to_datetime(predictions_df["date"])
        predictions_df = predictions_df.set_index("date")
    else:
        predictions_df = pd.DataFrame(
            columns=["prediction_zscore", "true_zscore", "prediction", "true"]
        )
        predictions_df.index.name = "date"

    parents_df = (
        pd.concat(parents_log, axis=0)
        if parents_log
        else pd.DataFrame(columns=["date", "variable", "value"])
    )

    return {"predictions": predictions_df, "parents_of_target": parents_df}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Apply monthly SFS feature sets to daily data for step-ahead prediction. "
        "Loads an existing PBFS pickle and re-forecasts at daily frequency."
    )
)
parser.add_argument(
    "--fs_method", type=str, required=True,
    help="SFS method name without '_nocluster' suffix, e.g. sfscv-backward-svm",
)
parser.add_argument(
    "--target", type=str, default="SPY",
    help="Target variable: 'SPY' for ETF task, 'EURCHF' for currency task.",
)
parser.add_argument(
    "--data_name", type=str, default=None,
    help="Dataset sub-folder used in the source pickle path. "
         "Defaults to 'etfs_macro_large' for SPY and 'monetary-policy-processed' for EURCHF.",
)
parser.add_argument("--estimation_window", type=int, default=84)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--correl_window", type=int, default=100000)
parser.add_argument("--daily_ols_window", type=int, default=252,
                    help="Rolling window of trading days for daily OLS.")
parser.add_argument("--intercept", type=str, default="True")
parser.add_argument("--eurchf_csv", type=str, default=None,
                    help="Path to daily EUR/CHF CSV. "
                         "Defaults to <inputs_path>/eurchf_yahoo_daily.csv")
parser.add_argument(
    "--inputs_path", type=str,
    default=os.path.join(os.path.dirname(__file__), "data", "inputs"),
)
parser.add_argument(
    "--outputs_path", type=str,
    default=os.path.join(os.path.dirname(__file__), "data", "outputs"),
)

if __name__ == "__main__":
    args = parser.parse_args()
    args.intercept = str_2_bool(args.intercept)

    is_currency = args.target.upper() in CURRENCY_TARGETS
    target_label = args.target.upper() if is_currency else args.target

    # --- Resolve data_name and source pickle target name -------------------
    if args.data_name is None:
        args.data_name = (
            "monetary-policy-processed" if is_currency else "etfs_macro_large"
        )

    # The monthly pickle may use a different target name (e.g. ldEXME for EURCHF)
    if is_currency:
        monthly_target = "ldEXME"
        # Currency pickle uses p=-1 by convention
        src_p = -1
    else:
        monthly_target = args.target
        src_p = args.p

    # --- Load source PBFS pickle -------------------------------------------
    src_dir    = os.path.join(args.outputs_path, f"{args.fs_method}_nocluster", args.data_name)
    src_file   = os.path.join(
        src_dir,
        f"{monthly_target}_{args.estimation_window}_{args.correl_window}_{src_p}.pickle",
    )
    if not os.path.exists(src_file):
        raise FileNotFoundError(
            f"Source pickle not found: {src_file}\n"
            f"Available files: {os.listdir(src_dir) if os.path.isdir(src_dir) else 'directory not found'}"
        )

    print(f"Loading PBFS pickle: {src_file}")
    src_obj = load_pickle(src_file)
    monthly_predictions = src_obj["predictions"]
    parent_sets         = _monthly_parent_sets(src_obj["parents_of_target"])
    print(f"  Monthly predictions: {len(monthly_predictions)} months, "
          f"{monthly_predictions.index[0].date()} – {monthly_predictions.index[-1].date()}")

    # --- Build daily returns -----------------------------------------------
    if is_currency:
        eurchf_csv = args.eurchf_csv or os.path.join(args.inputs_path, "eurchf_yahoo_daily.csv")
        print(f"Building daily EUR/CHF returns from {eurchf_csv} ...")
        daily_returns = build_daily_data_currency(eurchf_csv, target_col=target_label)
    else:
        print("Building daily ETF returns from daily_etfs.csv ...")
        daily_returns = build_daily_data_etf(args.inputs_path)

    print(f"  Daily returns: {daily_returns.shape}, "
          f"{daily_returns.index[0].date()} – {daily_returns.index[-1].date()}")

    # --- Run daily forecast -------------------------------------------------
    results = run_sfs_daily_forecast(
        monthly_predictions=monthly_predictions,
        monthly_parent_sets=parent_sets,
        daily_returns=daily_returns,
        monthly_target=monthly_target,
        daily_target=target_label,
        daily_ols_window=args.daily_ols_window,
        intercept=args.intercept,
    )
    results["args"] = args

    # --- Save ---------------------------------------------------------------
    out_dir = os.path.join(
        args.outputs_path, f"{args.fs_method}-daily_nocluster", args.data_name
    )
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"{target_label}_{args.estimation_window}_{args.correl_window}_{args.p}.pickle",
    )
    save_pickle(path=out_file, obj=results)
    print(f"Saved to: {out_file}")
    print(f"Daily predictions: {results['predictions'].shape}")
    print(results["predictions"].head())
