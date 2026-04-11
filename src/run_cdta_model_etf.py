"""
Runner for Gong et al. 2017 CDTA (Causal Discovery from Temporally Aggregated
time series) on ETF + macro data, and on EUR/CHF + macro data.

Reference: R. Gong, D. Janzing, B. Schölkopf, "Causal Discovery from
Temporally Aggregated Time Series", UAI 2017.

Supported targets
-----------------
  --target SPY     : daily_etfs.csv → monthly mean price → log-return
                     merged with FRED-MD macro (shifted +1 month)
                     CDTA causal structure learned monthly; predictions at
                     daily frequency using daily ETF log-returns.
                     output → cdta_nocluster/etfs_macro_large/

  --target EURCHF  : eurchf_yahoo_daily.csv (from download_eurchf_yahoo.py) →
                     monthly mean price → log-return merged with
                     monetary-policy-processed.csv (ldEXME dropped, +1 month shift)
                     CDTA causal structure learned monthly; predictions at
                     daily frequency using daily EUR/CHF log-returns.
                     output → cdta_nocluster/monetary_policy/

Pipeline
--------
  Monthly loop (each step advances one month):
    1. LASSO pre-filter on z-scored monthly data (≤ max_features)
    2. VAR(p) on filtered data → residuals
    3. Scale residuals by √k (aggregation correction) → DirectLiNGAM → B matrix
    4. Extract parents of target from B; add target itself as AR parent

  Daily prediction (for every trading day in the out-of-sample month):
    5. Among the monthly parents, keep those available at daily frequency
    6. Rolling daily OLS (last daily_ols_window trading days): lag-1 of each
       parent predicts today's return
    7. Record prediction vs actual at daily granularity

Output saved as:
  <outputs_path>/cdta_nocluster/<data_name>/<target>_<window>_<correl_window>_<p>.pickle
  Content: {'predictions': DataFrame (daily), 'parents_of_target': DataFrame,
            'dags': dict, 'args': args}
"""

import argparse
import os
import pickle
import warnings

import lingam
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tsa.api import VAR
from tqdm import tqdm

from metadata.etfs import etfs_large
from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

# ---------------------------------------------------------------------------
# Aggregation factor: approximate number of trading days in a calendar month
# Used to correct the variance-shrinkage introduced by mean aggregation.
TRADING_DAYS_PER_MONTH = 21
# ---------------------------------------------------------------------------


def build_dataset(inputs_path: str, etf_target: str):
    """
    1. Load daily ETF prices → monthly mean price → log-return.
    2. Load FRED-MD already-transformed data → shift forward by 1 month to
       avoid look-ahead bias (macro published with a lag).
    3. Inner-join on date.

    Returns a pd.DataFrame with DatetimeIndex (month-end or month-start
    consistent with FRED-MD), columns = [etf_target, ...ETFs..., ...macro...].
    """
    # --- ETF prices --------------------------------------------------------
    etf_path = os.path.join(inputs_path, "daily_etfs.csv")
    etf_daily = pd.read_csv(etf_path)
    if "Unnamed: 0" in etf_daily.columns:
        etf_daily = etf_daily.drop("Unnamed: 0", axis=1)
    etf_daily["date"] = pd.to_datetime(etf_daily["date"])
    etf_daily = etf_daily.set_index("date")

    # Monthly mean price, then log-return
    etf_monthly_price = etf_daily.resample("MS").mean()   # month-start label
    etf_monthly_ret = np.log(etf_monthly_price).diff()
    etf_monthly_ret = etf_monthly_ret.dropna(how="all")

    # Keep only the ETFs used in the paper (etfs_large) to avoid newer tickers
    # (e.g. XLC, USHY launched post-2018) from truncating the sample via dropna.
    etf_cols_to_keep = [c for c in etfs_large if c in etf_monthly_ret.columns]
    etf_monthly_ret = etf_monthly_ret[etf_cols_to_keep].dropna(axis=1, how="all")

    # --- FRED-MD macro (shared loader) -------------------------------------
    macro = _load_macro(inputs_path)

    # --- Merge -------------------------------------------------------------
    data = etf_monthly_ret.join(macro, how="inner")

    # Make sure target is present
    if etf_target not in data.columns:
        raise ValueError(
            f"Target column '{etf_target}' not found in ETF data. "
            f"Available ETF columns: {list(etf_monthly_ret.columns)}"
        )

    # Drop rows with any NaN
    data = data.dropna()

    # Put target first
    cols = [etf_target] + [c for c in data.columns if c != etf_target]
    data = data[cols]

    return data


def _load_macro(inputs_path: str) -> pd.DataFrame:
    """
    Load FRED-MD transformed data, normalise to month-start index, and shift
    forward by 1 month to avoid publication-lag look-ahead bias.
    Shared by both the ETF and currency dataset builders.
    """
    macro_path = os.path.join(inputs_path, "fredmd_transf_df.csv")
    macro = pd.read_csv(macro_path)
    if "Unnamed: 0" in macro.columns:
        macro = macro.drop("Unnamed: 0", axis=1)

    date_col = [c for c in macro.columns if c.lower() == "date"]
    if date_col:
        macro["date"] = pd.to_datetime(macro[date_col[0]])
        macro = macro.drop(date_col[0], axis=1) if date_col[0] != "date" else macro
        macro = macro.set_index("date")
    else:
        macro.index = pd.to_datetime(macro.index)

    macro.index = macro.index.to_period("M").to_timestamp()
    macro = macro.shift(1)
    macro = macro.dropna(how="all")
    macro = macro.loc[:, macro.isna().mean() < 0.5]
    macro = macro.fillna(method="ffill").dropna()
    return macro


def build_dataset_currency(
    inputs_path: str,
    eurchf_csv: str,
    target_col: str = "EURCHF",
) -> pd.DataFrame:
    """
    Build monthly dataset for the EUR/CHF experiment.

    Parameters
    ----------
    inputs_path : str
        Directory containing monetary-policy-processed.csv.
    eurchf_csv : str
        Path to the daily EUR/CHF CSV produced by download_eurchf_yahoo.py.
        Expected format: Date index + yfinance columns
        (Open, High, Low, Close, Adj Close, Volume).
    target_col : str
        Name given to the EUR/CHF column in the output DataFrame (default 'EURCHF').

    Pipeline
    --------
    1. Load daily EUR/CHF prices → monthly mean price → log-return.
    2. Load monetary-policy-processed.csv, drop ldEXME (replaced by Yahoo series),
       shift +1 month to avoid publication-lag bias.
    3. Inner-join on date.
    """
    # --- EUR/CHF daily → monthly log-return --------------------------------
    eurchf_daily = pd.read_csv(eurchf_csv, header=[0, 1], index_col=0, parse_dates=True)

    # yfinance CSVs may have a two-level header; flatten to get "Close"/"Adj Close"
    if isinstance(eurchf_daily.columns, pd.MultiIndex):
        eurchf_daily.columns = [" ".join(c).strip() for c in eurchf_daily.columns]

    # Pick Adj Close if present, otherwise Close
    price_col = next(
        (c for c in eurchf_daily.columns if "adj close" in c.lower()),
        next((c for c in eurchf_daily.columns if "close" in c.lower()), None),
    )
    if price_col is None:
        raise ValueError(
            f"Could not find a Close/Adj Close column in {eurchf_csv}. "
            f"Columns found: {list(eurchf_daily.columns)}"
        )

    eurchf_daily = eurchf_daily[[price_col]].rename(columns={price_col: target_col})
    eurchf_daily.index = pd.to_datetime(eurchf_daily.index)
    eurchf_daily.index.name = "date"
    eurchf_daily = eurchf_daily.dropna()

    eurchf_monthly = np.log(eurchf_daily.resample("MS").mean()).diff().dropna()

    # --- Monetary-policy features ------------------------------------------
    mp_path = os.path.join(inputs_path, "monetary-policy-processed.csv")
    mp = pd.read_csv(mp_path)
    if "Unnamed: 0" in mp.columns:
        mp = mp.drop("Unnamed: 0", axis=1)
    mp["date"] = pd.to_datetime(mp["date"])
    mp = mp.set_index("date")
    mp.index = mp.index.to_period("M").to_timestamp()

    # Drop ldEXME — replaced by the freshly aggregated Yahoo Finance series
    mp = mp.drop(columns=["ldEXME"], errors="ignore")

    # Shift +1 month (same publication-lag correction as FRED-MD)
    mp = mp.shift(1).dropna(how="all")

    # --- Merge -------------------------------------------------------------
    data = eurchf_monthly.join(mp, how="inner").dropna()
    cols = [target_col] + [c for c in data.columns if c != target_col]
    return data[cols]


def build_daily_data_etf(inputs_path: str) -> pd.DataFrame:
    """
    Load daily ETF prices from daily_etfs.csv and return daily log-returns
    for the etfs_large universe. Used as the daily prediction dataset for SPY.
    """
    etf_path = os.path.join(inputs_path, "daily_etfs.csv")
    df = pd.read_csv(etf_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    cols = [c for c in etfs_large if c in df.columns]
    df = df[cols]
    daily_ret = np.log(df).diff().dropna(how="all")
    return daily_ret


def build_daily_data_currency(eurchf_csv: str, target_col: str = "EURCHF") -> pd.DataFrame:
    """
    Load the daily EUR/CHF CSV produced by download_eurchf_yahoo.py and return
    daily log-returns as a single-column DataFrame.
    """
    raw = pd.read_csv(eurchf_csv, header=[0, 1], index_col=0, parse_dates=True)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [" ".join(c).strip() for c in raw.columns]

    price_col = next(
        (c for c in raw.columns if "adj close" in c.lower()),
        next((c for c in raw.columns if "close" in c.lower()), None),
    )
    if price_col is None:
        raise ValueError(
            f"Could not find Close/Adj Close column in {eurchf_csv}. "
            f"Columns: {list(raw.columns)}"
        )

    prices = raw[[price_col]].rename(columns={price_col: target_col})
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices = prices.dropna()
    daily_ret = np.log(prices).diff().dropna()
    return daily_ret


def lasso_prefilter(train_z: pd.DataFrame, target: str, max_features: int = 20):
    """
    Use LassoCV to reduce the feature set before running CDTA.
    Returns the list of columns to keep (including target).
    """
    y = train_z[target].values
    X = train_z.drop(target, axis=1)
    alphas = np.linspace(0.001, 0.5, 50)
    try:
        # Treat non-convergence as a handled failure and fallback robustly.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            lasso = LassoCV(
                cv=5,
                max_iter=20000,
                tol=1e-3,
                alphas=alphas,
                random_state=42,
            ).fit(X, y)
        coefs = pd.Series(np.abs(lasso.coef_), index=X.columns)
        selected = coefs[coefs > 0].nlargest(max_features).index.tolist()
    except (ConvergenceWarning, ValueError, FloatingPointError):
        # fallback: top max_features by absolute correlation
        corr = train_z.drop(target, axis=1).corrwith(train_z[target]).abs()
        selected = corr.nlargest(max_features).index.tolist()
    if not selected:
        corr = train_z.drop(target, axis=1).corrwith(train_z[target]).abs()
        selected = corr.nlargest(max_features).index.tolist()
    return [target] + selected


def _learn_causal_parents(
    train_monthly_z: pd.DataFrame,
    target: str,
    selected_p: int,
    beta_threshold: float,
    k_agg: int,
) -> tuple:
    """
    Fit VAR(p) on z-scored monthly data, apply CDTA aggregation correction,
    run DirectLiNGAM, and return (parent_cols, B_df).
    parent_cols always includes target itself (AR term).
    B_df is None on failure.
    """
    # --- VAR residuals -------------------------------------------------------
    try:
        var_fit = VAR(train_monthly_z).fit(maxlags=selected_p, ic=None)
        residuals = pd.DataFrame(var_fit.resid, columns=train_monthly_z.columns)
    except Exception:
        residuals_list = []
        for col in train_monthly_z.columns:
            y_col = train_monthly_z[col].iloc[selected_p:]
            X_lags = pd.concat(
                [train_monthly_z.shift(l).iloc[selected_p:] for l in range(1, selected_p + 1)],
                axis=1,
            )
            X_lags.columns = [
                f"{c}(t-{l})"
                for l in range(1, selected_p + 1)
                for c in train_monthly_z.columns
            ]
            ols_fit = sm.OLS(y_col, sm.add_constant(X_lags)).fit()
            residuals_list.append(ols_fit.resid.rename(col))
        residuals = pd.concat(residuals_list, axis=1)

    # --- CDTA aggregation correction -----------------------------------------
    residuals_corrected = residuals * np.sqrt(k_agg)

    # --- DirectLiNGAM --------------------------------------------------------
    B_df = None
    parent_cols = []
    try:
        dlm = lingam.DirectLiNGAM()
        dlm.fit(residuals_corrected.values)
        B = dlm.adjacency_matrix_
        B_df = pd.DataFrame(
            B, index=train_monthly_z.columns, columns=train_monthly_z.columns
        )
        target_row = B_df.loc[target]
        parent_cols = target_row[
            (np.abs(target_row) > beta_threshold) & (target_row.index != target)
        ].index.tolist()
    except Exception:
        pass

    # Always include target as AR parent
    if target not in parent_cols:
        parent_cols.append(target)

    return parent_cols, B_df


def run_cdta_forecast_daily(
    monthly_data: pd.DataFrame,
    daily_returns: pd.DataFrame,
    target: str,
    estimation_window: int,
    daily_ols_window: int,
    p: int,
    beta_threshold: float,
    fix_start: bool,
    intercept: bool,
    k_agg: int = TRADING_DAYS_PER_MONTH,
    max_features: int = 20,
):
    """
    Rolling CDTA pipeline: causal structure learned monthly, predictions daily.

    For each monthly step:
      - Learn causal parents of `target` from `monthly_data` using CDTA.
      - Hold the parent set fixed for the entire out-of-sample month.
      - For every trading day in that month (from `daily_returns`):
          * Fit rolling OLS on the past `daily_ols_window` trading days.
          * Features = lag-1 daily log-returns of parents available in
            `daily_returns` (macro parents, not observable daily, are dropped).
          * Predict today's return; record vs actual.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        Monthly merged dataset (target + features), DatetimeIndex month-start.
    daily_returns : pd.DataFrame
        Daily log-returns, DatetimeIndex. Must contain `target` column and
        any daily-frequency parent candidates (ETFs for SPY; none for EURCHF).
    daily_ols_window : int
        Number of trading days used to fit each daily OLS (default 252).

    Returns
    -------
    dict with keys 'predictions' (daily DataFrame), 'parents_of_target', 'dags'.
    """
    predictions = []
    parents_of_target = []
    dags = {}
    selected_p = max(1, p)

    n_steps = len(monthly_data) - estimation_window
    for step in tqdm(range(n_steps), total=n_steps, desc=f"CDTA daily: {target}"):
        if fix_start or step == 0:
            start = 0
        else:
            start += 1

        # ------------------------------------------------------------------
        # 1. Monthly causal structure
        # ------------------------------------------------------------------
        train_monthly = monthly_data.iloc[start : estimation_window + step].copy()
        mean_m = train_monthly.mean()
        std_m  = train_monthly.std().replace(0, 1e-8)
        train_z = (train_monthly - mean_m) / std_m

        keep_cols = lasso_prefilter(train_z, target, max_features=max_features)
        train_z_sub = train_z[keep_cols]

        parent_cols, B_df = _learn_causal_parents(
            train_z_sub, target, selected_p, beta_threshold, k_agg
        )

        # ------------------------------------------------------------------
        # 2. Record monthly DAG + parent log
        # ------------------------------------------------------------------
        out_month = monthly_data.index[estimation_window + step]

        if B_df is not None:
            dags[train_monthly.index[-1].strftime("%Y%m%d")] = {
                "B": B_df, "threshold": beta_threshold,
            }

        parents_of_target.append(pd.DataFrame({
            "date":     [out_month] * len(parent_cols),
            "variable": parent_cols,
            "value":    1,
        }))

        # ------------------------------------------------------------------
        # 3. Daily predictions for all trading days in out_month
        # ------------------------------------------------------------------
        # Parents available at daily frequency
        daily_feat_cols = [c for c in parent_cols if c in daily_returns.columns and c != target]
        # Always use target's own lag (AR)
        daily_all_cols = [target] + daily_feat_cols   # target first

        # Trading days that fall in out_month and exist in daily_returns
        month_mask = (
            (daily_returns.index.year  == out_month.year) &
            (daily_returns.index.month == out_month.month)
        )
        month_days = daily_returns.index[month_mask]

        for day in month_days:
            day_iloc = daily_returns.index.get_loc(day)
            if day_iloc < 2:
                continue  # need at least 2 days for a lag

            # Rolling training window: [day_iloc - daily_ols_window, day_iloc)
            train_start = max(0, day_iloc - daily_ols_window)
            window = daily_returns.iloc[train_start:day_iloc][daily_all_cols].dropna()

            if len(window) < 5:
                pred_z, pred_raw = 0.0, 0.0
            else:
                mean_d = window.mean()
                std_d  = window.std().replace(0, 1e-8)
                window_z = (window - mean_d) / std_d

                # Lag-1 prediction: X[0..n-2] → y[1..n-1]
                X_tr = window_z[daily_all_cols].iloc[:-1].values
                y_tr = window_z[target].iloc[1:].values
                X_te = window_z[daily_all_cols].iloc[[-1]].values  # lag features for `day`

                if intercept:
                    ones = np.ones((len(X_tr), 1))
                    X_tr = np.hstack([ones, X_tr])
                    X_te = np.hstack([[1.0], X_te.ravel()]).reshape(1, -1)

                try:
                    ols     = sm.OLS(y_tr, X_tr).fit()
                    pred_z  = float(ols.predict(X_te)[0])
                    pred_raw = pred_z * float(std_d[target]) + float(mean_d[target])
                except Exception:
                    pred_z, pred_raw = 0.0, 0.0

            true_raw = float(daily_returns.at[day, target])
            mean_d_t = daily_returns.iloc[max(0, day_iloc - daily_ols_window):day_iloc][target].mean()
            std_d_t  = daily_returns.iloc[max(0, day_iloc - daily_ols_window):day_iloc][target].std()
            std_d_t  = std_d_t if std_d_t > 0 else 1e-8
            true_z   = (true_raw - mean_d_t) / std_d_t

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
        pd.concat(parents_of_target, axis=0)
        if parents_of_target
        else pd.DataFrame(columns=["date", "variable", "value"])
    )

    return {
        "predictions":       predictions_df,
        "parents_of_target": parents_df,
        "dags":              dags,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Targets that use the EUR/CHF Yahoo Finance pipeline instead of daily_etfs.csv
CURRENCY_TARGETS = {"EURCHF"}

parser = argparse.ArgumentParser(
    description="Run CDTA forecast. Supports SPY (ETF) and EURCHF (currency) targets."
)
parser.add_argument("--estimation_window", type=int, default=12 * 7)  # 84 months
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--correl_window", type=int, default=100000)  # kept for filename convention
parser.add_argument("--beta_threshold", type=float, default=0.0)
parser.add_argument("--fix_start", type=str, default="True")
parser.add_argument("--intercept", type=str, default="True")
parser.add_argument("--target", type=str, default="SPY",
                    help="Target variable. Use 'SPY' for ETF or 'EURCHF' for EUR/CHF.")
parser.add_argument("--k_agg", type=int, default=TRADING_DAYS_PER_MONTH,
                    help="Aggregation factor: trading days per month")
parser.add_argument("--max_features", type=int, default=20,
                    help="Max features to keep after LASSO pre-filter before CDTA")
parser.add_argument("--daily_ols_window", type=int, default=252,
                    help="Rolling window of trading days used to fit each daily OLS")
parser.add_argument("--data_name", type=str, default=None,
                    help="Output sub-folder name. Defaults to 'etfs_macro_large' for SPY "
                         "and 'monetary_policy' for EURCHF.")
# EUR/CHF-specific option
parser.add_argument("--eurchf_csv", type=str, default=None,
                    help="Path to the daily EUR/CHF CSV from download_eurchf_yahoo.py. "
                         "Defaults to <inputs_path>/eurchf_yahoo_daily.csv")
parser.add_argument(
    "--inputs_path",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "data", "inputs"),
)
parser.add_argument(
    "--outputs_path",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "data", "outputs"),
)

if __name__ == "__main__":
    args = parser.parse_args()
    args.fix_start = str_2_bool(args.fix_start)
    args.intercept = str_2_bool(args.intercept)

    is_currency = args.target.upper() in CURRENCY_TARGETS

    # Resolve default data_name
    if args.data_name is None:
        args.data_name = "monetary_policy" if is_currency else "etfs_macro_large"

    target_label = args.target.upper() if is_currency else args.target

    if is_currency:
        eurchf_csv = args.eurchf_csv or os.path.join(
            args.inputs_path, "eurchf_yahoo_daily.csv"
        )
        print(f"Building EUR/CHF monthly dataset from {eurchf_csv} + monetary-policy-processed.csv...")
        monthly_data = build_dataset_currency(
            inputs_path=args.inputs_path,
            eurchf_csv=eurchf_csv,
            target_col=target_label,
        )
        print(f"Building EUR/CHF daily returns from {eurchf_csv}...")
        daily_returns = build_daily_data_currency(eurchf_csv, target_col=target_label)
    else:
        print("Building ETF monthly dataset from daily_etfs.csv + FRED-MD macro...")
        monthly_data = build_dataset(inputs_path=args.inputs_path, etf_target=args.target)
        print("Building ETF daily returns from daily_etfs.csv...")
        daily_returns = build_daily_data_etf(args.inputs_path)

    print(f"  Monthly dataset : {monthly_data.shape}, {monthly_data.index[0]} – {monthly_data.index[-1]}")
    print(f"  Daily returns   : {daily_returns.shape}, {daily_returns.index[0]} – {daily_returns.index[-1]}")

    print("Running CDTA daily forecast...")
    results = run_cdta_forecast_daily(
        monthly_data=monthly_data,
        daily_returns=daily_returns,
        target=target_label,
        estimation_window=args.estimation_window,
        daily_ols_window=args.daily_ols_window,
        p=args.p,
        beta_threshold=args.beta_threshold,
        fix_start=args.fix_start,
        intercept=args.intercept,
        k_agg=args.k_agg,
        max_features=args.max_features,
    )
    results["args"] = args

    out_dir = os.path.join(args.outputs_path, "cdta_nocluster", args.data_name)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(
        out_dir,
        f"{target_label}_{args.estimation_window}_{args.correl_window}_{args.p}.pickle",
    )
    save_pickle(path=out_file, obj=results)
    print(f"Saved results to: {out_file}")
    print(f"Predictions shape: {results['predictions'].shape}")
    print(results["predictions"].head())
