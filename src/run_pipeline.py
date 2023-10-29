import argparse
import pandas as pd
import os

from forecast.forecast_funcs import run_forecast
from metadata.etfs import etfs_large, etfs_small

parser = argparse.ArgumentParser(description="Run forecast.")
parser.add_argument("--estimation_window", type=int, default=12 * 8)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--correl_window", type=int, default=1000)
parser.add_argument("--beta_threshold", type=float, default=0)
parser.add_argument("--fix_start", type=bool, default=True)
parser.add_argument("--incercept", type=bool, default=True)
parser.add_argument("--fs_method", type=str, default="lasso")
parser.add_argument("--cv_type", type=str, default="cv")
parser.add_argument("--inputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "inputs"))

if __name__ == "__main__":

    args = parser.parse_args()

    data = pd.read_csv(os.path.join(args.inputs_path, 'etfs_macro_large.csv'))
    
    # fix dates
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")

    etfs_large = etfs_large.copy()

    for target in etfs_large:

        # select etfs to remove
        removed_etfs = [etf for etf in etfs_large if etf != target]

        # delete etfs
        selected_data = data.drop(removed_etfs, axis=1)

        results = run_forecast(data=selected_data,
                               target=target,
                               fix_start=args.fix_start,
                               estimation_window=args.estimation_window,
                               correl_window=args.correl_window,
                               p=args.p,
                               beta_threshold=args.beta_threshold,
                               incercept=args.incercept,
                               fs_method=args.fs_method,
                               cv_type=args.cv_type)