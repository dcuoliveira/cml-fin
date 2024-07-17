from fredapi import Fred
import pandas as pd
from tqdm import tqdm
import datetime
import numpy as np
import os

if __name__ == "__main__":

    # Initialize the FRED API with your API key
    fred = Fred(api_key='12d77a40907e43a92e9a295801db18d2')
    fred_raw = pd.read_csv('https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv')
    all_series = fred_raw.drop("sasdate", axis=1).columns

    transform = fred_raw.loc[0,].reset_index()
    transform.columns = transform.iloc[0,]
    transform = transform.drop(0)
    transform.columns = ["fred", "tcode"]

    fred_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "fredmd_transf.csv"), sep=",")
    fred_data["date"] = pd.to_datetime(fred_data["date"])
    fred_data.set_index("date", inplace=True)
    fred_data = fred_data.shift(+1).resample("M").last()

    # load forecast data and preprocess
    etfs_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "wrds_etf_returns.csv"))
    etfs_data = etfs_data[[col for col in etfs_data.columns if "t+1" not in col]]

    ## fix dates
    etfs_data["date"] = pd.to_datetime(etfs_data["date"])
    etfs_data = etfs_data.set_index("date")

    # compute first non-nan index
    returns_data = etfs_data.copy()
    first_non_nan = returns_data.apply(lambda x: x.first_valid_index())

    # subset etfs
    big_etfs_index = list(first_non_nan[first_non_nan <= "2000-02-02"].index)
    big_returns_data = returns_data[big_etfs_index].dropna()

    mid_etfs_index = list(first_non_nan[first_non_nan <= "2007-05-19"].index)
    mid_returns_data = returns_data[mid_etfs_index].dropna()

    small_etfs_index = list(first_non_nan[first_non_nan <= "2011-12-16"].index)
    small_returns_data = returns_data[small_etfs_index].dropna()

    # merge datasets
    big_data = pd.merge(big_returns_data, fred_data, left_index=True, right_index=True).fillna(0)
    mid_data = pd.merge(mid_returns_data, fred_data, left_index=True, right_index=True).fillna(0)
    small_data = pd.merge(small_returns_data, fred_data, left_index=True, right_index=True).fillna(0)

    # save datasets
    big_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_large.csv"))
    mid_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_mid.csv"))
    small_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_small.csv"))

