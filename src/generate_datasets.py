import os
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # load monthly macroeconomic data, after transformation
    fred_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "fredmd_transf.csv"))

    # drop snp from fred
    fred_data.drop("S&P 500", axis=1, inplace=True)

    # load full daily etfs data
    etfs_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs.csv"))

    # resample to business day frequency and forward fill
    fred_data["date"] = pd.to_datetime(fred_data["date"])
    fred_data.set_index("date", inplace=True)
    fred_data = fred_data.resample("B").last().ffill()

    etfs_data["date"] = pd.to_datetime(etfs_data["date"])
    etfs_data.set_index("date", inplace=True)
    etfs_data = etfs_data.resample("B").last().ffill()

    # compute log returns
    returns_data = np.log(etfs_data).diff()

    # compute first non-nan index
    first_non_nan = returns_data.apply(lambda x: x.first_valid_index())

    # subset etfs
    big_etfs_index = list(first_non_nan[first_non_nan <= "2000-01-04"].index)
    big_returns_data = returns_data[big_etfs_index].dropna()

    mid_etfs_index = list(first_non_nan[first_non_nan <= "2007-04-19"].index)
    mid_returns_data = returns_data[mid_etfs_index].dropna()

    small_etfs_index = list(first_non_nan[first_non_nan <= "2011-11-16"].index)
    small_returns_data = returns_data[small_etfs_index].dropna()

    # merge datasets
    big_data = pd.merge(big_returns_data, fred_data, left_index=True, right_index=True).resample("M").last()
    mid_data = pd.merge(mid_returns_data, fred_data, left_index=True, right_index=True).resample("M").last()
    small_data = pd.merge(small_returns_data, fred_data, left_index=True, right_index=True).resample("M").last()

    # save datasets
    big_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_large.csv"))
    mid_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_mid.csv"))
    small_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_small.csv"))

