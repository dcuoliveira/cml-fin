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

    fred_desc = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "utils", "fredmd_description.csv"), sep=";")

    fred_series = []
    error_series = []

    # Define today's date for ALFRED API queries
    today = datetime.date.today().strftime('%Y-%m-%d')
    for series_id in tqdm(all_series, desc="Fetching FRED data", total=len(all_series)):
        try:
            # Try fetching as usual (from FRED)
            all_releases = fred.get_series_all_releases(series_id)
        except Exception as e:
            print(f"Error fetching {series_id} from ALFRED: {e}")
            error_series.append(series_id)
            continue

        # Process and store data
        all_releases_df = pd.DataFrame(all_releases)
        all_releases_df.rename(columns={'realtime_start': 'actual_release_date', 'date': 'date', 'value': 'value'}, inplace=True)
        fred_raw = all_releases_df[['actual_release_date', 'value']]
        fred_raw.rename(columns={'actual_release_date': 'date', 'value': series_id}, inplace=True)
        fred_raw["date"] = pd.to_datetime(fred_raw["date"])
        fred_raw.set_index("date", inplace=True)
        fred_raw[series_id] = pd.to_numeric(fred_raw[series_id], errors='coerce')

        tcode = transform[transform["fred"] == series_id]["tcode"].iloc[0]
        ttype = fred_desc[fred_desc["tcode"] == tcode]["ttype"].iloc[0]

        if ttype == "First difference of natural log: ln(x)-ln(x-1)":
            tmp = np.log(fred_raw[series_id]).diff()
        elif ttype == "Level (i.e. no transformation): x(t)":
            tmp = fred_raw[series_id]
        elif ttype == "First difference: x(t)-x(t-1)":
            tmp = fred_raw[series_id].diff()
        elif ttype == "Natural log: ln(x)":
            tmp = np.log(fred_raw[series_id])
        elif ttype == "Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))":
            tmp = np.log(fred_raw[series_id]).diff(1) - np.log(fred_raw[series_id]).diff(2)
        elif ttype == "First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)":
            tmp = fred_raw[series_id].pct_change(1).diff() - fred_raw[series_id].pct_change(2).diff() 
        else:
            raise ValueError("Unknown transformation type")

        fred_series.append(tmp.resample("B").last())

    fred_data = pd.concat(fred_series, axis=1).ffill()

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
    big_data = pd.merge(big_returns_data, fred_data, left_index=True, right_index=True)
    mid_data = pd.merge(mid_returns_data, fred_data, left_index=True, right_index=True)
    small_data = pd.merge(small_returns_data, fred_data, left_index=True, right_index=True)


    # save datasets
    big_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_large.csv"))
    mid_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_mid.csv"))
    small_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_small.csv"))

