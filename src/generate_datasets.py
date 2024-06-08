import os
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # load forecast data and preprocess
    etfs_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "wrds_etf_returns.csv"))
    etfs_data = etfs_data[[col for col in etfs_data.columns if "t+1" not in col]]

    ## fix dates
    etfs_data["date"] = pd.to_datetime(etfs_data["date"])
    etfs_data["date"] = etfs_data["date"] + pd.DateOffset(months=1)
    etfs_data = etfs_data.set_index("date")

    ## resample and match memory data dates
    etfs_data = etfs_data.resample("MS").last().ffill()

    # load fred description
    fred_desc = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "utils", "fredmd_description.csv"), sep=";")

    # load
    fred_raw = pd.read_csv('https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv')
    transform = fred_raw.loc[0,].reset_index()
    transform.columns = transform.iloc[0,]
    transform = transform.drop(0)
    transform.columns = ["fred", "tcode"]
    fred_raw = fred_raw.drop(0)
    fred_raw.rename(columns={"sasdate": "date"}, inplace=True)
    fred_raw["date"] = pd.to_datetime(fred_raw["date"], format="%m/%d/%Y")
    fred_raw.set_index("date", inplace=True)

    # transform
    fred_data = []
    for col in fred_raw.columns:
        tcode = transform[transform["fred"] == col]["tcode"].iloc[0]
        ttype = fred_desc[fred_desc["tcode"] == tcode]["ttype"].iloc[0]

        if ttype == "First difference of natural log: ln(x)-ln(x-1)":
            tmp = np.log(fred_raw[col]).diff()
        elif ttype == "Level (i.e. no transformation): x(t)":
            tmp = fred_raw[col]
        elif ttype == "First difference: x(t)-x(t-1)":
            tmp = fred_raw[col].diff()
        elif ttype == "Natural log: ln(x)":
            tmp = np.log(fred_raw[col])
        elif ttype == "Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))":
            tmp = np.log(fred_raw[col]).diff(1) - np.log(fred_raw[col]).diff(2)
        elif ttype == "First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)":
            tmp = fred_raw[col].pct_change(1).diff() - fred_raw[col].pct_change(2).diff() 
        else:
            raise ValueError("Unknown transformation type")
        
        fred_data.append(pd.DataFrame(tmp, columns=[col], index=tmp.index))
    fred_data = pd.concat(fred_data, axis=1)

    # drop snp from fred
    fred_data.drop("S&P 500", axis=1, inplace=True)

    # # compute log returns
    # returns_data = np.log(etfs_data).diff(22)
    returns_data = etfs_data.copy()

    # compute first non-nan index
    first_non_nan = returns_data.apply(lambda x: x.first_valid_index())

    # subset etfs
    big_etfs_index = list(first_non_nan[first_non_nan <= "2000-02-02"].index)
    big_returns_data = returns_data[big_etfs_index].dropna()

    mid_etfs_index = list(first_non_nan[first_non_nan <= "2007-05-19"].index)
    mid_returns_data = returns_data[mid_etfs_index].dropna()

    small_etfs_index = list(first_non_nan[first_non_nan <= "2011-12-16"].index)
    small_returns_data = returns_data[small_etfs_index].dropna()

    # merge datasets
    big_data = pd.merge(big_returns_data, fred_data, left_index=True, right_index=True).resample("MS").last()
    mid_data = pd.merge(mid_returns_data, fred_data, left_index=True, right_index=True).resample("MS").last()
    small_data = pd.merge(small_returns_data, fred_data, left_index=True, right_index=True).resample("MS").last()

    # save datasets
    big_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_large.csv"))
    mid_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_mid.csv"))
    small_data.to_csv(os.path.join(os.path.dirname(__file__), "data", "inputs", "etfs_macro_small.csv"))

