import os
import pandas as pd
import numpy as np
from FredMD import FredMD
from metadata.etfs import etfs_large, etfs_small

DATA_UTILS_PATH = os.path.join(os.path.dirname(__file__), "data", "utils")
INPUTS_PATH = os.path.join(os.path.dirname(__file__), "data", "inputs")
START_DATE = "1960-01-01"

def gen_fred_dataset(start_date):
    fredmd = FredMD()
    fredmd.apply_transforms()

    # load fredmd data from URL
    raw_fredmd_df = fredmd.rawseries
    transf_fredmd_df = fredmd.series

    # descriptions
    des_raw_fredmd_df = pd.DataFrame(raw_fredmd_df.iloc[0]).reset_index()
    des_raw_fredmd_df.columns = ["fred", "ttype"]
    des_raw_fredmd_df = des_raw_fredmd_df.drop([0], axis=0)
    des_fredmd_df = pd.read_csv(os.path.join(DATA_UTILS_PATH, "fredmd_description.csv"), delimiter=";")

    # # select variables with description
    # raw_fredmd_df = raw_fredmd_df[list(set(list(des_fredmd_df["fred"])) & set(list(raw_fredmd_df.columns)))]

    # # select price data with logdiff transf
    # des_prices = des_fredmd_df.loc[(des_fredmd_df["group"] == "Prices")&(des_fredmd_df["tcode"] == 6)]
    # prices_var_names = des_prices["fred"]
    # fredmd_prices_df = raw_fredmd_df[list(prices_var_names)]
    # change_fredmd_prices_df = np.log(fredmd_prices_df).diff()

    # # add log diff prices to original data
    # selected_transf_fredmd_df = transf_fredmd_df.drop(list(prices_var_names), axis=1)
    # target_df = pd.concat([selected_transf_fredmd_df, change_fredmd_prices_df], axis=1)

    # delete rows with NaN in all columns
    transf_fredmd_df = transf_fredmd_df.dropna(how="all")
    raw_fredmd_df = raw_fredmd_df.dropna(how="all")

    # fix index name
    transf_fredmd_df.index.name = "date"
    raw_fredmd_df.index.name = "date"

    #
    dict_df = des_fredmd_df[['fred', 'group']]
    dict_df = dict_df.loc[dict_df['group'] != 'Stock Market']
    names_dict = list(dict_df['fred'])
    transf_fredmd_df = transf_fredmd_df[list(transf_fredmd_df.columns[transf_fredmd_df.columns.isin(names_dict)])]
    raw_fredmd_df = raw_fredmd_df[list(raw_fredmd_df.columns[raw_fredmd_df.columns.isin(names_dict)])]

    # export
    transf_fredmd_df.loc[start_date:].to_csv(os.path.join(INPUTS_PATH,  "fredmd_transf_df.csv"))
    raw_fredmd_df.loc[start_date:].to_csv(os.path.join(INPUTS_PATH,  "fredmd_raw_df.csv"))

def merge_fred_etfs():
    # read datasets
    transf_fredmd = pd.read_csv(os.path.join(INPUTS_PATH,  "fredmd_transf_df.csv"))
    etfs_returns = pd.read_csv(os.path.join(INPUTS_PATH,  "wrds_etf_returns.csv"))

    # fix fred dates
    transf_fredmd['date'] = pd.to_datetime(transf_fredmd['date'])
    transf_fredmd.set_index('date', inplace=True)
    transf_fredmd = transf_fredmd.astype(float)

    # select subset of etfs
    etfs_returns = etfs_returns[[col for col in etfs_returns.columns if "t+1" not in col]]
    etfs_returns = etfs_returns[['date'] + etfs_large]

    # fix etfs dates
    etfs_returns["date"] = pd.to_datetime(etfs_returns["date"])
    etfs_returns["date"] = etfs_returns["date"] + pd.DateOffset(months=1)
    etfs_returns = etfs_returns.set_index("date")
    etfs_returns = etfs_returns.resample("MS").last().ffill()

    # fill missing values
    transf_fredmd = transf_fredmd.interpolate(method='linear', limit_direction='forward', axis=0)
    transf_fredmd = transf_fredmd.fillna(method='ffill')
    transf_fredmd = transf_fredmd.fillna(method='bfill')

    # merge etfs with fred data
    final_df = pd.merge(etfs_returns, transf_fredmd, left_index=True, right_index=True)

    # Group by year and count the number of months (rows) in each year
    final_df['year'] = final_df.index.year
    months_per_year = final_df.groupby('year').size()

    # Check if all years have 12 months
    all_years_have_12_months = months_per_year.eq(12).all()

    if all_years_have_12_months:
        print("Each year has 12 months.")
    else:
        print("Some years do not have 12 months.")
        # Display the years that do not have 12 months
        print(months_per_year[months_per_year != 12])

    # drop year column
    final_df.drop(['year'], axis=1, inplace=True)

    # export
    final_df.to_csv(os.path.join(INPUTS_PATH,  "etfs_macro_large.csv"))

    # shift forward
    final_df = final_df.shift(+1).dropna()
    final_df.to_csv(os.path.join(INPUTS_PATH,  "etfs_macro_large_lagged.csv"))

if __name__ == "__main__":
    gen_fred_dataset(start_date=START_DATE)
    merge_fred_etfs()
  



