import os
from glob import glob
import pandas as pd

from utils.conn_data import load_pickle, save_pickle

outputs_path = os.path.join(os.path.dirname(__file__), 'data', 'outputs')

# list dirs onl in outputs
output_dirs = glob(os.path.join(outputs_path, '*'))
for outputs_dir in output_dirs:

    if ('.pickle' in outputs_dir.split("/")[-1]) or ('results' in outputs_dir.split("/")[-1]):
        continue

    # list files in output dir
    output_files = os.listdir(os.path.join(outputs_dir, 'etfs_macro_large'))
    for output_file in output_files:

        # read file
        output = load_pickle(os.path.join(outputs_dir, 'etfs_macro_large', output_file))

        # relevant data
        preds = output['predictions']

        # add momentum returns
        preds["cumret12m"] = preds["true"].rolling(12).sum().shift(+1)

        # substitute zero value predictions with sign of cumret12m
        if len(preds.loc[preds["prediction"] == 0]) > 0:
            preds.loc[preds["prediction"] == 0, "prediction"] = preds["cumret12m"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            preds.loc[preds["prediction"] == 0, "prediction_zscore"] = preds["cumret12m"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

            # save file
            del preds['cumret12m']
            output['predictions'] = preds
            save_pickle(output, os.path.join(outputs_dir, 'etfs_macro_large', output_file))
