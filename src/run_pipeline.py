import argparse
import pandas as pd
import os

from forecast.forecast_funcs import run_forecast
from metadata.etfs import etfs_large, etfs_small
from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser(description="Run forecast.")

parser.add_argument("--estimation_window", type=int, default=12 * 4)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--correl_window", type=int, default=100000) # all available data
parser.add_argument("--beta_threshold", type=float, default=0.4)
parser.add_argument("--pval_threshold", type=float, default=0.05)
parser.add_argument("--fix_start", type=str, default=True)
parser.add_argument("--incercept", type=str, default=True)
parser.add_argument("--fs_method", type=str, default="var-lingam")
parser.add_argument("--opt_k_method", type=str, default="eigen")
parser.add_argument("--clustering_method", type=str, default="rolling_kmeans")
parser.add_argument("--n_clusters", type=int, default=0)
parser.add_argument("--data_name", type=str, default="etfs_macro_large")
parser.add_argument("--inputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "inputs"))
parser.add_argument("--outputs_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "outputs"))

if __name__ == "__main__":

    args = parser.parse_args()

    args.fix_start = str_2_bool(args.fix_start)
    args.incercept = str_2_bool(args.incercept)

    data = pd.read_csv(os.path.join(args.inputs_path, f'{args.data_name}.csv'))
    
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
                               pval_threshold=args.pval_threshold,
                               incercept=args.incercept,
                               fs_method=args.fs_method,
                               opt_k_method=args.opt_k_method,
                               clustering_method=args.clustering_method,
                               n_clusters=args.n_clusters)

        results['args'] = args

        # add rolling cluster tag
        if args.clustering_method == "no":
            out_fs_method = f"{args.fs_method}_nocluster"
        elif args.clustering_method == "rolling_kmeans":
            out_fs_method = f"{args.fs_method}_rollingkmeans"
        elif args.clustering_method == "rolling_spectral":
            out_fs_method = f"{args.fs_method}_spectral"
        elif args.clustering_method == "kmeans":
            out_fs_method = f"{args.fs_method}_kmeans"
        elif args.clustering_method == "spectral":
            out_fs_method = f"{args.fs_method}_spectral"
        else:
            raise ValueError(f"Clustering method not recognized: {args.clustering_method}")

        # add number of clusters tag
        if args.clustering_method == "no":
            pass
        elif (args.n_clusters != 0) and (args.clustering_method != "no"):
            out_fs_method += f"_k{args.n_clusters}"
        elif (args.n_clusters == 0) and (args.clustering_method != "no"):
            out_fs_method += f"_kauto"
        else:
            raise ValueError(f"Clustering method not recognized: {args.clustering_method} and n_clusters: {args.n_clusters}")
        
        # add cv type tag
        if ((args.n_clusters == 0) and (args.clustering_method != "no")) and (args.opt_k_method == "cv"):
            out_fs_method += "_cv"
        elif ((args.n_clusters == 0) and (args.clustering_method != "no")) and (args.opt_k_method == "eigen"):
            out_fs_method += "_eigen"
        else:
            raise ValueError(f"Optimal k method type not recognized: {args.opt_k_method}")

        # check if results folder exists
        if not os.path.exists(os.path.join(args.outputs_path, out_fs_method, args.data_name)):
            os.makedirs(os.path.join(args.outputs_path, out_fs_method, args.data_name))
        
        # save results
        save_path = os.path.join(args.outputs_path, out_fs_method, args.data_name, "{}_{}_{}_{}.pickle".format(target,
                                                                                                                args.estimation_window,
                                                                                                                args.correl_window,
                                                                                                                args.p))
        save_pickle(path=save_path, obj=results)