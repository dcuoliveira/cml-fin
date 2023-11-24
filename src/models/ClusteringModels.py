from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np

class ClusteringModels:
    def __init__(self) -> None:
        self.fred_des = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'utils', 'fredmd_description.csv'), sep=';')

    def kmeans(self, input: pd.DataFrame, n_clusters: int=20):

        # compute forward looking cluster of the correlation matrix
        clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(input)

        return clusters
    
    def add_cluster_description(self, clusters):
        
        labelled_clusters = pd.DataFrame({"fred": self.feature_names, "cluster": clusters.labels_})
        labelled_clusters.sort_values(by="cluster")
        labelled_clusters = pd.merge(labelled_clusters, self.fred_des[["fred", "description"]], on='fred')
        
        return labelled_clusters

    def compute_clusters(self, data: pd.DataFrame, target: str, clustering_method: str):

        input = data.drop([target], axis=1).corr()
        self.feature_names = list(input.columns)

        if clustering_method == "kmeans":
            clusters = self.kmeans(input=input)
            return clusters
        else:
            raise ValueError("clustering_method not supported")
    
    def compute_within_cluster_corr_rank(self,
                                         data: pd.DataFrame,
                                         target: str,
                                         labelled_clusters: pd.DataFrame,
                                         correl_window: int):

        correl_dict = {}
        rank_list = []
        for c in labelled_clusters['cluster'].unique():
            clustes_variables = labelled_clusters.loc[labelled_clusters['cluster'] == c]['fred'].values

            clusters_features_df = data[[target] + list(clustes_variables)]
            
            # compute rolling correlation
            rolling_corr_df = clusters_features_df.rolling(window=correl_window, min_periods=12).corr()

            # compute correlation with the target
            rolling_corr_df = rolling_corr_df[[target]].reset_index()
            rolling_corr_df = rolling_corr_df.loc[rolling_corr_df["level_1"] != target]
            rolling_corr_df = rolling_corr_df.pivot_table(index=["date"], columns=["level_1"])
            rolling_corr_df.columns = rolling_corr_df.columns.droplevel()

            # save correl
            correl_dict[c] = rolling_corr_df

            # compute rankings given correl
            rank_df = rolling_corr_df.rank(axis=1, ascending=False)

            # save rank
            rank_list.append(rank_df)

        final_rank_df = pd.concat(rank_list, axis=1)

        return final_rank_df
