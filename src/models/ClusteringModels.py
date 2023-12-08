from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np
from sklearn.metrics import silhouette_score


class ClusteringModels:
    def __init__(self) -> None:
        self.fred_des = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'utils', 'fredmd_description.csv'), sep=';')
        

    def kmeans(self, input: pd.DataFrame, n_clusters: int=20):

        # compute forward looking cluster of the correlation matrix
        clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(input)

        return clusters
    
    def spectral(self, input: pd.DataFrame, n_clusters: int=20, signed: bool = False):
        
        if not signed:
            input += 1    
        # compute forward looking cluster of the correlation matrix
        clusters = spectral_mathod(n_clusters = n_clusters).fit((input).to_numpy())

        return clusters
    
    def add_cluster_description(self, clusters):
        
        labelled_clusters = pd.DataFrame({"fred": self.feature_names, "cluster": clusters.labels_})
        labelled_clusters.sort_values(by="cluster")
        labelled_clusters = pd.merge(labelled_clusters, self.fred_des[["fred", "description"]], on='fred')
        
        return labelled_clusters

    def compute_clusters(self, 
                         data: pd.DataFrame, 
                         target: str, 
                         clustering_method: str, 
                         n_clusters: int = 20, 
                         method: str = "spectral",
                         threshold: float = 0.8):

        input = data.drop([target], axis=1).corr()
        self.feature_names = list(input.columns)
        
        if not n_clusters:
            if method == "spectral":
                n_clusters = threshold_variance_explained(input, threshold)
            else:
                raise ValueError("n_clusters method not supported")
        
        if clustering_method == "kmeans":
            clusters = self.kmeans(input=input,  n_clusters = n_clusters)
        elif clustering_method == "spectral":
            clusters = self.spectral(input = input, n_clusters = n_clusters)
        elif clustering_method == "signed_spectral":
            clusters = self.spectral(input = input, n_clusters = n_clusters, signed = True)
        else:
            raise ValueError("clustering_method not supported")
        
        return clusters
    
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


def threshold_variance_explained(A, threshold):
    assert 0 <= threshold <= 1
    ratio_expalined, _ = np.linalg.eigh(A)
    ratio_expalined = ratio_expalined / ratio_expalined.sum()
    cum_ratio_expalined = ratio_expalined[::-1].cumsum()
    n_clusters = np.argmax(cum_ratio_expalined >= threshold) + 1
    return n_clusters



class spectral_mathod:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        
    def fit(self, A):
        self.labels_, _ = spectral_clustering(A, self.n_clusters, symmetrize = False)
        return self
        

def A_Hermitian(A):
    '''
    Given an adjacency matrix, compute its Hermitian adjacency.
    H_uv = (A_uv - A_vu) i
    '''
    A = (A - A.T) * 1j
    return A
   

def symmetrize_matrix(M, method):
    '''
    Symmetrize given matrix
    '''
    if method == 'M+MT' or method == 'MT+M':
        M = M + M.T
    elif method == 'MMT':
        M = np.matmul(M, M.T)
    elif method == 'MTM':
        M = np.matmul(M.T, M)
    elif method == 'MTM+MMT' or method == 'MMT+MTM':
        M = np.matmul(M, M.T) + np.matmul(M.T, M)
    return M


def Laplacian(A, symmetrize, normalize, remove_diag):
    '''
    Compute Laplacian of the given matrix A.
    A is an (weighted) adjacency matrix of an unsigned network, that is Aij >= 0 for all i,j

    Parameters
    ----------
    A : np.array
        N x N matrix.
    symmetric : str, optional
        If sys/symmetric, return D^(-1/2) L D^(-1/2), If simple, return D^(-1) L. 
        If None/False return L without normalization

    Returns
    -------
    L : np.array
        Normalized matrix.
    '''
    assert A.shape[0] == A.shape[1]
    assert (A >= 0).all().all()
    
    if symmetrize:
        A = symmetrize_matrix(A, symmetrize)
        
    if remove_diag:
        np.fill_diagonal(A, 0)
    
    D = np.abs(A).sum(axis = 1)
    L = np.diag(D) - A 
    
    if normalize:
        if normalize.startswith('sym'):
            D = np.sqrt(D)
            L = (L.T / D).T / D
        elif normalize == 'rw':
            L = (L.T / D).T
        else:
            raise NotImplementedError
                
    return L


def spectral_clustering(A, n_clusters, n_eig = None, symmetrize = 'M+MT', normalize = 'rw', init = 'k-means++', random_state = None, n_init = 500):
    '''
    Spectral Clustering on weighted, undirected and unsigned graph A

    Parameters
    ----------
    A : np.array
        (Weighted) Adjacency matrix of A.
    n_clusters : int
        Number of output clusters.
    n_eig : int, optional
        Number of top eigenvalues to choose. The default is 10.

    Returns
    -------
    clusters : np.array
        Labels of clusters.

    '''
    
    ''' 1. Compute Laplacian matrix '''
    L = Laplacian(A, symmetrize, normalize, True)

    ''' 2. Spectural deconposition of L, get to n eigenvalues and coreesponding eigenvectors '''
    eig_values, eig_matrix = np.linalg.eigh(L)

    
    #assert (eig_values >= - 1e-10).all()  
    if not n_eig:
        n_eig = n_clusters
    
        
    # Small eigenvalues
    eig_values, eig_matrix = eig_values[ : n_eig], eig_matrix[:, : n_eig]
    
    
    eig_matrix = (eig_matrix.T / np.sqrt((eig_matrix ** 2).sum(axis = 1))).T
    
    ''' 3. Clustering on rows of eigenvectos with K means '''
    if isinstance(init, np.ndarray):
        n_clusters = len(set(init))
        n_init = 1
        print(pd.DataFrame(eig_matrix))
        init = pd.DataFrame(eig_matrix).groupby(init).mean()

    
    kmeans = KMeans(n_clusters = n_clusters, init = init, random_state = random_state, n_init = n_init)
    clusters = kmeans.fit(eig_matrix).labels_
    
    return clusters, (eig_values, eig_matrix)






