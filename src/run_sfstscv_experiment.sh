python run_pipeline.py --fs_method sfstscv --clustering_method rolling_kmeans --opt_k_method eigen --n_clusters 0 --intra_cluster_selection rank
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_spectral --opt_k_method eigen --n_clusters 0 --intra_cluster_selection rank
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_kmeans --opt_k_method no --n_clusters 5 --intra_cluster_selection rank
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_kmeans --opt_k_method no --n_clusters 10 --intra_cluster_selection rank
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_spectral --opt_k_method no --n_clusters 5 --intra_cluster_selection rank
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_spectral --opt_k_method no --n_clusters 10 --intra_cluster_selection rank
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_kmeans --opt_k_method no --n_clusters 5 --intra_cluster_selection pca
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_kmeans --opt_k_method no --n_clusters 10 --intra_cluster_selection pca
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_spectral --opt_k_method no --n_clusters 5 --intra_cluster_selection pca
python run_pipeline.py --fs_method sfstscv --clustering_method rolling_spectral --opt_k_method no --n_clusters 10 --intra_cluster_selection pca