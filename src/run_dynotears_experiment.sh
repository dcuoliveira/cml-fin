# python run_pipeline.py --fs_method dynotears --clustering_method no --n_clusters 0
# python run_pipeline.py --fs_method dynotears --clustering_method kmeans --n_clusters 0
# python run_pipeline.py --fs_method dynotears --clustering_method kmeans --n_clusters 5
# python run_pipeline.py --fs_method dynotears --clustering_method kmeans --n_clusters 10
python run_pipeline.py --fs_method dynotears --clustering_method rolling_kmeans --n_clusters 0
python run_pipeline.py --fs_method dynotears --clustering_method rolling_kmeans --n_clusters 5
python run_pipeline.py --fs_method dynotears --clustering_method rolling_kmeans --n_clusters 10