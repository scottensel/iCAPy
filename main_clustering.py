# main_clustering.py
from input_scripts.Inputs_Clustering_data import setup_clustering_data_params
from input_scripts.Inputs_Clustering import setup_clustering_params
from functions.Run_Clustering import run_clustering  # you will implement this

def main():
    # Data-related params (path, subjects, TR, etc.)
    param = setup_clustering_data_params()

    # Clustering-specific params
    param.update(setup_clustering_params())

    # Run clustering (Aggregation + k-means + consensus, etc.)
    run_clustering(param)


if __name__ == "__main__":
    main()
