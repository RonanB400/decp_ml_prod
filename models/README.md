# DECP Clustering API

API for clustering public procurement contracts based on HDBSCAN.

## Setup

1. Make sure you have the trained models in the `models/` directory:
   - `hdbscan_clusterer.pkl` - The trained HDBSCAN model
   - `pca_model.pkl` - The trained PCA model
   - `preprocessing_pipeline.pkl` - The preprocessing pipeline
   - `cluster_profiles.pkl` - (Optional) Dataframe with cluster profiles

2. Install dependencies:
