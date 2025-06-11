# Pre-trained Models

This directory should contain the following pre-trained model files:

- `pca_model.pkl`: PCA dimensionality reduction model
- `hdbscan_model.pkl`: HDBSCAN clustering model
- `cluster_profiles.pkl`: DataFrame with cluster profiles

## Model Loading

These models are loaded at API startup and used for contract clustering predictions.
If you need to update these models, replace the files and restart the API service.

## Model Format

All models should be saved using pickle format with protocol 4 or higher.
