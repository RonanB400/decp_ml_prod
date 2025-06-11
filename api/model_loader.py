import pickle
import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_models(model_dir="models"):
    """
    Load models from disk
    """
    try:
        # Additional debugging
        print(f"Loading models from {model_dir}...")
        models_path = Path(model_dir)

        # Load preprocessing pipeline first
        try:
            print(f"Attempting to load pipeline from {models_path / 'pipeline_marche_sim.pkl'}")
            with open(models_path / "pipeline_marche_sim.pkl", "rb") as f:
                pipeline = pickle.load(f)
                print("Pipeline type:", type(pipeline))
                print("Pipeline is fitted:", hasattr(pipeline, "transformers_"))
                print("Preprocessing pipeline loaded successfully")
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading preprocessing pipeline: {e}")
            pipeline = None

        # Load PCA model
        try:
            with open(models_path / "pca_model.pkl", "rb") as f:
                pca_model = pickle.load(f)
                print("PCA model loaded successfully")
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading PCA model: {e}")
            pca_model = None

        # Load HDBSCAN model
        try:
            with open(models_path / "hdbscan_clusterer.pkl", "rb") as f:
                hdbscan_model = pickle.load(f)
                # Check if prediction data exists (should be there from saved model)
                if hasattr(hdbscan_model, 'prediction_data_') and hdbscan_model.prediction_data_ is not None:
                    print("HDBSCAN model loaded with existing prediction data")
                else:
                    # Only generate if somehow missing
                    print("Warning: Prediction data missing, generating now...")
                    hdbscan_model.generate_prediction_data()
                print("HDBSCAN model loaded successfully")
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading HDBSCAN model: {e}")
            hdbscan_model = None

        # Load or create cluster profiles
        try:
            with open(models_path / "cluster_profiles.pkl", "rb") as f:
                cluster_profiles = pickle.load(f)
                print("Cluster profiles loaded successfully")
        except (FileNotFoundError, pickle.PickleError):
            # Create empty profiles dataframe as fallback
            print("Cluster profiles not found, creating empty DataFrame")
            cluster_profiles = pd.DataFrame(columns=[
                'cluster_id', 'size', 'pct_total', 'top_cpv', 'top_cpv_pct',
                'mean_amount', 'median_amount', 'mean_duration', 'median_duration'
            ])

        return pca_model, hdbscan_model, pipeline, cluster_profiles
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None
