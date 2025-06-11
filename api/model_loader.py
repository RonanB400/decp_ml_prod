import pickle
import pandas as pd
from pathlib import Path
from keras.models import load_model


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
            pipeline_path = models_path / 'pipeline_marche_sim.pkl'
            print(f"Attempting to load pipeline from {pipeline_path}")
            with open(pipeline_path, "rb") as f:
                cluster_pipeline = pickle.load(f)
                print("Pipeline type:", type(cluster_pipeline))
                fitted_check = hasattr(cluster_pipeline, "transformers_")
                print("Pipeline is fitted:", fitted_check)
                print("Preprocessing pipeline loaded successfully")
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading preprocessing pipeline: {e}")
            cluster_pipeline = None

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
                # Check if prediction data exists
                has_prediction_data = (
                    hasattr(hdbscan_model, 'prediction_data_') 
                    and hdbscan_model.prediction_data_ is not None
                )
                if has_prediction_data:
                    msg = "HDBSCAN model loaded with existing prediction data"
                    print(msg)
                else:
                    # Only generate if somehow missing
                    warning_msg = "Warning: Prediction data missing, " \
                                "generating now..."
                    print(warning_msg)
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
                'mean_amount', 'median_amount', 'mean_duration', 
                'median_duration'
            ])

        # Load amount prediction pipeline
        try:
            with open(models_path / "pipeline_pred_montant.pkl", "rb") as f:
                amount_pipeline = pickle.load(f)
                print("Amount prediction pipeline loaded successfully")
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"Error loading amount prediction pipeline: {e}")
            amount_pipeline = None

        # Load amount prediction model
        try:
            amount_model = load_model(models_path / "model_montant_100.keras")
            print("Amount prediction model loaded successfully")
        except (FileNotFoundError, Exception) as e:
            print(f"Error loading amount prediction model: {e}")
            amount_model = None

        return (pca_model, hdbscan_model, cluster_pipeline, cluster_profiles, 
                amount_pipeline, amount_model)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None, None, None
