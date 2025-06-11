# api/fast.py
import hdbscan
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import the model loader
from api.model_loader import load_models

# Import the find_similar_clusters function
from api.prediction import find_similar_clusters

# Initialize FastAPI
app = FastAPI(
    title="DECP Clustering API",
    description="API Lanterne publique : etudes des marchés publics",
    version="1.0.0"
)

# Load models at startup
pca_model, hdbscan_model, preprocessing_pipeline, cluster_profiles = load_models()

# Input data model
class Contract(BaseModel):
    """Model for contract data input"""
    montant: float
    dureeMois: int
    offresRecues: int
    procedure: str
    nature: str
    formePrix: str
    ccag: Optional[str] = "Pas de CCAG"
    sousTraitanceDeclaree: Optional[float] = 0.0
    origineFrance: Optional[float] = 0.0
    marcheInnovant: Optional[float] = 0.0
    idAccordCadre: Optional[str] = None
    typeGroupementOperateurs: Optional[str] = None
    tauxAvance: Optional[float] = 0.0
    codeCPV_2_3: Optional[int] = None

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"status": "ok", "message": "DECP Clustering API is running"}

@app.get("/api/clusters")
def get_clusters():
    """Get all cluster profiles"""
    if cluster_profiles is None or len(cluster_profiles) == 0:
        raise HTTPException(status_code=404, detail="No cluster profiles available")

    return {
        "num_clusters": len(cluster_profiles),
        "clusters": cluster_profiles.to_dict(orient="records")
    }

@app.post("/api/predict")
def predict_cluster(contract: Contract):
    """
    Predict the cluster for a new contract

    Returns cluster ID, probability, and similar clusters
    """
    # Check if models are loaded
    if pca_model is None or hdbscan_model is None or preprocessing_pipeline is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Convert contract to DataFrame for preprocessing
    contract_df = pd.DataFrame([contract.dict()])

    # Add tauxAvance column from tauxAvance_cat if needed
    # (e.g., convert "5%" to 5.0, "0%" to 0.0, etc.)
    if 'tauxAvance_cat' in contract_df.columns and 'tauxAvance' not in contract_df.columns:
        # Map standard values or use a default of 0
        tax_map = {"0%": 0.0, "5%": 5.0, "10%": 10.0, "15%": 15.0, "20%": 20.0, "25%": 25.0, "30%": 30.0}

        def map_taux(val):
            if pd.isna(val):
                return 0.0
            return tax_map.get(val, 0.0)

        contract_df['tauxAvance'] = contract_df['tauxAvance_cat'].apply(map_taux)

    try:
        # Preprocess the input
        contract_preprocessed = preprocessing_pipeline.transform(contract_df)

        # Apply PCA
        contract_pca = pca_model.transform(contract_preprocessed)

        # Predict cluster
        cluster_id, probability = hdbscan.approximate_predict(hdbscan_model, contract_pca)
        predicted_cluster = int(cluster_id[0])
        prob = float(probability[0])

        # Handle noise assignment
        if predicted_cluster == -1:
            # Find nearest cluster
            distances = [np.min(np.linalg.norm(contract_pca - hdbscan_model.exemplars_[c], axis=1))
                        for c in range(len(hdbscan_model.exemplars_))]
            nearest_cluster = int(np.argmin(distances))

            response = {
                "cluster_id": -1,
                "nearest_cluster": nearest_cluster,
                "probability": 0.0,
                "is_noise": True
            }
            similar_clusters = []
        else:
            # Get cluster profile
            cluster_profile = cluster_profiles[cluster_profiles['cluster_id'] == predicted_cluster]

            # Find similar clusters based on contract data
            similar_clusters = find_similar_clusters(
                contract_data=contract.dict(),
                profiles_df=cluster_profiles[cluster_profiles['cluster_id'] != predicted_cluster],
                top_n=5  # Adjust number as needed
            )

            response = {
                "cluster_id": predicted_cluster,
                "probability": prob,
                "is_noise": False,
                "cluster_profile": cluster_profile.to_dict(orient="records")[0] if len(cluster_profile) > 0 else {},
                "similar_clusters": similar_clusters
            }

        return response

    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())  # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
