# api/fast.py
import os
import pickle
import hdbscan
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import from local modules
from api.preprocessing import preprocess_input
from api.prediction import find_similar_clusters
from decp.params import PCA_PATH, HDBSCAN_PATH, PROFILES_PATH

# Initialize FastAPI
app = FastAPI(
    title="Procurement Clustering API",
    description="API for clustering public procurement contracts",
    version="1.0.0"
)

# Global variables to store loaded models
pca_model = None
hdbscan_model = None
cluster_profiles = None

# Input data model
class ContractData(BaseModel):
    """Contract data for prediction"""
    montant: float
    dureeMois: float
    offresRecues: Optional[int] = 1
    procedure: str
    nature: str
    formePrix: Optional[str] = "FORFAIT"
    codeCPV: Optional[str] = None

# Response data model
class ClusterResponse(BaseModel):
    """Response with cluster prediction details"""
    cluster_id: int
    probability: float
    similar_clusters: List[Dict[str, Any]]
    cluster_profile: Dict[str, Any]

def get_models():
    """Dependency to ensure models are loaded"""
    if pca_model is None or hdbscan_model is None or cluster_profiles is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again later."
        )
    return pca_model, hdbscan_model, cluster_profiles

@app.on_event("startup")
async def load_models():
    """Load pre-trained models on startup"""
    global pca_model, hdbscan_model, cluster_profiles

    try:
        print(f"Loading PCA model from {PCA_PATH}")
        with open(PCA_PATH, "rb") as f:
            pca_model = pickle.load(f)

        print(f"Loading HDBSCAN model from {HDBSCAN_PATH}")
        with open(HDBSCAN_PATH, "rb") as f:
            hdbscan_model = pickle.load(f)

        print(f"Loading cluster profiles from {PROFILES_PATH}")
        with open(PROFILES_PATH, "rb") as f:
            cluster_profiles = pickle.load(f)

        print("All models loaded successfully")

    except Exception as e:
        print(f"Error loading models: {e}")
        # Allow startup without models, will return error during requests

@app.get("/")
def read_root():
    """Root endpoint to check if API is running"""
    return {
        "status": "ok",
        "message": "Procurement Clustering API is running",
        "models_loaded": {
            "pca": pca_model is not None,
            "hdbscan": hdbscan_model is not None,
            "profiles": cluster_profiles is not None
        }
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    if pca_model is None or hdbscan_model is None or cluster_profiles is None:
        return {"status": "error", "message": "Models not loaded"}
    return {"status": "ok", "models_loaded": True}

@app.get("/api/clusters")
def get_clusters(models=Depends(get_models)):
    """Get available clusters and their profiles"""
    _, _, profiles = models

    return {
        "num_clusters": len(profiles),
        "clusters": profiles.to_dict(orient="records")
    }

@app.post("/api/predict", response_model=ClusterResponse)
def predict_cluster(contract: ContractData, models=Depends(get_models)):
    """Predict cluster for new contract data"""
    pca, hdbscan_model, profiles = models

    try:
        # Process input data
        features = preprocess_input(contract)

        # Transform with PCA
        pca_features = pca.transform(features.reshape(1, -1))

        # Predict with HDBSCAN
        cluster_labels, probabilities = hdbscan.approximate_predict(hdbscan_model, pca_features)
        cluster_id = int(cluster_labels[0])
        probability = float(probabilities[0])

        # If assigned to noise, find nearest cluster
        if cluster_id == -1:
            distances = [np.min(np.linalg.norm(pca_features - hdbscan_model.exemplars_[c], axis=1))
                       for c in range(len(hdbscan_model.exemplars_))]
            cluster_id = int(np.argmin(distances))
            probability = 0.0

        # Get cluster profile
        profile = profiles[profiles['cluster_id'] == cluster_id].to_dict(orient="records")

        # Find similar clusters
        similar = find_similar_clusters(contract.dict(), profiles, top_n=3)

        return {
            "cluster_id": cluster_id,
            "probability": probability,
            "similar_clusters": similar,
            "cluster_profile": profile[0] if profile else {}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.fast:app", host="0.0.0.0", port=8000, reload=True)
