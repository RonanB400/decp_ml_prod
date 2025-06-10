# api/fast.py
import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from decp.params import *

# Initialize FastAPI
app = FastAPI(
    title="Procurement Clustering API",
    description="API for clustering public procurement contracts",
    version="1.0.0"
)

# Paths to model files
#MODELS_DIR = "models"
#PCA_PATH = os.path.join(MODELS_DIR, "pca_reducer.pkl")
#HDBSCAN_PATH = os.path.join(MODELS_DIR, "hdbscan_clusterer.pkl")
#PROFILES_PATH = os.path.join(MODELS_DIR, "cluster_profiles.pkl")

# Global variables to store loaded models
pca_model = None
hdbscan_model = None
cluster_profiles = None

# Input data model
class ContractData(BaseModel):
    montant: float
    dureeMois: float
    offresRecues: Optional[int] = 1
    procedure: str
    nature: str
    formePrix: Optional[str] = "FORFAIT"
    codeCPV: Optional[str] = None

# Response data model
class ClusterResponse(BaseModel):
    cluster_id: int
    probability: float
    similar_clusters: List[Dict[str, Any]]
    cluster_profile: Dict[str, Any]

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global pca_model, hdbscan_model, cluster_profiles

    try:
        with open(PCA_PATH, "rb") as f:
            pca_model = pickle.load(f)

        with open(HDBSCAN_PATH, "rb") as f:
            hdbscan_model = pickle.load(f)

        with open(PROFILES_PATH, "rb") as f:
            cluster_profiles = pickle.load(f)

    except Exception as e:
        print(f"Error loading models: {e}")
        # Allow startup without models, will return error during requests

@app.get("/")
def read_root():
    """Root endpoint to check if API is running"""
    return {"status": "ok", "message": "Procurement Clustering API is running"}

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    if pca_model is None or hdbscan_model is None or cluster_profiles is None:
        return {"status": "error", "message": "Models not loaded"}
    return {"status": "ok", "models_loaded": True}

@app.get("/api/clusters")
def get_clusters():
    """Get available clusters and their profiles"""
    if cluster_profiles is None:
        raise HTTPException(status_code=500, detail="Cluster profiles not loaded")

    return {
        "num_clusters": len(cluster_profiles),
        "clusters": cluster_profiles.to_dict(orient="records")
    }

@app.post("/api/predict", response_model=ClusterResponse)
def predict_cluster(contract: ContractData):
    """Predict cluster for new contract data"""
    if pca_model is None or hdbscan_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # Process input data
        features = preprocess_input(contract)

        # Transform with PCA
        pca_features = pca_model.transform(features.reshape(1, -1))

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
        profile = cluster_profiles[cluster_profiles['cluster_id'] == cluster_id].to_dict(orient="records")

        # Find similar clusters
        similar = find_similar_clusters(contract.dict(), cluster_profiles, top_n=3)

        return {
            "cluster_id": cluster_id,
            "probability": probability,
            "similar_clusters": similar,
            "cluster_profile": profile[0] if profile else {}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def preprocess_input(contract_data: ContractData):
    """Preprocess input contract data to match model requirements"""
    # Get dictionary representation
    data = contract_data.dict()

    # Create feature vector (match order used during training)
    numeric_features = np.array([
        data['montant'],
        data['dureeMois'],
        data.get('offresRecues', 1)
    ])

    # Encode categorical features (must match training encoding)
    # This is simplified; in production you'd use the same encoder as training
    categorical_features = []

    # Add procedure one-hot encoding
    if data['procedure'] == 'PROCEDURE_ADAPTEE':
        categorical_features.extend([1, 0, 0])
    elif data['procedure'] == 'PROCEDURE_FORMALISEE':
        categorical_features.extend([0, 1, 0])
    else:
        categorical_features.extend([0, 0, 1])

    # Add nature one-hot encoding
    if data['nature'] == 'SERVICES':
        categorical_features.extend([1, 0, 0])
    elif data['nature'] == 'TRAVAUX':
        categorical_features.extend([0, 1, 0])
    else:
        categorical_features.extend([0, 0, 1])

    # Add formePrix one-hot encoding
    if data.get('formePrix') == 'PRIX_REVISABLES':
        categorical_features.append(1)
    else:
        categorical_features.append(0)

    # Combine features
    all_features = np.concatenate([numeric_features, categorical_features])

    return all_features

def find_similar_clusters(contract_data, profiles_df, top_n=3):
    """Find clusters similar to the input contract"""
    # Extract key metrics
    contract_dict = {
        'amount': contract_data['montant'],
        'duration': contract_data['dureeMois'],
    }

    if 'codeCPV' in contract_data and contract_data['codeCPV']:
        contract_dict['cpv'] = contract_data['codeCPV']

    # Score each cluster
    scores = []

    for _, profile in profiles_df.iterrows():
        score = 0
        weight_sum = 0

        # Compare CPV codes if available
        if 'top_cpv' in profile and 'cpv' in contract_dict:
            if profile['top_cpv'] == contract_dict['cpv']:
                score += 3  # High weight for matching CPV
            weight_sum += 3

        # Compare contract amount
        if 'median_amount' in profile and 'amount' in contract_dict:
            amount_ratio = min(profile['median_amount'], contract_dict['amount']) / max(profile['median_amount'], contract_dict['amount'])
            score += amount_ratio * 2  # Weight of 2 for amount similarity
            weight_sum += 2

        # Compare duration
        if 'median_duration' in profile and 'duration' in contract_dict:
            if profile['median_duration'] > 0 and contract_dict['duration'] > 0:
                duration_ratio = min(profile['median_duration'], contract_dict['duration']) / max(profile['median_duration'], contract_dict['duration'])
                score += duration_ratio * 1  # Weight of 1 for duration similarity
                weight_sum += 1

        # Normalize score
        normalized_score = score / weight_sum if weight_sum > 0 else 0
        scores.append(normalized_score)

    # Create a copy with scores
    profiles = profiles_df.copy()
    profiles['similarity_score'] = scores

    # Return top similar clusters
    similar = profiles.sort_values('similarity_score', ascending=False).head(top_n)
    return similar.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
