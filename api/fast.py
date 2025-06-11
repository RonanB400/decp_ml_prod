# api/fast.py
import hdbscan
import numpy as np
import pandas as pd
import os
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

# Load CPV descriptions
cpv_descriptions = {}
try:
    cpv_desc_path = "models/cpv_descriptions.csv"
    if os.path.exists(cpv_desc_path):
        cpv_df = pd.read_csv(cpv_desc_path)
        cpv_descriptions = dict(zip(cpv_df["codeCPV_2"].astype(int), cpv_df["codeCPV_FR"]))
        print(f"Loaded {len(cpv_descriptions)} CPV descriptions")
    else:
        print(f"CPV descriptions file not found: {cpv_desc_path}")
except Exception as e:
    print(f"Error loading CPV descriptions: {str(e)}")

def get_cpv_description(cpv_code):
    """Get human-readable French description for a CPV code"""
    try:
        # Extract first two digits for category
        if cpv_code is None:
            return "Catégorie inconnue"

        category_code = int(str(cpv_code)[:2])

        if category_code in cpv_descriptions:
            return cpv_descriptions[category_code]
        else:
            return f"Services avec code CPV {cpv_code}"
    except (ValueError, TypeError):
        return "Catégorie inconnue"

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

@app.get("/api/clusters/top")
def get_top_clusters(n: int = 10):
    """
    Get the top N biggest clusters by size

    Parameters:
    - n: Number of top clusters to return (default: 10)
    """
    if cluster_profiles is None or len(cluster_profiles) == 0:
        raise HTTPException(status_code=404, detail="No cluster profiles available")

    # Sort clusters by size in descending order and take top n
    top_clusters = cluster_profiles.sort_values(by="size", ascending=False).head(n)

    return {
        "num_clusters": len(top_clusters),
        "clusters": top_clusters.to_dict(orient="records")
    }

@app.get("/api/clusters/insights")
def get_cluster_insights(n: int = 10):
    """
    Get human-readable insights for the top N clusters

    Parameters:
    - n: Number of top clusters to return insights for (default: 10)
    """
    if cluster_profiles is None or len(cluster_profiles) == 0:
        raise HTTPException(status_code=404, detail="No cluster profiles available")

    # Get top n clusters by size
    top_clusters = cluster_profiles.sort_values(by="size", ascending=False).head(n)

    insights = []

    for _, cluster in top_clusters.iterrows():
        # Format large numbers with commas
        size = f"{int(cluster['size']):,}"
        median_amount = f"{cluster['median_amount']:,.2f}€"
        median_duration = f"{cluster['median_duration']:.1f}"

        # Get CPV category description
        cpv_code = cluster.get('top_cpv')
        cpv_description = get_cpv_description(cpv_code)
        cpv_percentage = f"{cluster.get('top_cpv_pct', 0) * 100:.1f}%"

        # Create a human-readable description
        description = (
            f"Le cluster {int(cluster['cluster_id'])} représente {size} marchés "
            f"principalement pour '{cpv_description}' ({cpv_percentage} des marchés). "
            f"Les contrats types ont une valeur médiane de {median_amount} "
            f"et durent {median_duration} mois. "
        )

        # Add information about contract procedure if available
        if 'top_procedure' in cluster and 'top_procedure_pct' in cluster:
            procedure = cluster['top_procedure']
            proc_pct = f"{cluster['top_procedure_pct'] * 100:.1f}%"
            description += f"La plupart des marchés ({proc_pct}) utilisent la procédure '{procedure}'. "

        # Add information about price structure if available
        if 'top_forme_prix' in cluster and 'top_forme_prix_pct' in cluster:
            price_form = cluster['top_forme_prix']
            price_pct = f"{cluster['top_forme_prix_pct'] * 100:.1f}%"
            description += f"{price_pct} utilisent le format de prix '{price_form}'. "

        insights.append({
            "cluster_id": int(cluster['cluster_id']),
            "size": int(cluster['size']),
            "main_category": cpv_description,
            "median_value": float(cluster['median_amount']),
            "median_duration": float(cluster['median_duration']),
            "description": description,
            "stats": {
                "mean_amount": float(cluster.get('mean_amount', 0)),
                "median_amount": float(cluster.get('median_amount', 0)),
                "min_amount": float(cluster.get('min_amount', 0)),
                "max_amount": float(cluster.get('max_amount', 0)),
                "mean_duration": float(cluster.get('mean_duration', 0)),
                "median_duration": float(cluster.get('median_duration', 0)),
            }
        })

    return {
        "num_clusters": len(insights),
        "insights": insights
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
