# api/fast.py
import datetime
import hdbscan
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import the model loader
from api.model_loader import load_models
# Import the RAG system
from decp_rag.rag_bigquery import RAGQuerySystem

# Import the find_similar_clusters function
from api.prediction import find_similar_clusters

#from decp.params import PCA_PATH, HDBSCAN_PATH, PROFILES_PATH
from decp_amount.amount_query import amount_prediction


# Initialize FastAPI
app = FastAPI(
    title="Lanterne publique",
    description="API Lanterne publique : etudes des marchés publics",
    version="1.0.0"
)

# Load models at startup
models = load_models()
# pca_model, hdbscan_model, preprocessing_pipeline, cluster_profiles = models
(pca_model, hdbscan_model, cluster_pipeline, cluster_profiles,
 amount_pipeline, amount_model) = models

# Initialize RAG system at startup
try:
    rag_system = RAGQuerySystem()
except Exception as e:
    print(f"Warning: RAG system initialization failed: {e}")
    rag_system = None


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


class RAGQuestion(BaseModel):
    """Model for RAG question input"""
    question: str


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
    if (pca_model is None or hdbscan_model is None or
            cluster_pipeline is None):
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Convert contract to DataFrame for preprocessing
    contract_df = pd.DataFrame([contract.dict()])

    # Add tauxAvance column from tauxAvance_cat if needed
    # (e.g., convert "5%" to 5.0, "0%" to 0.0, etc.)
    if ('tauxAvance_cat' in contract_df.columns and
            'tauxAvance' not in contract_df.columns):
        # Map standard values or use a default of 0
        tax_map = {
            "0%": 0.0, "5%": 5.0, "10%": 10.0, "15%": 15.0,
            "20%": 20.0, "25%": 25.0, "30%": 30.0
        }

        def map_taux(val):
            if pd.isna(val):
                return 0.0
            return tax_map.get(val, 0.0)

        contract_df['tauxAvance'] = contract_df['tauxAvance_cat'].apply(
            map_taux
        )

    try:
        # Preprocess the input
        contract_preprocessed = cluster_pipeline.transform(contract_df)

        # Apply PCA
        contract_pca = pca_model.transform(contract_preprocessed)

        # Predict cluster
        cluster_id, probability = hdbscan.approximate_predict(
            hdbscan_model, contract_pca
        )
        predicted_cluster = int(cluster_id[0])
        prob = float(probability[0])

        # Handle noise assignment
        if predicted_cluster == -1:
            # Find nearest cluster
            distances = [
                np.min(np.linalg.norm(
                    contract_pca - hdbscan_model.exemplars_[c], axis=1
                ))
                for c in range(len(hdbscan_model.exemplars_))
            ]
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
            cluster_profile = cluster_profiles[
                cluster_profiles['cluster_id'] == predicted_cluster
            ]

            # Find similar clusters based on contract data
            similar_clusters = find_similar_clusters(
                contract_data=contract.dict(),
                profiles_df=cluster_profiles[
                    cluster_profiles['cluster_id'] != predicted_cluster
                ],
                top_n=5  # Adjust number as needed
            )

            response = {
                "cluster_id": predicted_cluster,
                "probability": prob,
                "is_noise": False,
                "cluster_profile": (
                    cluster_profile.to_dict(orient="records")[0]
                    if len(cluster_profile) > 0 else {}
                ),
                "similar_clusters": similar_clusters
            }

        return response

    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())  # Print the full traceback
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


class AmountRequest(BaseModel):
    """Input data for amount prediction"""
    dureeMois: int
    offresRecues: Optional[int] = 5
    annee: Optional[int] = datetime.datetime.now().year
    procedure: str
    nature: str
    formePrix: str
    ccag: Optional[str] = "Pas de CCAG"
    sousTraitanceDeclaree: Optional[float] = 0.0
    origineFrance: Optional[float] = 0.0
    marcheInnovant: Optional[float] = 0.0
    idAccordCadre: Optional[int] = 0
    typeGroupementOperateurs: Optional[str] = None
    tauxAvance: Optional[float] = 0.0
    codeCPV_3: Optional[int] = None
    acheteur_tranche_effectif: str
    acheteur_categorie: str

class AmountResponse(BaseModel):
    """Response for amount prediction"""
    prediction: list

@app.post("/api/montant", response_model=AmountResponse)
def predict_amount(request: AmountRequest):
    """
    Predict contract amount using the ML model.
    """
    # Check if models are loaded
    if amount_pipeline is None or amount_model is None:
        raise HTTPException(
            status_code=500,
            detail="Amount models not loaded"
        )
    
    try:
        # Convert request to DataFrame
        X = pd.DataFrame([request.dict()])
        
        # Fix column naming and missing columns to match pipeline expectations
        # Add 'annee' column if missing (use the annee from request or current year)
        if 'annee' not in X.columns:
            X['annee'] = (
                request.annee if hasattr(request, 'annee') and request.annee
                else datetime.datetime.now().year
            )
        
        # Convert tauxAvance to tauxAvance_cat if needed
        if ('tauxAvance' in X.columns and
                'tauxAvance_cat' not in X.columns):
            # Map numeric tauxAvance to categorical tauxAvance_cat
            def map_taux_to_cat(val):
                if pd.isna(val) or val == 0:
                    return "0%"
                elif val <= 5:
                    return "5%"
                elif val <= 10:
                    return "10%"
                elif val <= 15:
                    return "15%"
                elif val <= 20:
                    return "20%"
                elif val <= 25:
                    return "25%"
                else:
                    return "30%"
            
            X['tauxAvance_cat'] = X['tauxAvance'].apply(map_taux_to_cat)
        
        # Convert numeric columns to float64 to match training data dtype
        numeric_cols = [
            'dureeMois', 'offresRecues', 'annee', 'sousTraitanceDeclaree',
            'origineFrance', 'marcheInnovant', 'tauxAvance', 'codeCPV_3',
            'idAccordCadre'
        ]
        for col in numeric_cols:
            if col in X.columns:
                X[col] = X[col].astype('float64')
        
        y_pred = amount_prediction(X, amount_pipeline, amount_model)
        return {"prediction": y_pred.tolist()}
    except Exception as e:
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Montant prediction error: {str(e)}"
        )


@app.post("/api/rag")
def rag_query(question: RAGQuestion):
    """
    Query the RAG system with a natural language question
    
    Returns the natural language answer
    """
    # Check if RAG system is loaded
    if rag_system is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG system not initialized"
        )
    
    try:
        # Query the RAG system - this returns just the answer string
        answer = rag_system.query(question.question)
        
        return {"answer": answer}
    
    except Exception as e:
        import traceback
        print(f"Error during RAG query: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"RAG query error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
