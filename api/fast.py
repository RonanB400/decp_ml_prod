# api/fast.py
import datetime
import hdbscan
import numpy as np
import pandas as pd
import os
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


# Load CPV descriptions with enhanced debugging
cpv_descriptions = {}
try:
    # Try both relative and absolute paths
    cpv_desc_path = "models/cpv_descriptions.csv"
    alt_path = os.path.join(os.path.dirname(__file__), "..", "models/cpv_descriptions.csv")

    if os.path.exists(cpv_desc_path):
        print(f"Loading CPV descriptions from {cpv_desc_path}")
        cpv_df = pd.read_csv(cpv_desc_path)
    elif os.path.exists(alt_path):
        print(f"Loading CPV descriptions from alternate path {alt_path}")
        cpv_df = pd.read_csv(alt_path)
    else:
        print(f"CPV descriptions file not found at {cpv_desc_path} or {alt_path}")
        raise FileNotFoundError(f"CPV file not found at either path")

    # Print the first few rows to verify content
    print(f"CSV first 3 rows: \n{cpv_df.head(3)}")

    # Ensure the column name matches what's in your CSV file
    cpv_columns = list(cpv_df.columns)
    print(f"Found CPV columns: {cpv_columns}")

    # Get the code column and description column
    code_col = cpv_columns[0]  # First column (codeCPV_2)
    desc_col = cpv_columns[1]  # Second column (codeCPV_FR)

    # Create string version of the dictionary for debug
    str_dict = {str(row[code_col]): row[desc_col] for _, row in cpv_df.iterrows()}
    print(f"Loaded {len(str_dict)} CPV string descriptions")

    # Convert all CPV codes to integers for consistent lookup
    cpv_df[code_col] = cpv_df[code_col].astype(int)

    # Create a dictionary with CPV codes as keys and descriptions as values
    # Use both string and integer versions to ensure matches
    cpv_descriptions = dict(zip(cpv_df[code_col], cpv_df[desc_col]))

    # Also create string versions of keys for fallback
    for key, value in cpv_descriptions.items():
        str_dict[str(key)] = value

    print(f"Loaded {len(cpv_descriptions)} CPV integer descriptions")

    # Specific check for 45200000
    test_code = 45200000
    test_code_str = "45200000"
    print(f"45200000 in integer dict? {test_code in cpv_descriptions}")
    print(f"'45200000' in string dict? {test_code_str in str_dict}")

    if test_code in cpv_descriptions:
        print(f"CPV 45200000 description: {cpv_descriptions[test_code]}")
    elif test_code_str in str_dict:
        print(f"CPV '45200000' description (string key): {str_dict[test_code_str]}")
        # Copy to integer dict for future lookups
        cpv_descriptions[test_code] = str_dict[test_code_str]
    else:
        print("CPV 45200000 not found in any dictionary")
        # Print nearby keys to debug
        all_keys = sorted([int(k) for k in cpv_descriptions.keys()])
        idx = -1
        for i, k in enumerate(all_keys):
            if k > test_code:
                idx = max(0, i-1)
                break
        print(f"Nearby keys: {all_keys[max(0,idx-3):min(len(all_keys),idx+3)]}")

except Exception as e:
    print(f"Error loading CPV descriptions: {str(e)}")
    import traceback
    print(traceback.format_exc())

def get_cpv_description(cpv_code):
    """
    Get human-readable French description for a CPV code with enhanced robustness.
    """
    try:
        if cpv_code is None:
            return "Catégorie inconnue"

        # Print debug info for problematic CPV codes
        if cpv_code == 45200000:
            print(f"Looking up problematic CPV 45200000")
            print(f"Type: {type(cpv_code)}")
            print(f"In dictionary? {cpv_code in cpv_descriptions}")

        # Try integer lookup first
        if isinstance(cpv_code, (int, float)):
            cpv_int = int(cpv_code)
            if cpv_int in cpv_descriptions:
                return cpv_descriptions[cpv_int]

        # Try string conversion
        cpv_str = str(cpv_code).replace('-', '').strip()
        cpv_int = int(cpv_str)

        # Try direct lookup with the integer
        if cpv_int in cpv_descriptions:
            return cpv_descriptions[cpv_int]

        # Special case for known problematic codes
        if cpv_int == 45200000:
            return "Travaux de construction complète ou partielle et travaux de génie civil"

        # Try prefix matching (hierarchical approach)
        for length in range(len(cpv_str), 1, -2):
            try:
                prefix = int(cpv_str[:length])
                if prefix in cpv_descriptions:
                    return cpv_descriptions[prefix]
            except:
                pass

        # Return generic description as last resort
        return f"Services avec code CPV {cpv_code}"
    except Exception as e:
        print(f"Error in get_cpv_description for {cpv_code}: {str(e)}")
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
            f"Le cluster {int(cluster['cluster_id'])} comprend {size} autres contrats "
            f"principalement pour '{cpv_description}' ({cpv_percentage} des marchés du cluster). "
            f"Les contrats types de ce cluster ont une valeur médiane de {median_amount} "
            f"et durent majoritairement {median_duration} mois. "
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

    Returns cluster ID, probability, similar clusters and a summary description
    Along with examples of nearest observations
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

            # Get nearest cluster profile
            nearest_profile = cluster_profiles[
                cluster_profiles['cluster_id'] == nearest_cluster
            ]

            # Generate description for nearest cluster
            summary_description = generate_cluster_description(nearest_profile.iloc[0] if len(nearest_profile) > 0 else None,
                                                              "Ce marché n'appartient clairement à aucun cluster, mais le plus proche est")

            response = {
                "cluster_id": -1,
                "nearest_cluster": nearest_cluster,
                "probability": 0.0,
                "is_noise": True,
                "summary_description": summary_description
            }
            similar_clusters = []
        else:
            # Get cluster profile
            cluster_profile = cluster_profiles[
                cluster_profiles['cluster_id'] == predicted_cluster
            ]

            # Generate human-readable description
            cluster_data = cluster_profile.iloc[0] if len(cluster_profile) > 0 else None
            confidence_level = get_confidence_level(prob)
            summary_description = generate_cluster_description(cluster_data,
                                                             f"Ce marché appartient {confidence_level} au cluster")

            # Find similar clusters based on contract data
            similar_clusters = find_similar_clusters(
                contract_data=contract.dict(),
                profiles_df=cluster_profiles[
                    cluster_profiles['cluster_id'] != predicted_cluster
                ],
                top_n=5  # Adjust number as needed
            )

            # Add descriptions to similar clusters
            for i, cluster in enumerate(similar_clusters):
                cluster_id = cluster.get('cluster_id')
                cluster_data = cluster_profiles[cluster_profiles['cluster_id'] == cluster_id].iloc[0] if len(cluster_profiles[cluster_profiles['cluster_id'] == cluster_id]) > 0 else None
                similar_clusters[i]['description'] = generate_cluster_description(cluster_data, "Ce cluster représente")

            response = {
                "cluster_id": predicted_cluster,
                "probability": prob,
                "is_noise": False,
                "summary_description": summary_description,
                "cluster_profile": (
                    cluster_profile.to_dict(orient="records")[0]
                    if len(cluster_profile) > 0 else {}
                ),
                "similar_clusters": similar_clusters
            }

            # Find the nearest exemplars from the cluster
            if predicted_cluster != -1:
                # Get the exemplars for this cluster
                cluster_exemplars = hdbscan_model.exemplars_[predicted_cluster]

                # Calculate distances to all exemplars in this cluster
                distances = np.linalg.norm(contract_pca - cluster_exemplars, axis=1)

                # Get indices of the closest exemplars (top 5)
                closest_indices = np.argsort(distances)[:5]

                # Extract the nearest exemplars with more meaningful information
                nearest_exemplars = []
                for idx in closest_indices:
                    # Get exemplar features
                    exemplar = cluster_exemplars[idx]

                    # Calculate similarity score (1 - normalized distance)
                    distance = distances[idx]
                    max_distance = np.max(distances) if np.max(distances) > 0 else 1
                    similarity = 1 - (distance / max_distance)

                    # Create a more meaningful description of this exemplar
                    exemplar_info = {
                        "similarity_score": float(similarity),
                        "distance": float(distance)
                    }

                    # Add a human-readable interpretation of this exemplar
                    # Each field describes a key characteristic of this example contract
                    contract_characteristics = {
                        "montant_relatif": "plus élevé" if exemplar[0] > 0 else "moins élevé",
                        "durée_relative": "plus longue" if exemplar[1] > 0 else "plus courte",
                        "offres_reçues": "nombreuses" if exemplar[2] > 0 else "peu nombreuses",
                        "procédure": "appel d'offres" if exemplar[3] > 0 else "autre procédure",
                        "forme_prix": "unitaire" if exemplar[6] > 0 else "autre forme"
                    }

                    # Describe this contract example in human terms
                    description = (
                        f"Cet exemple de contrat similaire a un montant {contract_characteristics['montant_relatif']} "
                        f"que la moyenne, une durée {contract_characteristics['durée_relative']}, avec "
                        f"des offres {contract_characteristics['offres_reçues']}. "
                        f"Ce contrat a utilisé une procédure de type {contract_characteristics['procédure']} "
                        f"et une forme de prix {contract_characteristics['forme_prix']}."
                    )

                    exemplar_info["description"] = description

                    # Get cluster profile data for estimates
                    if len(cluster_profile) > 0:
                        median_amount = cluster_profile.iloc[0]['median_amount']
                        amount_std = cluster_profile.iloc[0]['amount_std']
                        median_duration = cluster_profile.iloc[0]['median_duration']
                    else:
                        # Fallback values if cluster profile is missing
                        median_amount = 600000
                        amount_std = 500000
                        median_duration = 24

                    # Add estimated values for key features
                    feature_values = {}

                    # Montant (Amount) - Using cluster statistics for better estimates
                    # Higher positive values mean higher amounts
                    if exemplar[0] > 2:  # Very high
                        estimated_amount = median_amount + (amount_std * 1.5)
                    elif exemplar[0] > 0:  # Moderately high
                        estimated_amount = median_amount + (amount_std * 0.5)
                    elif exemplar[0] > -1:  # Slightly below average
                        estimated_amount = median_amount * 0.8
                    else:  # Much below average
                        estimated_amount = median_amount * 0.5

                    feature_values["montant"] = f"{estimated_amount:,.2f}€"

                    # Duration - Using cluster median duration as reference
                    if exemplar[1] > 2:  # Very long
                        estimated_duration = median_duration * 1.5
                    elif exemplar[1] > 0:  # Above average
                        estimated_duration = median_duration * 1.2
                    elif exemplar[1] > -1:  # Slightly below average
                        estimated_duration = median_duration * 0.8
                    else:  # Much shorter
                        estimated_duration = median_duration * 0.5

                    feature_values["dureeMois"] = f"{estimated_duration:.1f} mois"

                    # Offers received - Estimate number of offers
                    if exemplar[2] > 1:
                        estimated_offers = "10+"
                    elif exemplar[2] > 0:
                        estimated_offers = "5-9"
                    elif exemplar[2] > -1:
                        estimated_offers = "2-4"
                    else:
                        estimated_offers = "1"

                    feature_values["offresRecues"] = estimated_offers

                    # Procedure type
                    if exemplar[3] > 0:
                        feature_values["procedure"] = "Appel d'offres ouvert"
                    else:
                        feature_values["procedure"] = "Procédure adaptée"

                    # Contract nature (typically binary)
                    if exemplar[4] > 0:
                        feature_values["nature"] = "Marché"
                    else:
                        feature_values["nature"] = "Accord-cadre"

                    # Price form
                    if exemplar[6] > 0:
                        feature_values["formePrix"] = "Unitaire"
                    else:
                        feature_values["formePrix"] = "Forfaitaire"

                    exemplar_info["feature_values"] = feature_values

                    # Only include the raw features for developer/debug purposes
                    exemplar_info["features"] = exemplar.tolist()

                    nearest_exemplars.append(exemplar_info)

                # Add to response
                response["nearest_examples"] = nearest_exemplars

        return response

    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())  # Print the full traceback
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

def get_confidence_level(probability):
    """Return a confidence level based on probability"""
    if probability >= 0.9:
        return "avec une très forte probabilité"
    elif probability >= 0.7:
        return "avec une forte probabilité"
    elif probability >= 0.5:
        return "probablement"
    else:
        return "possiblement"

def generate_cluster_description(cluster_data, prefix_text="Ce cluster représente"):
    """Generate a human-readable description for a cluster"""
    if cluster_data is None:
        return "Information de cluster non disponible."

    # Format large numbers with commas
    size = f"{int(cluster_data.get('size', 0)):,}"
    median_amount = f"{cluster_data.get('median_amount', 0):,.2f}€"
    median_duration = f"{cluster_data.get('median_duration', 0):.1f}"

    # Get CPV category description
    cpv_code = cluster_data.get('top_cpv')
    cpv_description = get_cpv_description(cpv_code)

    # Fix percentage calculation - ensure it's a decimal value between 0-1
    # Then convert to percentage for display
    top_cpv_pct = cluster_data.get('top_cpv_pct', 0)
    # If percentage is greater than 1, assume it's already in percentage form and divide by 100
    if top_cpv_pct > 1:
        top_cpv_pct = top_cpv_pct / 100
    cpv_percentage = f"{top_cpv_pct * 100:.1f}%"

    # Create a human-readable description
    description = (
        f"{prefix_text} {int(cluster_data.get('cluster_id', 0))} qui comprend {size} autres contrats "
        f"principalement pour '{cpv_description}' ({cpv_percentage} des marchés). "
        f"Les contrats types ont une valeur médiane de {median_amount} "
        f"et durent majoritairement {median_duration} mois. "
    )

    # Add information about contract procedure if available
    if 'top_procedure' in cluster_data and 'top_procedure_pct' in cluster_data:
        procedure = cluster_data.get('top_procedure', '')
        proc_pct_value = cluster_data.get('top_procedure_pct', 0)
        # Fix percentage calculation for procedure
        if proc_pct_value > 1:
            proc_pct_value = proc_pct_value / 100
        proc_pct = f"{proc_pct_value * 100:.1f}%"
        description += f"La plupart des marchés ({proc_pct}) utilisent la procédure '{procedure}'. "

    # Add information about price structure if available
    if 'top_forme_prix' in cluster_data and 'top_forme_prix_pct' in cluster_data:
        price_form = cluster_data.get('top_forme_prix', '')
        price_pct_value = cluster_data.get('top_forme_prix_pct', 0)
        # Fix percentage calculation for price form
        if price_pct_value > 1:
            price_pct_value = price_pct_value / 100
        price_pct = f"{price_pct_value * 100:.1f}%"
        description += f"{price_pct} utilisent le format de prix '{price_form}'. "

    return description


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
    idAccordCadre: Optional[str] = None
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
            'origineFrance', 'marcheInnovant', 'tauxAvance', 'codeCPV_3'
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
