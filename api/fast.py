# api/fast.py

import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from decp.params import *


# ==============================================================================
# 1. INITIALISATION DE L'API ET CHARGEMENT DES MODÈLES
# ==============================================================================

# Créez une instance de l'application FastAPI
app = FastAPI()

# --- Chargement des modèles au démarrage de l'API ---

# Définir les chemins relatifs vers les modèles
# Note : on part du principe que ce script est dans api/ et que models/ est au même niveau que api/
#MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
#PIPELINE_PATH = os.path.join(MODELS_DIR, 'full_pipeline.pkl')
#MODEL_PATH = os.path.join(MODELS_DIR, 'hdbscan_model.pkl')
#PROFILES_PATH = os.path.join(MODELS_DIR, 'cluster_profiles.csv')

# Charger les artefacts
try:
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    cluster_profiles_df = pd.read_csv(PROFILES_PATH)
    print("Modèles et profils chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")
    pipeline, model, cluster_profiles_df = None, None, None


# ==============================================================================
# 2. DÉFINITION DU MODÈLE DE DONNÉES D'ENTRÉE (PYDANTIC)
# ==============================================================================

# Ce modèle Pydantic définit la structure et les types de données
# que l'API attendra pour une prédiction.
# FastAPI l'utilisera pour valider les requêtes entrantes.
class MarketContract(BaseModel):
    # Ajoutez ici les champs qui correspondent aux colonnes
    # attendues par votre pipeline de prétraitement.
    # Voici quelques exemples :
    montant: float
    duree_mois: int
    code_cpv_2_digits: str # ex: "45"
    type_procedure: str # ex: "Appel d'offres ouvert"
    # ... ajoutez les autres features nécessaires


# ==============================================================================
# 3. DÉFINITION DES POINTS DE TERMINAISON (ENDPOINTS)
# ==============================================================================

# --- Endpoint Racine (GET /) ---
# Idéal pour vérifier rapidement si l'API est en ligne.
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bienvenue sur l'API d'analyse des marchés publics"}


# --- Endpoint pour obtenir le profil d'un cluster (GET /cluster_profile/{cluster_id}) ---
@app.get("/cluster_profile/{cluster_id}")
def get_cluster_profile(cluster_id: int):
    if cluster_profiles_df is None:
        return JSONResponse(status_code=500, content={"message": "Les profils de cluster ne sont pas chargés."})

    # Chercher le profil du cluster dans le DataFrame
    profile = cluster_profiles_df[cluster_profiles_df['cluster_id'] == cluster_id]

    if profile.empty:
        return JSONResponse(status_code=404, content={"message": f"Cluster ID {cluster_id} non trouvé."})

    # Convertir le profil en dictionnaire et le renvoyer
    return profile.to_dict(orient='records')[0]


# --- Endpoint de Prédiction (POST /predict) ---
@app.post("/predict")
def predict_cluster(contract: MarketContract):
    if pipeline is None or model is None:
        return JSONResponse(status_code=500, content={"message": "Les modèles ne sont pas chargés."})

    # 1. Convertir les données d'entrée en DataFrame pandas
    # Le pipeline s'attend à recevoir un DataFrame
    input_df = pd.DataFrame([contract.model_dump()])

    # 2. Prétraiter les données avec le pipeline
    # Le pipeline va gérer l'encodage, la normalisation, la PCA, et l'UMAP
    transformed_data = pipeline.transform(input_df)

    # 3. Prédire le cluster avec HDBSCAN
    # NOTE : HDBSCAN prédit sur de nouvelles données avec la méthode `approximate_predict`
    # qui prend en entrée les données transformées par le réducteur de dimension (UMAP dans votre cas).
    # Votre `full_pipeline.pkl` devrait donc renvoyer ces données prêtes pour la prédiction.
    try:
        # La méthode exacte dépend de comment vous avez sauvegardé votre modèle.
        # hdbscan.approximate_predict est la méthode standard pour les nouveaux points.
        cluster_label, _ = model.approximate_predict(transformed_data)
        predicted_cluster = int(cluster_label[0])
    except Exception as e:
         return JSONResponse(status_code=500, content={"message": f"Erreur lors de la prédiction : {e}"})

    # 4. Renvoyer la prédiction
    return {"predicted_cluster": predicted_cluster}
