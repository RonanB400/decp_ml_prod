import os
import pickle

from keras.models import load_model

# Charger le pipeline depuis un fichier .pkl
with open('../models/pipeline_pred_montant.pkl', 'rb') as f:
    pipeline = pickle.load(f)

def amount_prediction(X):
    #preprocessing du marché
    X_preproc = pipeline.transform(X)
    #chargement du model de prediction du montant
    model = load_model(os.path.join('..','models','model_montant_100.keras'))
    #prediction des probabilités de fourchettes de prix
    y = model.predict(X_preproc)

    return y


# # ...existing code...

# if __name__ == "__main__":
#     # Exemple d'utilisation pour tester le script
#     import pandas as pd

#     # Remplacer par un DataFrame de test adapté à votre pipeline
#     X_test = pd.DataFrame([{
#         # 'col1': value1,
#         # 'col2': value2,
#         # ...
#     }])

#     y_pred = amount_prediction(X_test)
#     print("Prédiction :", y_pred)
# # ...existing code...
