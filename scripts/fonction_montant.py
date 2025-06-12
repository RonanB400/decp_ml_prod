import numpy as np
import matplotlib.pyplot as plt

from scripts.preprocess_pipeline import create_pipeline

from sklearn.model_selection import train_test_split


numerical_columns = ['dureeMois', 'offresRecues', 'annee']

binary_columns = ['sousTraitanceDeclaree', 'origineFrance',
                          'marcheInnovant', 'idAccordCadre']

categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'tauxAvance_cat',
                               'codeCPV_3', 'acheteur_tranche_effectif', 'acheteur_categorie']

pipeline = create_pipeline(numerical_columns, binary_columns, categorical_columns)


def predict_X(X):

    X_preproc = pipeline.transform(X)

    from keras.models import load_model

    model = load_model("model_montant_100.keras")
    y = model.predict(X_preproc)

    return y


# y = predict_X(X)
# # Créer les indices (abs) pour l'axe des abscisses
# abs = np.arange(len(y))

# # Tracer l'histogramme
# plt.bar(abs, y, color='skyblue', label='Histogramme')

# # Tracer la courbe par-dessus
# plt.plot(abs, y, color='darkblue', linewidth=2, label='Courbe')

# # Ajouter des titres et une légende
# plt.title("Probabilités des fouchettes de prix")
# plt.xlabel("Fourchette")
# plt.ylabel("Probabilité")
# plt.legend()

# # Afficher le graphique
# plt.tight_layout()
# plt.show()
