
📁 decp_ml_prod/
├── 📄 setup.py          # Fichier pour installer les dépendances du projet.
└── 📁 decp/
    ├── 📄 __init__.py
    ├── 📄 params.py        # Fichier de configuration pour les paramètres (chemins, hyperparamètres).
    ├── 📄 utils.py         # Fonctions utilitaires (logs, validation des données).
    │
    ├── 📁 interface/
    │   ├── 📄 __init__.py
    │   ├── 📄 main.py        # Point d'entrée de l'API (FastAPI) pour la prédiction.
    │   └── 📄 main_local.py  # Interface pour tester le modèle en local.
    │
    ├── 📁 ml_logic/
    │   ├── 📄 __init__.py
    │   ├── 📄 data.py        # Charger, nettoyer et transformer les données.
    │   ├── 📄 model.py       # Définir l'architecture du modèle.
    │   ├── 📄 train.py       # Lancer l'entraînement du modèle.
    │   ├── 📄 evaluate.py    # Évaluer les performances du modèle.
    │   └── 📄 predict.py     # Effectuer des prédictions avec un modèle entraîné.
    │
    └── 📁 registry/
        ├── 📄 __init__.py
        └── 📄 registry.py    # Gérer le stockage et la récupération des modèles.
