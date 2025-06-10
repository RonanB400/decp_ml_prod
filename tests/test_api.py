from fastapi.testclient import TestClient
from api.fast import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Bienvenue sur l'API d'analyse des marchés publics"}

def test_get_cluster_profile():
    # Replace with an existing cluster ID
    response = client.get("/cluster_profile/0")
    assert response.status_code == 200
    # Add more assertions based on expected data

def test_predict():
    test_data = {
        "montant": 100000.0,
        "duree_mois": 12,
        "code_cpv_2_digits": "45",
        "type_procedure": "Appel d'offres ouvert"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "predicted_cluster" in response.json()
