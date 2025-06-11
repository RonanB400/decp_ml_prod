from fastapi.testclient import TestClient
from api.fast import app
import hdbscan
import numpy as np
import pytest
from unittest.mock import patch, Mock
import pandas as pd

client = TestClient(app)

# Mock data for testing
MOCK_CLUSTERS = pd.DataFrame({
    'cluster_id': [0, 1],
    'size': [100, 200],
    'pct_total': [10.0, 20.0],
    'top_cpv': ['33000000', '45000000'],
    'top_cpv_pct': [60.0, 70.0],
    'mean_amount': [12000.0, 22000.0],
    'median_amount': [10000.0, 20000.0],
    'mean_duration': [14.0, 26.0],
    'median_duration': [12.0, 24.0],
})

@pytest.fixture
def mock_models():
    """Fixture to mock loaded models"""
    with patch('api.model_loader.load_models') as mock_loader:
        # Create mock models
        mock_pca = Mock()
        mock_pca.transform.return_value = np.array([[0.1, 0.2]])

        mock_hdbscan = Mock()
        mock_hdbscan.exemplars_ = [np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])]

        mock_pipeline = Mock()
        mock_pipeline.transform.return_value = np.array([[1.0, 2.0, 3.0]])

        # Instead of mocking DataFrame, use a real DataFrame
        mock_profiles = MOCK_CLUSTERS.copy()

        # Set the return value for load_models
        mock_loader.return_value = (mock_pca, mock_hdbscan, mock_pipeline, mock_profiles)

        # Patch the models in api.fast
        with patch('api.fast.pca_model', mock_pca), \
             patch('api.fast.hdbscan_model', mock_hdbscan), \
             patch('api.fast.preprocessing_pipeline', mock_pipeline), \
             patch('api.fast.cluster_profiles', mock_profiles):

            yield

def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_health_check_no_models():
    """Test health check when models are not loaded"""
    with patch('api.fast.pca_model', None):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "error"

@pytest.mark.usefixtures("mock_models")
def test_health_check_with_models():
    """Test health check when models are loaded"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.usefixtures("mock_models")
def test_get_clusters():
    """Test getting cluster profiles"""
    response = client.get("/api/clusters")
    assert response.status_code == 200
    assert "num_clusters" in response.json()
    assert "clusters" in response.json()

@pytest.mark.usefixtures("mock_models")
def test_predict_cluster():
    """Test cluster prediction"""
    # Mock the approximate_predict function
    with patch('hdbscan.approximate_predict', return_value=(np.array([1]), np.array([0.8]))):
        test_contract = {
            "montant": 15000,
            "dureeMois": 18,
            "offresRecues": 3,
            "procedure": "PROCEDURE_ADAPTEE",
            "nature": "SERVICES",
            "formePrix": "FORFAIT",
            "sousTraitanceDeclaree": False,
            "origineFrance": True,
            "codeCPV_2_3": "33000000"
        }

        response = client.post("/api/predict", json=test_contract)
        assert response.status_code == 200
        result = response.json()
        assert "cluster_id" in result
        assert "probability" in result
        assert "similar_clusters" in result
        assert "cluster_profile" in result
        assert result["cluster_id"] == 1
        assert result["probability"] == 0.8

@pytest.mark.usefixtures("mock_models")
def test_predict_noise_cluster():
    """Test prediction with noise assignment"""
    # Mock the approximate_predict function to return noise
    with patch('hdbscan.approximate_predict', return_value=(np.array([-1]), np.array([0.0]))):
        test_contract = {
            "montant": 15000,
            "dureeMois": 18,
            "offresRecues": 3,
            "procedure": "PROCEDURE_ADAPTEE",
            "nature": "SERVICES",
            "formePrix": "FORFAIT",
            "sousTraitanceDeclaree": False,
            "origineFrance": True
        }

        response = client.post("/api/predict", json=test_contract)
        assert response.status_code == 200
        result = response.json()
        assert result["is_noise"] == True
        assert result["probability"] == 0.0
        assert "nearest_cluster" in result

# The helper functions below are useful for model training but not needed for API tests
# You can keep them for reference or move them to a separate module

def get_hdbscan_model():
    """
    Create and return an HDBSCAN clustering model with optimized parameters.
    """
    # Parameters found through optimization in the notebook
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=55,
        min_samples=20,
        cluster_selection_epsilon=0.3,
        metric='euclidean',
        gen_min_span_tree=True,
        cluster_selection_method='eom'
    )
    return clusterer

def train_clustering_model(X_train_pca):
    """Train the HDBSCAN model on PCA-reduced data."""
    clusterer = get_hdbscan_model()
    clusterer.fit(X_train_pca)
    cluster_labels = clusterer.labels_

    # Make the model prediction-ready
    clusterer.generate_prediction_data()

    return clusterer, cluster_labels

def predict_cluster(model, test_pca):
    """Predict the cluster for new data using the trained HDBSCAN model."""
    # Use HDBSCAN's approximate_predict method to assign to a cluster
    cluster_id, cluster_prob = hdbscan.approximate_predict(model, test_pca)

    # Get the cluster ID and probability
    predicted_cluster = cluster_id[0]
    probability = cluster_prob[0]

    # If assigned to noise, find the nearest cluster
    if predicted_cluster == -1:
        distances = [np.min(np.linalg.norm(test_pca - model.exemplars_[c], axis=1))
                   for c in range(len(model.exemplars_))]
        nearest_cluster = np.argmin(distances)
        return nearest_cluster, 0.0

    return predicted_cluster, probability
