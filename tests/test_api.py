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
    'median_amount': [10000, 20000],
    'median_duration': [12, 24]
})

@pytest.fixture
def mock_models():
    """Fixture to mock loaded models"""
    with patch('api.fast.pca_model') as mock_pca, \
         patch('api.fast.hdbscan_model') as mock_hdbscan, \
         patch('api.fast.cluster_profiles') as mock_profiles:

        # Configure mocks
        mock_pca.transform.return_value = np.array([[0.1, 0.2]])

        mock_hdbscan_exemplars_ = [np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])]
        mock_hdbscan.exemplars_ = mock_hdbscan_exemplars_

        mock_profiles.__len__.return_value = 2
        mock_profiles.to_dict.return_value = [
            {'cluster_id': 0, 'size': 100},
            {'cluster_id': 1, 'size': 200}
        ]
        mock_profiles.__getitem__.return_value = mock_profiles

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

@patch('api.fast.pca_model', Mock())
@patch('api.fast.hdbscan_model', Mock())
@patch('api.fast.cluster_profiles', Mock())
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
            "formePrix": "FORFAIT"
        }

        response = client.post("/api/predict", json=test_contract)
        assert response.status_code == 200
        result = response.json()
        assert "cluster_id" in result
        assert "probability" in result
        assert "similar_clusters" in result
        assert "cluster_profile" in result

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
            "formePrix": "FORFAIT"
        }

        response = client.post("/api/predict", json=test_contract)
        assert response.status_code == 200
        assert response.json()["probability"] == 0.0  # For noise points

def get_hdbscan_model():
    """
    Create and return an HDBSCAN clustering model with optimized parameters.

    Returns:
        hdbscan.HDBSCAN: A configured HDBSCAN clustering model
    """

    # Parameters found through optimization in the notebook
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=55,
        min_samples=20,
        cluster_selection_epsilon=0.3,
        metric='euclidean',
        gen_min_span_tree=True,
        cluster_selection_method='eom'  # 'eom' is usually better for variable density
    )

    return clusterer


def train_clustering_model(X_train_pca):
    """
    Train the HDBSCAN model on PCA-reduced data.

    Args:
        X_train_pca (numpy.ndarray): PCA-reduced training data

    Returns:
        tuple: (model, cluster_labels)
            - model: Trained HDBSCAN model
            - cluster_labels: Array of cluster assignments for training data
    """
    clusterer = get_hdbscan_model()
    clusterer.fit(X_train_pca)
    cluster_labels = clusterer.labels_

    # Make the model prediction-ready
    clusterer.generate_prediction_data()

    return clusterer, cluster_labels


def predict_cluster(model, test_pca):
    """
    Predict the cluster for new data using the trained HDBSCAN model.

    Args:
        model (hdbscan.HDBSCAN): Trained HDBSCAN model
        test_pca (numpy.ndarray): PCA-reduced test data

    Returns:
        tuple: (predicted_cluster, probability)
            - predicted_cluster: Predicted cluster ID
            - probability: Probability of belonging to the cluster
    """

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
        return nearest_cluster, 0.0  # Return nearest cluster with 0 probability to indicate it's a best guess

    return predicted_cluster, probability
