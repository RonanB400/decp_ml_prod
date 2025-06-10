from fastapi.testclient import TestClient
from api.fast import app
import hdbscan
import numpy as np

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
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
