import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score

class HDBSCANClusterer:
    """
    HDBSCAN clustering model for procurement data.
    Identifies similar contracts based on PCA-reduced features.
    """

    def __init__(self,
                min_cluster_size=55,
                min_samples=20,
                cluster_selection_epsilon=0.3,
                metric='euclidean',
                cluster_selection_method='eom'):
        """
        Initialize the HDBSCAN clustering model with parameters.

        Parameters:
        -----------
        min_cluster_size : int, default=55
            The minimum size of clusters
        min_samples : int, default=20
            The number of samples in a neighborhood for a point to be considered a core point
        cluster_selection_epsilon : float, default=0.3
            The distance threshold for cluster merging
        metric : str, default='euclidean'
            The metric to use for distance computation
        cluster_selection_method : str, default='eom'
            The method to use for selecting clusters ('eom' or 'leaf')
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

        # Initialize the HDBSCAN model
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            gen_min_span_tree=True,
            cluster_selection_method=self.cluster_selection_method
        )

        self.labels_ = None
        self.profiles_df = None

    def fit(self, X):
        """
        Fit the HDBSCAN model to the data.

        Parameters:
        -----------
        X : array-like
            The input data to cluster, typically PCA-reduced features

        Returns:
        --------
        self : object
            Returns self
        """
        self.clusterer.fit(X)
        self.labels_ = self.clusterer.labels_

        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.

        Parameters:
        -----------
        X : array-like
            The input data to cluster

        Returns:
        --------
        labels : array
            Cluster labels for each point
        """
        self.labels_ = self.clusterer.fit_predict(X)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for new data points.
        Uses approximate_predict from HDBSCAN.

        Parameters:
        -----------
        X : array-like
            New data points to predict cluster for

        Returns:
        --------
        labels : array
            Predicted cluster labels
        probabilities : array
            Probabilities of cluster membership
        """
        if not hasattr(self.clusterer, 'prediction_data_'):
            self.clusterer.generate_prediction_data()

        # Use HDBSCAN's approximate_predict
        labels, probabilities = hdbscan.approximate_predict(self.clusterer, X)
        return labels, probabilities

    def evaluate(self):
        """
        Evaluate the clustering model using metrics.

        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet")

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        noise_ratio = n_noise / len(self.labels_)

        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio
        }

        # Calculate silhouette score if more than one cluster and not all noise
        if n_clusters > 1 and n_noise < len(self.labels_):
            # Filter out noise points
            mask = self.labels_ != -1
            if np.sum(mask) > 1 and len(np.unique(self.labels_[mask])) > 1:
                silhouette_avg = silhouette_score(
                    self.clusterer._raw_data[mask],
                    self.labels_[mask]
                )
                metrics['silhouette_score'] = silhouette_avg

        return metrics

    def create_cluster_profiles(self, X_orig, X_pca=None):
        """
        Create profiles for each cluster based on original data features.

        Parameters:
        -----------
        X_orig : DataFrame
            Original dataset with features
        X_pca : array-like, optional
            PCA-transformed data used for clustering

        Returns:
        --------
        DataFrame
            Profiles for each cluster
        """
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet")

        # Add cluster labels to original data
        X_with_clusters = X_orig.copy()
        X_with_clusters['cluster'] = self.labels_

        profiles = []

        for cluster_id in np.sort(np.unique(self.labels_)):
            # Skip noise points
            if cluster_id == -1:
                continue

            # Get data for this cluster
            cluster_data = X_with_clusters[X_with_clusters['cluster'] == cluster_id]

            # Skip very small clusters
            if len(cluster_data) < 20:
                continue

            # Calculate key metrics
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'pct_total': round(len(cluster_data) / len(X_with_clusters) * 100, 2),
            }

            # Add CPV characteristics if available
            cpv_column = [col for col in X_orig.columns if col.startswith('codeCPV')]
            if cpv_column and cpv_column[0] in cluster_data.columns:
                cpv_col = cpv_column[0]
                profile.update({
                    'top_cpv': cluster_data[cpv_col].value_counts().index[0],
                    'top_cpv_pct': round(cluster_data[cpv_col].value_counts().iloc[0] / len(cluster_data) * 100, 2),
                    'cpv_diversity': len(cluster_data[cpv_col].unique()),
                })

            # Add financial characteristics if available
            if 'montant' in cluster_data.columns:
                profile.update({
                    'mean_amount': round(cluster_data['montant'].mean(), 2),
                    'median_amount': round(cluster_data['montant'].median(), 2),
                    'amount_std': round(cluster_data['montant'].std(), 2),
                })

            # Add duration characteristics if available
            if 'dureeMois' in cluster_data.columns:
                profile.update({
                    'mean_duration': round(cluster_data['dureeMois'].mean(), 2),
                    'median_duration': round(cluster_data['dureeMois'].median(), 2),
                })

            # Add derived metrics
            if 'montant' in cluster_data.columns and 'dureeMois' in cluster_data.columns:
                if profile['median_duration'] > 0:
                    profile['euro_per_month'] = round(profile['median_amount'] / profile['median_duration'], 2)
                else:
                    profile['euro_per_month'] = round(profile['median_amount'], 2)

            profiles.append(profile)

        # Convert to DataFrame for easier analysis
        self.profiles_df = pd.DataFrame(profiles).sort_values('size', ascending=False)
        return self.profiles_df
