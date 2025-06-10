import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

class ClusterAnalyzer:
    """
    Tools for analyzing and interpreting HDBSCAN clusters in procurement data.
    """

    def __init__(self, clusterer, X_orig, X_pca=None, cpv_column=None):
        """
        Initialize cluster analyzer.

        Parameters:
        -----------
        clusterer : HDBSCANClusterer
            Fitted HDBSCAN clustering model
        X_orig : DataFrame
            Original data with features
        X_pca : array-like, optional
            PCA-transformed data used for clustering
        cpv_column : str, optional
            Name of the CPV code column
        """
        self.clusterer = clusterer
        self.X_orig = X_orig
        self.X_pca = X_pca
        self.cpv_column = cpv_column

        # Create a DataFrame with cluster assignments
        self.X_with_clusters = X_orig.copy()
        self.X_with_clusters['cluster'] = clusterer.labels_

        # Get profiles if not already created
        if clusterer.profiles_df is None:
            self.profiles = clusterer.create_cluster_profiles(X_orig, X_pca)
        else:
            self.profiles = clusterer.profiles_df

    def get_largest_clusters(self, top_n=10):
        """
        Get the IDs of the largest clusters.

        Parameters:
        -----------
        top_n : int, default=10
            Number of top clusters to return

        Returns:
        --------
        list
            IDs of the largest clusters
        """
        cluster_sizes = pd.Series(self.clusterer.labels_).value_counts()
        cluster_sizes = cluster_sizes[cluster_sizes.index != -1]  # Remove noise points
        return cluster_sizes.nlargest(top_n).index.tolist()

    def analyze_cluster(self, cluster_id):
        """
        Get detailed analysis of a specific cluster.

        Parameters:
        -----------
        cluster_id : int
            ID of the cluster to analyze

        Returns:
        --------
        dict
            Detailed cluster analysis
        """
        if cluster_id == -1:
            return {"error": "Cannot analyze noise points as a cluster"}

        # Get data for this cluster
        cluster_data = self.X_with_clusters[self.X_with_clusters['cluster'] == cluster_id]

        if len(cluster_data) == 0:
            return {"error": f"Cluster {cluster_id} is empty or does not exist"}

        # Get basic metrics
        analysis = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'pct_total': len(cluster_data) / len(self.X_orig) * 100
        }

        # Get CPV distribution if available
        if self.cpv_column and self.cpv_column in cluster_data.columns:
            cpv_counts = cluster_data[self.cpv_column].value_counts().head(5)
            cpv_pcts = (cpv_counts / len(cluster_data) * 100).round(1)

            analysis['cpv_distribution'] = {
                cpv: {
                    'count': count,
                    'percentage': cpv_pcts[cpv]
                } for cpv, count in cpv_counts.items()
            }

        # Get numeric feature statistics
        numeric_cols = ['montant', 'dureeMois', 'offresRecues']
        numeric_stats = {}

        for col in numeric_cols:
            if col in cluster_data.columns:
                numeric_stats[col] = {
                    'mean': cluster_data[col].mean(),
                    'median': cluster_data[col].median(),
                    'std': cluster_data[col].std(),
                    'min': cluster_data[col].min(),
                    'max': cluster_data[col].max()
                }

        analysis['numeric_stats'] = numeric_stats

        # Get categorical feature distributions
        cat_cols = ['procedure', 'nature', 'formePrix']
        cat_distributions = {}

        for col in cat_cols:
            if col in cluster_data.columns:
                counts = cluster_data[col].value_counts().head(3)
                pcts = (counts / len(cluster_data) * 100).round(1)

                cat_distributions[col] = {
                    cat: {
                        'count': count,
                        'percentage': pcts[cat]
                    } for cat, count in counts.items()
                }

        analysis['categorical_distributions'] = cat_distributions

        return analysis

    def plot_cluster_sizes(self, figsize=(14, 6)):
        """
        Plot histogram of cluster sizes.

        Parameters:
        -----------
        figsize : tuple, default=(14, 6)
            Figure size

        Returns:
        --------
        matplotlib figure
            Histogram of cluster sizes
        """
        plt.figure(figsize=figsize)

        cluster_sizes = pd.Series(self.clusterer.labels_).value_counts()
        cluster_sizes = cluster_sizes[cluster_sizes.index != -1]  # Remove noise

        plt.hist(cluster_sizes.values, bins=30)
        plt.xlabel('Cluster Size (number of samples)')
        plt.ylabel('Count')
        plt.title('Distribution of Cluster Sizes')

        # Add reference lines
        plt.axvline(
            x=self.clusterer.min_cluster_size,
            color='r',
            linestyle='--',
            label='min_cluster_size'
        )
        plt.axvline(
            x=cluster_sizes.median(),
            color='g',
            linestyle='--',
            label='Median size'
        )

        plt.legend()
        plt.tight_layout()

        return plt.gcf()

    def plot_cpv_heatmap(self, top_n=10, figsize=(15, 8)):
        """
        Create a heatmap of CPV codes across top clusters.

        Parameters:
        -----------
        top_n : int, default=10
            Number of top clusters to include
        figsize : tuple, default=(15, 8)
            Figure size

        Returns:
        --------
        matplotlib figure
            Heatmap of CPV distribution
        """
        if not self.cpv_column or self.cpv_column not in self.X_with_clusters.columns:
            raise ValueError("CPV column not available")

        # Get top clusters
        top_clusters = self.get_largest_clusters(top_n)

        # Create cross-tabulation
        cpv_cluster_cross = pd.crosstab(
            self.X_with_clusters['cluster'],
            self.X_with_clusters[self.cpv_column],
            normalize='index'
        )

        # Plot as heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(cpv_cluster_cross.loc[top_clusters], cmap='YlGnBu')
        plt.title('CPV Distribution in Top Clusters')
        plt.tight_layout()

        return plt.gcf()

    def plot_cluster_profiles(self, figsize=(14, 8)):
        """
        Create a scatter plot of clusters by amount and duration.

        Parameters:
        -----------
        figsize : tuple, default=(14, 8)
            Figure size

        Returns:
        --------
        matplotlib figure
            Scatter plot of clusters
        """
        plt.figure(figsize=figsize)

        # Check if required columns exist
        if 'median_amount' not in self.profiles.columns or 'median_duration' not in self.profiles.columns:
            raise ValueError("Profiles missing required columns (median_amount, median_duration)")

        # Create scatter plot
        scatter = plt.scatter(
            self.profiles['median_amount'],
            self.profiles['median_duration'],
            s=self.profiles['size']/30,  # Size proportional to cluster size
            c=self.profiles['top_cpv'].astype('category').cat.codes,  # Color by CPV
            alpha=0.7
        )

        plt.xscale('log')
        plt.xlabel('Median Amount (€)')
        plt.ylabel('Median Duration (months)')
        plt.title('Clusters by Amount, Duration and CPV')
        plt.colorbar(scatter, label='Top CPV Code')
        plt.tight_layout()

        return plt.gcf()

    def identify_distinguishing_features(self, cluster_id, n_features=5):
        """
        Identify features that distinguish a cluster from others.

        Parameters:
        -----------
        cluster_id : int
            ID of the cluster to analyze
        n_features : int, default=5
            Number of top features to return

        Returns:
        --------
        dict
            Top distinguishing features and their importance
        """
        if self.X_pca is None:
            raise ValueError("PCA-transformed data not provided")

        # Create binary target: 1 for this cluster, 0 for others
        y_binary = (self.clusterer.labels_ == cluster_id).astype(int)

        # Train a simple model to distinguish this cluster
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

        # Sample for large datasets
        sample_size = min(50000, self.X_pca.shape[0])
        indices = np.random.choice(self.X_pca.shape[0], size=sample_size, replace=False)

        rf.fit(self.X_pca[indices], y_binary[indices])

        # Get feature importance
        importances = rf.feature_importances_
        top_indices = importances.argsort()[-n_features:][::-1]

        # Create result dictionary
        result = {
            'cluster_id': cluster_id,
            'top_features': [
                {
                    'component': int(idx),
                    'importance': float(importances[idx])
                }
                for idx in top_indices
            ]
        }

        return result
