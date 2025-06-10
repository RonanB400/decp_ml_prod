import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

class DimensionalityReducer:
    """
    Dimensionality reduction for procurement data using PCA and UMAP.
    Helps prepare data for clustering and visualization.
    """

    def __init__(self, n_components_pca=0.9, n_components_umap=2, random_state=0):
        """
        Initialize PCA and UMAP models.

        Parameters:
        -----------
        n_components_pca : float or int, default=0.9
            Number of components to keep for PCA.
            If float between 0 and 1, select components that explain this fraction of variance.
        n_components_umap : int, default=2
            Number of components for UMAP (typically 2 for visualization)
        random_state : int, default=0
            Random seed for reproducibility
        """
        self.n_components_pca = n_components_pca
        self.n_components_umap = n_components_umap
        self.random_state = random_state

        self.pca = PCA(n_components=self.n_components_pca)
        self.umap = None  # Will be initialized during fit

        self.pca_components_ = None
        self.explained_variance_ratio_ = None

    def fit_transform_pca(self, X):
        """
        Fit PCA and transform data.

        Parameters:
        -----------
        X : array-like
            Input data to reduce dimensions

        Returns:
        --------
        X_pca : array
            PCA-transformed data
        """
        X_pca = self.pca.fit_transform(X)
        self.pca_components_ = self.pca.components_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        print(f"Original dimensions: {X.shape[1]}")
        print(f"Reduced dimensions after PCA: {X_pca.shape[1]}")
        print(f"Explained variance: {sum(self.pca.explained_variance_ratio_):.2f}")

        return X_pca

    def transform_pca(self, X):
        """
        Transform new data using the fitted PCA model.

        Parameters:
        -----------
        X : array-like
            Input data to reduce dimensions

        Returns:
        --------
        X_pca : array
            PCA-transformed data
        """
        if not hasattr(self.pca, 'components_'):
            raise ValueError("PCA model has not been fitted yet")

        return self.pca.transform(X)

    def fit_transform_umap(self, X, n_neighbors=15, min_dist=0.1):
        """
        Fit UMAP and transform data.

        Parameters:
        -----------
        X : array-like
            Input data to reduce dimensions (typically PCA-reduced)
        n_neighbors : int, default=15
            Size of local neighborhood for UMAP
        min_dist : float, default=0.1
            Minimum distance between points in the projection

        Returns:
        --------
        X_umap : array
            UMAP-transformed data
        """
        # Initialize UMAP
        self.umap = UMAP(
            n_neighbors=n_neighbors,
            n_components=self.n_components_umap,
            min_dist=min_dist,
            random_state=self.random_state
        )

        # Fit and transform
        X_umap = self.umap.fit_transform(X)

        print(f"Reduced dimensions after UMAP: {X_umap.shape[1]}")
        return X_umap

    def transform_umap(self, X):
        """
        Transform new data using the fitted UMAP model.

        Parameters:
        -----------
        X : array-like
            Input data to reduce dimensions (should be PCA-reduced)

        Returns:
        --------
        X_umap : array
            UMAP-transformed data
        """
        if self.umap is None:
            raise ValueError("UMAP model has not been fitted yet")

        return self.umap.transform(X)

    def plot_explained_variance(self):
        """
        Plot explained variance ratio for PCA components.

        Returns:
        --------
        matplotlib figure
            Plot of cumulative explained variance
        """
        if not hasattr(self.pca, 'explained_variance_ratio_'):
            raise ValueError("PCA has not been fitted yet")

        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o')
        plt.axhline(y=0.9, color='r', linestyle='-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)

        return plt.gcf()

    def plot_umap_projection(self, X_umap, labels=None, figsize=(12, 10)):
        """
        Plot UMAP projection of the data.

        Parameters:
        -----------
        X_umap : array
            UMAP-transformed data (2D)
        labels : array, optional
            Cluster labels for coloring points
        figsize : tuple, default=(12, 10)
            Figure size

        Returns:
        --------
        matplotlib figure
            Plot of UMAP projection
        """
        plt.figure(figsize=figsize)

        if labels is not None:
            # Color by cluster label (excluding noise points)
            unique_labels = np.unique(labels)
            noise_mask = labels == -1

            # Plot noise points in gray
            if -1 in unique_labels:
                plt.scatter(
                    X_umap[noise_mask, 0],
                    X_umap[noise_mask, 1],
                    s=5,
                    c='lightgray',
                    alpha=0.5,
                    label='Noise'
                )

            # Plot cluster points with different colors
            for label in unique_labels:
                if label != -1:
                    mask = labels == label
                    plt.scatter(
                        X_umap[mask, 0],
                        X_umap[mask, 1],
                        s=5,
                        alpha=0.8,
                        label=f'Cluster {label}'
                    )

            plt.legend(markerscale=2)
            plt.title('UMAP projection with cluster labels')
        else:
            # Simple scatter plot without labels
            plt.scatter(X_umap[:, 0], X_umap[:, 1], s=5, alpha=0.5)
            plt.title('UMAP projection of data')

        plt.tight_layout()
        return plt.gcf()
