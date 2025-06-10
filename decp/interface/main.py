import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from clustering.hdbscan_model import HDBSCANClusterer
from clustering.pca_umap import DimensionalityReducer
from clustering.cluster_analysis import ClusterAnalyzer
from utils.profile_utils import create_cluster_profiles, find_similar_contracts

def load_data(filepath):
    """
    Load and prepare procurement data for clustering.

    Parameters:
    -----------
    filepath : str
        Path to the CSV data file

    Returns:
    --------
    DataFrame
        Cleaned and prepared data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} records")
    print(f"Data columns: {df.columns.tolist()}")

    return df

def preprocess_data(df, numeric_cols=None, categorical_cols=None):
    """
    Preprocess data for clustering.

    Parameters:
    -----------
    df : DataFrame
        Raw data
    numeric_cols : list, optional
        List of numeric columns to scale
    categorical_cols : list, optional
        List of categorical columns to encode

    Returns:
    --------
    tuple
        (X_preprocessed, feature_names)
    """
    print("Preprocessing data...")

    # Default columns if not specified
    if numeric_cols is None:
        numeric_cols = ['montant', 'dureeMois']

    if categorical_cols is None:
        categorical_cols = ['procedure', 'nature', 'formePrix']

    # Make a copy of input data
    data = df.copy()

    # Handle missing values in numeric columns
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())

    # Handle categorical columns
    X_cat = pd.get_dummies(data[categorical_cols], drop_first=True)

    # Scale numeric columns
    scaler = StandardScaler()
    X_num = scaler.fit_transform(data[numeric_cols])
    X_num_df = pd.DataFrame(X_num, columns=numeric_cols)

    # Combine numeric and categorical features
    X = pd.concat([X_num_df, X_cat], axis=1)

    print(f"Preprocessed data shape: {X.shape}")

    return X, X.columns.tolist()

def reduce_dimensions_and_cluster(X, random_state=42):
    """
    Reduce dimensions with PCA and cluster with HDBSCAN.

    Parameters:
    -----------
    X : DataFrame or array
        Preprocessed features
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (reducer, X_pca, X_umap, clusterer, labels)
    """
    print("Reducing dimensions and clustering...")

    # Reduce dimensions with PCA
    reducer = DimensionalityReducer(
        n_components_pca=0.9,
        n_components_umap=2,
        random_state=random_state
    )

    X_pca = reducer.fit_transform_pca(X)

    print(f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")

    # Further reduce with UMAP for visualization
    X_umap = reducer.fit_transform_umap(X_pca)

    # Create and fit HDBSCAN model
    clusterer = HDBSCANClusterer(
        min_cluster_size=55,
        min_samples=20,
        cluster_selection_epsilon=0.3,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"HDBSCAN found {n_clusters} clusters")
    print(f"Noise points: {n_noise} ({n_noise / len(labels):.2%} of data)")

    return reducer, X_pca, X_umap, clusterer, labels

def analyze_clusters(df, clusterer, X, X_pca, X_umap, cpv_column=None):
    """
    Analyze clusters and create visualizations.

    Parameters:
    -----------
    df : DataFrame
        Original data
    clusterer : HDBSCANClusterer
        Fitted clustering model
    X : DataFrame or array
        Preprocessed features
    X_pca : array
        PCA-reduced data
    X_umap : array
        UMAP projection
    cpv_column : str, optional
        Name of CPV code column

    Returns:
    --------
    tuple
        (analyzer, profiles)
    """
    print("Analyzing clusters...")

    # Create cluster analyzer
    analyzer = ClusterAnalyzer(clusterer, df, X_pca, cpv_column)

    # Get profiles
    profiles = clusterer.create_cluster_profiles(df, X_pca)
    print(f"Created profiles for {len(profiles)} clusters")

    # Plot UMAP projection with cluster labels
    plt.figure(figsize=(10, 8))
    plt.title("UMAP projection of clusters")

    # Plot noise points
    noise_mask = clusterer.labels_ == -1
    plt.scatter(
        X_umap[noise_mask, 0],
        X_umap[noise_mask, 1],
        s=2,
        c='lightgray',
        alpha=0.5,
        label='Noise'
    )

    # Plot clusters with different colors
    for label in sorted(set(clusterer.labels_)):
        if label == -1:
            continue  # Skip noise

        mask = clusterer.labels_ == label
        plt.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            s=3,
            alpha=0.7,
            label=f"Cluster {label}"
        )

    plt.legend(markerscale=2)
    plt.savefig("cluster_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved cluster visualization to 'cluster_visualization.png'")

    return analyzer, profiles

def save_models(reducer, clusterer, output_dir="models"):
    """
    Save models for later use in production.

    Parameters:
    -----------
    reducer : DimensionalityReducer
        Fitted reducer
    clusterer : HDBSCANClusterer
        Fitted clusterer
    output_dir : str, default="models"
        Directory to save models
    """
    print(f"Saving models to {output_dir}...")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save PCA reducer
    with open(os.path.join(output_dir, "pca_reducer.pkl"), "wb") as f:
        pickle.dump(reducer.pca, f)

    # Save UMAP reducer
    with open(os.path.join(output_dir, "umap_reducer.pkl"), "wb") as f:
        pickle.dump(reducer.umap, f)

    # Save HDBSCAN clusterer
    with open(os.path.join(output_dir, "hdbscan_clusterer.pkl"), "wb") as f:
        pickle.dump(clusterer.clusterer, f)

    print("Models saved successfully.")

def predict_new_contract(new_data, reducer, clusterer, profiles):
    """
    Predict cluster for new contract data.

    Parameters:
    -----------
    new_data : dict
        New contract data
    reducer : DimensionalityReducer
        Fitted reducer
    clusterer : HDBSCANClusterer
        Fitted clusterer
    profiles : DataFrame
        Cluster profiles

    Returns:
    --------
    dict
        Prediction results
    """
    print("Predicting cluster for new contract...")

    # Transform data through PCA
    X_new_pca = reducer.transform_pca(new_data.reshape(1, -1))

    # Predict cluster
    cluster_id, probability = clusterer.predict(X_new_pca)

    # Find similar contracts
    similar_clusters = find_similar_contracts({
        'amount': new_data[0],  # Assuming first feature is amount
        'duration': new_data[1]  # Assuming second feature is duration
    }, profiles, top_n=3)

    result = {
        'predicted_cluster': int(cluster_id[0]),
        'probability': float(probability[0]),
        'similar_clusters': similar_clusters.to_dict(orient='records')
    }

    return result

def main():
    """Main execution function."""
    print("DECP Clustering Pipeline")
    print("-----------------------")

    # Load data
    data_path = "data/decp_sample.csv"  # Update with your data path
    df = load_data(data_path)

    # Define feature columns
    numeric_cols = ['montant', 'dureeMois', 'offresRecues']
    categorical_cols = ['procedure', 'nature', 'formePrix']
    cpv_column = 'codeCPV'  # Update with your CPV column

    # Preprocess data
    X, feature_names = preprocess_data(df, numeric_cols, categorical_cols)

    # Reduce dimensions and cluster
    reducer, X_pca, X_umap, clusterer, labels = reduce_dimensions_and_cluster(X)

    # Analyze clusters
    analyzer, profiles = analyze_clusters(df, clusterer, X, X_pca, X_umap, cpv_column)

    # Print some key statistics
    print("\nCluster Statistics")
    print("-----------------")
    metrics = clusterer.evaluate()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save models
    save_models(reducer, clusterer)

    # Example: Predict for a new contract
    print("\nPrediction Example")
    print("-----------------")
    # Sample new contract data (amount, duration, other features...)
    new_contract = np.array([100000, 12, 3] + [0] * (X.shape[1] - 3))

    prediction = predict_new_contract(new_contract, reducer, clusterer, profiles)
    print(f"Predicted cluster: {prediction['predicted_cluster']}")
    print(f"Probability: {prediction['probability']:.4f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
