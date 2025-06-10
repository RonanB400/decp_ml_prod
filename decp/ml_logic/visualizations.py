import pandas as pd
import numpy as np

def create_cluster_profiles(X_orig, cluster_labels, cpv_column=None):
    """
    Create profiles for each cluster based on original data features.

    Parameters:
    -----------
    X_orig : DataFrame
        Original dataset with features
    cluster_labels : array-like
        Cluster assignments for each data point
    cpv_column : str, optional
        Name of the CPV code column

    Returns:
    --------
    DataFrame
        Profiles for each cluster
    """
    # Add cluster labels to original data
    X_with_clusters = X_orig.copy()
    X_with_clusters['cluster'] = cluster_labels

    profiles = []

    for cluster_id in np.sort(np.unique(cluster_labels)):
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
        if cpv_column and cpv_column in cluster_data.columns:
            profile.update({
                'top_cpv': cluster_data[cpv_column].value_counts().index[0],
                'top_cpv_pct': round(cluster_data[cpv_column].value_counts().iloc[0] / len(cluster_data) * 100, 2),
                'cpv_diversity': len(cluster_data[cpv_column].unique()),
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
    profiles_df = pd.DataFrame(profiles).sort_values('size', ascending=False)
    return profiles_df

def find_similar_contracts(new_contract, profiles_df, top_n=5):
    """
    Find clusters with similar contracts to a new contract.

    Parameters:
    -----------
    new_contract : dict
        Dictionary with contract details (amount, duration, cpv)
    profiles_df : DataFrame
        Cluster profile information
    top_n : int, default=5
        Number of similar clusters to return

    Returns:
    --------
    DataFrame
        Top similar clusters
    """
    # Create a copy of profiles
    profiles = profiles_df.copy()

    # Calculate similarity scores based on available metrics
    scores = []

    for _, profile in profiles.iterrows():
        score = 0
        weight_sum = 0

        # Compare CPV codes if available
        if 'top_cpv' in profile and 'cpv' in new_contract:
            if profile['top_cpv'] == new_contract['cpv']:
                score += 3  # High weight for matching CPV
            weight_sum += 3

        # Compare contract amount if available
        if 'median_amount' in profile and 'amount' in new_contract:
            amount_ratio = min(profile['median_amount'], new_contract['amount']) / max(profile['median_amount'], new_contract['amount'])
            score += amount_ratio * 2  # Weight of 2 for amount similarity
            weight_sum += 2

        # Compare duration if available
        if 'median_duration' in profile and 'duration' in new_contract:
            if profile['median_duration'] > 0 and new_contract['duration'] > 0:
                duration_ratio = min(profile['median_duration'], new_contract['duration']) / max(profile['median_duration'], new_contract['duration'])
                score += duration_ratio * 1  # Weight of 1 for duration similarity
                weight_sum += 1

        # Normalize score
        normalized_score = score / weight_sum if weight_sum > 0 else 0
        scores.append(normalized_score)

    # Add scores to profiles
    profiles['similarity_score'] = scores

    # Return top N similar clusters
    return profiles.sort_values('similarity_score', ascending=False).head(top_n)
