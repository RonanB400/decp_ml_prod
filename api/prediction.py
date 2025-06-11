# api/prediction.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def find_similar_clusters(contract_data: Dict[str, Any],
                         profiles_df: pd.DataFrame,
                         top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Find clusters similar to the input contract.

    Args:
        contract_data: Dictionary with contract attributes
        profiles_df: DataFrame with cluster profiles
        top_n: Number of similar clusters to return

    Returns:
        List of dictionaries with similar clusters
    """
    # Extract key metrics
    contract_dict = {
        'amount': contract_data['montant'],
        'duration': contract_data['dureeMois'],
    }

    if 'codeCPV' in contract_data and contract_data['codeCPV']:
        contract_dict['cpv'] = contract_data['codeCPV']

    # Score each cluster
    scores = []

    for _, profile in profiles_df.iterrows():
        score = 0
        weight_sum = 0

        # Compare CPV codes if available
        if 'top_cpv' in profile and 'cpv' in contract_dict:
            if profile['top_cpv'] == contract_dict['cpv']:
                score += 3  # High weight for matching CPV
            weight_sum += 3

        # Compare contract amount
        if 'median_amount' in profile and 'amount' in contract_dict:
            amount_ratio = min(profile['median_amount'], contract_dict['amount']) / max(profile['median_amount'], contract_dict['amount'])
            score += amount_ratio * 2  # Weight of 2 for amount similarity
            weight_sum += 2

        # Compare duration
        if 'median_duration' in profile and 'duration' in contract_dict:
            if profile['median_duration'] > 0 and contract_dict['duration'] > 0:
                duration_ratio = min(profile['median_duration'], contract_dict['duration']) / max(profile['median_duration'], contract_dict['duration'])
                score += duration_ratio * 1  # Weight of 1 for duration similarity
                weight_sum += 1

        # Normalize score
        normalized_score = score / weight_sum if weight_sum > 0 else 0
        scores.append(normalized_score)

    # Create a copy with scores
    profiles = profiles_df.copy()
    profiles['similarity_score'] = scores

    # Return top similar clusters
    similar = profiles.sort_values('similarity_score', ascending=False).head(top_n)
    return similar.to_dict(orient="records")
