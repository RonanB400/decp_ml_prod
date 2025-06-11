## api/preprocessing.py
#import numpy as np
#from pydantic import BaseModel
#from typing import Optional, Dict, Any
#
#class ContractData(BaseModel):
#    """Contract data model (copy from fast.py)"""
#    montant: float
#    dureeMois: float
#    offresRecues: Optional[int] = 1
#    procedure: str
#    nature: str
#    formePrix: Optional[str] = "FORFAIT"
#    codeCPV: Optional[str] = None
#
#def preprocess_input(contract_data: ContractData) -> np.ndarray:
#    """
#    Preprocess input contract data to match model requirements.
#
#    Args:
#        contract_data: Contract data from API request
#
#    Returns:
#        numpy.ndarray: Feature vector ready for PCA transformation
#    """
#    # Get dictionary representation
#    data = contract_data.dict()
#
#    # Create feature vector (match order used during training)
#    numeric_features = np.array([
#        data['montant'],
#        data['dureeMois'],
#        data.get('offresRecues', 1)
#    ])
#
#    # Encode categorical features (must match training encoding)
#    categorical_features = []
#
#    # Add procedure one-hot encoding
#    if data['procedure'] == 'PROCEDURE_ADAPTEE':
#        categorical_features.extend([1, 0, 0])
#    elif data['procedure'] == 'PROCEDURE_FORMALISEE':
#        categorical_features.extend([0, 1, 0])
#    else:
#        categorical_features.extend([0, 0, 1])
#
#    # Add nature one-hot encoding
#    if data['nature'] == 'SERVICES':
#        categorical_features.extend([1, 0, 0])
#    elif data['nature'] == 'TRAVAUX':
#        categorical_features.extend([0, 1, 0])
#    else:
#        categorical_features.extend([0, 0, 1])
#
#    # Add formePrix one-hot encoding
#    if data.get('formePrix') == 'PRIX_REVISABLES':
#        categorical_features.append(1)
#    else:
#        categorical_features.append(0)
#
#    # Combine features
#    all_features = np.concatenate([numeric_features, categorical_features])
#
#    return all_features
#
