"""
Graph Neural Networks for Anomaly Detection in Public Procurement

This module implements Graph Neural Networks using tensorflow_gnn to
analyze buyer-supplier relationships and detect potential anomalies
in public procurement data.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import logging
from typing import Dict, Tuple, List

import tensorflow as tf
import tensorflow_gnn as tfgnn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import networkx as nx

from scripts.preprocess_pipeline import create_pipeline
from scripts.data_cleaner import filter_top_cpv_categories
from scripts.synthetic_anomaly_generator import (SyntheticAnomalyGenerator,
                                                OriginalAnomalyRemover)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ProcurementGraphBuilder:
    """Build graph structures from procurement data."""
    
    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load procurement data from CSV files."""
        logger.info(f"Loading data from {data_path}")

        X = pd.read_csv(os.path.join(data_path, 'data_clean.csv'), encoding='utf-8')
        # Basic data validation
        required_columns = ['acheteur_id', 'titulaire_id', 'montant',
                            'dateNotification']
        missing_cols = [col for col in required_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return X
    
    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        logger.info("Preprocessing data...")
        
        # Fill missing values
        X = filter_top_cpv_categories(X, top_n=60, cpv_column='codeCPV_3')

        # Split data first, before removing original anomalies
        X_train, X_test = train_test_split(X, 
                                           test_size=0.2, 
                                           random_state=0, 
                                           stratify=X['codeCPV_3'], 
                                           shuffle=True)

        X_train, X_val = train_test_split(X_train, 
                                          test_size=0.2, 
                                          random_state=0, 
                                          stratify=X_train['codeCPV_3'], 
                                          shuffle=True)

        # Remove original anomalies from training and validation sets
        logger.info("Removing original anomalies from training and validation sets...")
        anomaly_remover = OriginalAnomalyRemover()

        anomaly_types = ['single_bid_competitive', 
                        'price_inflation',
                        'price_deflation',
                        #'procedure_manipulation',
                        #'suspicious_modifications',
                        'high_market_concentration',  
                        'temporal_clustering',
                        #'excessive_subcontracting',
                        #'short_contract_duration',
                        'suspicious_buyer_supplier_pairs']
        
        # Clean training set
        logger.info("Cleaning training set...")
        X_train = anomaly_remover.clean_dataset(
            X_train, 
            anomaly_types=anomaly_types,
            strict_threshold=True
        )
        
        # Clean validation set  
        logger.info("Cleaning validation set...")
        X_val = anomaly_remover.clean_dataset(
            X_val,
            anomaly_types=anomaly_types,
            strict_threshold=True
        )

        # Preprocess pipeline
        numerical_columns = ['montant', 'dureeMois', 'offresRecues']

        binary_columns = ['sousTraitanceDeclaree', 'origineFrance', 
                          'marcheInnovant', 'idAccordCadre']
        
        categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'tauxAvance_cat',
                               'codeCPV_3']
        
        nodes_columns = ['acheteur_id', 'titulaire_id']
        
        preproc_pipeline = create_pipeline(numerical_columns, 
                                           binary_columns, 
                                         categorical_columns)

        X_train_preproc = preproc_pipeline.fit_transform(X_train)
        X_train_preproc.index = X_train.index
        X_train_preproc = pd.concat([X_train_preproc, X_train[nodes_columns]], 
                                   axis=1)

        X_val_preproc = preproc_pipeline.transform(X_val)
        X_val_preproc.index = X_val.index
        X_val_preproc = pd.concat([X_val_preproc, X_val[nodes_columns]], 
                                 axis=1)

        # Generate synthetic anomalies for test set
        logger.info("Generating synthetic anomalies for test set...")
        generator = SyntheticAnomalyGenerator(random_seed=42)
        X_test_copy = X_test.copy()
        # Reset index to avoid index mismatch issues
        X_test_copy = X_test_copy.reset_index(drop=True)

        # Generate anomalies
        X_test_anomalies = generator.generate_anomalies(
            X_test_copy,
            anomaly_percentage=0.10,  # 10% anomalies
            anomaly_types=anomaly_types
        )

        X_test_preproc = preproc_pipeline.transform(X_test_anomalies)
        X_test_preproc.index = X_test_anomalies.index
        X_test_preproc = pd.concat([X_test_preproc, 
                                   X_test_anomalies[nodes_columns]], axis=1)

        # Save the data to csv files
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        os.makedirs(data_dir, exist_ok=True)

        X_train_output_path = os.path.join(data_dir, 'X_train.csv')
        X_val_output_path = os.path.join(data_dir, 'X_val.csv')
        X_test_anomalies_output_path = os.path.join(data_dir, 
                                                    'X_test_anomalies.csv')

        # Save with index=False to avoid saving row indices
        X_train.to_csv(X_train_output_path, index=True)
        X_val.to_csv(X_val_output_path, index=True)
        X_test_anomalies.to_csv(X_test_anomalies_output_path, index=True)

        return X_train_preproc, X_val_preproc, X_test_preproc, X_train, X_val, X_test_anomalies
    

    
    def create_graph(self, X_processed: pd.DataFrame,
                     X_original: pd.DataFrame = None, type: str = 'train') -> Dict:
        """Transform preprocessed procurement data into graph structure.
        
        Args:
            X_processed: Preprocessed dataframe with encoded features
            X_original: Original dataframe for metadata (optional)
        """
        logger.info("Creating graph structure from preprocessed data...")
        
        # Remove rows with NaN buyer or supplier names
        valid_mask = (X_processed['acheteur_id'].notna() & 
                     X_processed['titulaire_id'].notna())
        X_processed = X_processed[valid_mask].copy()
        
        if X_original is not None:
            X_original = X_original[valid_mask].copy()
        
        logger.info(f"Filtered to {len(X_processed)} valid contracts "
                   f"(removed {(~valid_mask).sum()} contracts with missing names)")
        
        # Create unique identifiers for buyers and suppliers
        buyers = X_processed['acheteur_id'].unique()
        suppliers = X_processed['titulaire_id'].unique()
        
        # Create node mappings
        # buyer_to_id / supplier_to_id is a dictionary that maps each buyer / supplier to a unique integer
        buyer_to_id = {buyer: i for i, buyer in enumerate(buyers)}
        supplier_to_id = {supplier: i + len(buyers)
                          for i, supplier in enumerate(suppliers)}
        
        # Combine all nodes
        all_nodes = list(buyers) + list(suppliers)
        
        # OPTIMIZATION 1: Vectorized edge creation - ONE EDGE PER CONTRACT
        logger.info("Creating edges and edge features from preprocessed data...")
        
        # Map buyer and supplier names to IDs using vectorized operations
        # buyer_ids / supplier_ids is a list of integers that correspond to the unique integer ID of each buyer / supplier
        buyer_ids = X_processed['acheteur_id'].map(buyer_to_id).values.astype(np.int32)
        supplier_ids = X_processed['titulaire_id'].map(supplier_to_id).values.astype(np.int32)
        edges = np.column_stack([buyer_ids, supplier_ids]).astype(np.int32)
        
        # Extract all feature columns (excluding entity names)
        feature_columns = [col for col in X_processed.columns
                           if col not in ['acheteur_id', 'titulaire_id']]
        
        # Create edge features from all preprocessed features
        edge_features = X_processed[feature_columns].values.astype(np.float32)
        
        # Store contract IDs for edge-level analysis
        # contract_ids
        contract_ids = X_processed.index.tolist()
        
        # OPTIMIZATION 2: Bulk computation of node features from preprocessed data
        logger.info("Computing acheteur features from preprocessed data...")

        # Pre-compute aggregations for buyers
        buyer_stats = self._compute_bulk_node_features_preprocessed(
            X_processed, feature_columns, 'acheteur_id', 'titulaire_id')
        
        logger.info("Computing titulaire features from preprocessed data...")
        # Pre-compute aggregations for suppliers  
        supplier_stats = self._compute_bulk_node_features_preprocessed(
            X_processed, feature_columns, 'titulaire_id', 'acheteur_id')
        
        # Build node features arrays
        node_features = []
        node_types = []
        
        # Buyer features
        for buyer in buyers:
            features = buyer_stats[buyer]
            node_features.append(features)
            node_types.append(0)  # Buyer
        
        # Supplier features
        for supplier in suppliers:
            features = supplier_stats[supplier]
            node_features.append(features)
            node_types.append(1)  # Supplier
        
        # Create contract data for analysis (use original data if available)
        if X_original is not None:
            contract_data = X_original[['acheteur_id', 'titulaire_id',
                                        'montant', 'codeCPV_3', 'procedure',
                                        'dureeMois']].copy()
        else:
            # Create minimal contract data from preprocessed features
            contract_data = pd.DataFrame({
                'acheteur_id': X_processed['acheteur_id'],
                'titulaire_id': X_processed['titulaire_id'],
                'montant': X_processed['other_num_pipeline__montant'],
                'codeCPV_3': 'preprocessed',  # Placeholder
                'procedure': 'preprocessed',  # Placeholder
                'dureeMois': 'preprocessed'  # Placeholder
            })
        
        graph_data = {
            'nodes': all_nodes,
            'edges': edges,
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_features': edge_features,
            'node_types': np.array(node_types, dtype=np.int32),
            'buyer_to_id': buyer_to_id,
            'supplier_to_id': supplier_to_id,
            'contract_ids': contract_ids,
            'contract_data': contract_data,
            'feature_columns': feature_columns  # Store feature column names
        }

        # Save the graph data to a pickle file
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, f'graph_data_{type}.pkl'), 'wb') as f:
            pickle.dump(graph_data, f)
        
        return graph_data
    
    def _compute_bulk_node_features_preprocessed(self, X_processed: pd.DataFrame,
                                                feature_columns: List[str],
                                                entity_col: str, 
                                                partner_col: str) -> Dict[str, List[float]]:
        """Compute node features for all entities using preprocessed data."""
        
        # Group by entity and compute basic stats on preprocessed features
        entity_groups = X_processed.groupby(entity_col)
        
        entity_features = {}
        for entity, group in entity_groups:
            # Basic contract count and partner count
            num_contracts = len(group)
            num_partners = group[partner_col].nunique()
            
            # Aggregate the preprocessed features
            feature_stats = []
            
            # Mean of all preprocessed features
            feature_means = group[feature_columns].mean().values
            feature_stats.extend(feature_means)
            
            # Standard deviation of numerical features
            numerical_features = [col for col in feature_columns 
                                if col.startswith(('other_num_pipeline__', 
                                                  'offres_recues_pipeline__'))]
            if numerical_features:
                feature_stds = group[numerical_features].std().fillna(0).values
                feature_stats.extend(feature_stds)
            
            # Add basic entity statistics
            feature_stats.extend([
                num_contracts,
                num_partners,
                num_contracts / max(num_partners, 1)  # Contracts per partner
            ])
            
            entity_features[entity] = feature_stats
            
        return entity_features
    
    def visualize_procurement_graph(self, graph_data: Dict, title: str = "Procurement Network"):
        """Create an interactive visualization of the full procurement graph.
        
        Args:
            graph_data: Dictionary containing the graph data from create_graph
            title: Title for the visualization
        """
        from pyvis.network import Network
        import webbrowser
        import os
        
        # Create a new network
        net = Network(height="900px", width="100%", bgcolor="#ffffff",
                    font_color="black", notebook=False)
        
        # Add nodes
        for i, (name, node_type) in enumerate(zip(
            graph_data['nodes'], graph_data['node_types'])):
            
            # Calculate node size and color based on available features
            node_features = graph_data['node_features'][i]
            
            # Try to get meaningful metrics from node features
            if len(node_features) >= 3:
                # Last 3 features are: num_contracts, num_partners, contracts_per_partner
                num_contracts = int(node_features[-3]) if not np.isnan(node_features[-3]) else 1
                node_size = min(50 + num_contracts * 2, 100)
                
                # Use mean of feature values for color
                if len(node_features) > 3:
                    valid_features = node_features[:-3]
                    valid_features = valid_features[~np.isnan(valid_features)]
                    feature_value = float(np.mean(valid_features)) if len(valid_features) > 0 else 0.0
                else:
                    feature_value = float(node_features[-1]) if not np.isnan(node_features[-1]) else 0.0
            else:
                # Fallback
                num_contracts = 1
                node_size = 50
                feature_value = float(node_features[0]) if len(node_features) > 0 and not np.isnan(node_features[0]) else 0.0
            
            # Normalize feature value to a color scale (blue to red)
            # Use percentile-based normalization for better color distribution
            all_feature_values = []
            for nf in graph_data['node_features']:
                if len(nf) > 3:
                    valid_features = nf[:-3]
                    valid_features = valid_features[~np.isnan(valid_features)]
                    if len(valid_features) > 0:
                        all_feature_values.append(float(np.mean(valid_features)))
                    else:
                        all_feature_values.append(0.0)
                elif len(nf) >= 1:
                    val = float(nf[-1]) if not np.isnan(nf[-1]) else 0.0
                    all_feature_values.append(val)
                else:
                    all_feature_values.append(0.0)
            
            percentile_90 = np.percentile(all_feature_values, 90) if len(all_feature_values) > 0 else 1.0
            if np.isnan(percentile_90) or percentile_90 <= 0:
                percentile_90 = 1.0
            if np.isnan(feature_value):
                feature_value = 0.0
                
            feature_ratio = min(feature_value / max(percentile_90, 1), 1.0)
            if np.isnan(feature_ratio):
                feature_ratio = 0.0
                
            color = f"rgb({int(255 * feature_ratio)}, 0, {int(255 * (1 - feature_ratio))})"
            
            # Create tooltip with available information
            num_partners = int(node_features[-2]) if len(node_features) >= 2 and not np.isnan(node_features[-2]) else 0
            tooltip = f"Type: {'Buyer' if node_type == 0 else 'Supplier'}\n"
            tooltip += f"Contracts: {num_contracts}\n"
            tooltip += f"Partners: {num_partners}\n"
            tooltip += f"Feature Value: {feature_value:.3f}"
            
            # Add node with properties
            net.add_node(
                int(i),  # Convert to Python int
                label=str(name),  # Convert to Python string
                title=tooltip,
                color=color,
                size=node_size,
                shape="diamond" if node_type == 0 else "dot"
            )
        
        # Add edges with weights based on contract amounts
        for i, edge in enumerate(graph_data['edges']):
            # Get contract amount from contract_data for edge width
            if 'contract_data' in graph_data and i < len(graph_data['contract_data']):
                try:
                    contract_amount = float(graph_data['contract_data'].iloc[i]['montant'])
                except (KeyError, IndexError, ValueError):
                    # Fallback: try to get amount from preprocessed features
                    edge_features = graph_data['edge_features'][i]
                    if 'feature_columns' in graph_data:
                        amount_col_idx = None
                        for j, col in enumerate(graph_data['feature_columns']):
                            if 'montant' in col:
                                amount_col_idx = j
                                break
                        if amount_col_idx is not None:
                            contract_amount = float(edge_features[amount_col_idx])
                        else:
                            contract_amount = 100000  # Default fallback
                    else:
                        contract_amount = 100000  # Default fallback
            else:
                contract_amount = 100000  # Default fallback
            
            # Scale edge width based on contract amount
            edge_width = min(1 + contract_amount / 1e5, 5)  # Scale but cap at 5
            
            net.add_edge(
                int(edge[0]),  # Convert to Python int
                int(edge[1]),  # Convert to Python int
                width=edge_width,
                title=f"Amount: {contract_amount:,.2f}"
            )
        
        # Configure physics layout for initial spreading, then disable for static view
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08,
                    "damping": 0.9,  // Higher damping for faster settling
                    "avoidOverlap": 1
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 300,  // Lower for faster stop
                    "updateInterval": 25
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": true,
                "navigationButtons": true
            }
        }
        """)
        net.toggle_physics(False)
        
        # Add title
        net.set_title(title)
        
        # Save and open in browser from the data folder
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, 'procurement_graph.html')
        net.save_graph(output_path)
        webbrowser.open('file://' + output_path)


class GNNAnomalyDetector:
    """Graph Neural Network for anomaly detection."""
    
    def __init__(self, hidden_dim: int = 64, output_dim: int = 32,
                 num_layers: int = 3):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.node_model = None
        self.edge_model = None
        self.graph_tensor_train = None
        self.graph_tensor_val = None
        self.graph_tensor_test = None
        self.schema = None
        
    def create_tensorflow_graph(self, graph_data: Dict,
                                node_features_scaled: np.ndarray,
                                edge_features_scaled: np.ndarray
                                ) -> tfgnn.GraphTensor:
        """Create a TensorFlow GNN graph from our data."""
        logger.info("Creating TensorFlow GNN graph...")
        
        # Create the graph tensor with single node set
        graph_tensor = tfgnn.GraphTensor.from_pieces(
            node_sets={
                "entities": tfgnn.NodeSet.from_fields(
                    features={
                        "features": tf.constant(node_features_scaled,
                                                dtype=tf.float32),
                        "node_type": tf.constant(
                            np.array(graph_data['node_types'], 
                                   dtype=np.int32),
                            dtype=tf.int32)
                    },
                    sizes=tf.constant([len(node_features_scaled)],
                                      dtype=tf.int32)
                )
            },
            edge_sets={
                "contracts": tfgnn.EdgeSet.from_fields(
                    features={
                        "features": tf.constant(edge_features_scaled,
                                                dtype=tf.float32)
                    },
                    sizes=tf.constant([len(edge_features_scaled)],
                                      dtype=tf.int32),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("entities",
                                tf.constant(
                                    graph_data['edges'][:, 0].astype(np.int32),
                                    dtype=tf.int32)),
                        target=("entities",
                                tf.constant(
                                    graph_data['edges'][:, 1].astype(np.int32),
                                    dtype=tf.int32))
                    )
                )
            }
        )
        
        return graph_tensor
    
    def build_node_model(self, node_feature_dim: int,
                         edge_feature_dim: int,
                         l2_regularization: float = 5e-4,
                         dropout_rate: float = 0.3) -> tf.keras.Model:
        """Build the GNN model for node anomaly detection."""
        logger.info("Building GNN model for node anomaly detection...")
        
        # Helper function for regularized dense layers
        def dense_with_regularization(units, activation="relu", use_bn=True):
            """Dense layer with L2 regularization, batch norm, and dropout."""
            regularizer = tf.keras.regularizers.l2(l2_regularization)
            layers = [tf.keras.layers.Dense(
                units,
                activation=None,  # Apply activation after batch norm
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer)]
            
            if use_bn:
                layers.append(tf.keras.layers.BatchNormalization())
            
            if activation:
                layers.append(tf.keras.layers.Activation(activation))
                
            if dropout_rate > 0:
                layers.append(tf.keras.layers.Dropout(dropout_rate))
                
            return tf.keras.Sequential(layers)
        
        # Create input spec for batched graphs
        input_spec = tfgnn.GraphTensorSpec.from_piece_specs(
            node_sets_spec={
                "entities": tfgnn.NodeSetSpec.from_field_specs(
                    features_spec={
                        "features": tf.TensorSpec(
                            shape=(None, node_feature_dim),
                            dtype=tf.float32),
                        "node_type": tf.TensorSpec(shape=(None,),
                                                   dtype=tf.int32)
                    },
                    sizes_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32)
                )
            },
            edge_sets_spec={
                "contracts": tfgnn.EdgeSetSpec.from_field_specs(
                    features_spec={
                        "features": tf.TensorSpec(
                            shape=(None, edge_feature_dim),
                            dtype=tf.float32)
                    },
                    sizes_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                        source_node_set="entities",
                        target_node_set="entities"
                    )
                )
            }
        )
        
        # Input layer
        input_graph = tf.keras.layers.Input(type_spec=input_spec)
        # Merge batch to components for proper processing
        graph = input_graph.merge_batch_to_components()
        
        # Initialize hidden states for both nodes and edges
        def set_initial_node_state(node_set, *, node_set_name):
            return tf.keras.layers.Dense(self.hidden_dim)(node_set["features"])
            
        def set_initial_edge_state(edge_set, *, edge_set_name):
            return tf.keras.layers.Dense(self.hidden_dim)(edge_set["features"])
            
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=set_initial_node_state,
            edge_sets_fn=set_initial_edge_state
        )(graph)
        
        # GNN message passing layers with regularization
        for i in range(self.num_layers):
            graph = tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    "entities": tfgnn.keras.layers.NodeSetUpdate(
                        {"contracts": tfgnn.keras.layers.SimpleConv(
                            sender_edge_feature=tfgnn.HIDDEN_STATE,
                            message_fn=dense_with_regularization(
                                self.hidden_dim),
                            reduce_type="sum",
                            receiver_tag=tfgnn.TARGET)},
                        tfgnn.keras.layers.NextStateFromConcat(
                            dense_with_regularization(self.hidden_dim)))}
            )(graph)
        
        # Extract final node representations
        node_features = graph.node_sets["entities"][tfgnn.HIDDEN_STATE]
        
        # Create node embeddings with regularization
        node_embeddings = dense_with_regularization(
            self.output_dim, activation="tanh")(node_features)
        node_embeddings = tf.keras.layers.Lambda(
            lambda x: x, name="node_embeddings")(node_embeddings)
        
        # Node reconstruction pathway for anomaly detection (enhanced)
        node_reconstructed = dense_with_regularization(
            self.hidden_dim * 2, activation="relu")(node_embeddings)
        node_reconstructed = dense_with_regularization(
            self.hidden_dim, activation="relu")(node_reconstructed)
        # Final reconstruction layer without dropout to preserve quality
        node_reconstructed = tf.keras.layers.Dense(
            node_feature_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
            name="node_reconstructed")(node_reconstructed)
        
        model = tf.keras.Model(
            inputs=input_graph,
            outputs={
                'node_embeddings': node_embeddings,
                'node_reconstructed': node_reconstructed
            }
        )
        
        return model

    def build_edge_model(self, node_feature_dim: int,
                         edge_feature_dim: int,
                         l2_regularization: float = 5e-4,
                         dropout_rate: float = 0.3) -> tf.keras.Model:
        """Build the GNN model for edge anomaly detection."""
        logger.info("Building GNN model for edge anomaly detection...")
        
        # Helper function for regularized dense layers
        def dense_with_regularization(units, activation="relu", use_bn=True):
            """Dense layer with L2 regularization, batch norm, and dropout."""
            regularizer = tf.keras.regularizers.l2(l2_regularization)
            layers = [tf.keras.layers.Dense(
                units,
                activation=None,  # Apply activation after batch norm
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer)]
            
            if use_bn:
                layers.append(tf.keras.layers.BatchNormalization())
            
            if activation:
                layers.append(tf.keras.layers.Activation(activation))
                
            if dropout_rate > 0:
                layers.append(tf.keras.layers.Dropout(dropout_rate))
                
            return tf.keras.Sequential(layers)
        
        # Create input spec for batched graphs
        input_spec = tfgnn.GraphTensorSpec.from_piece_specs(
            node_sets_spec={
                "entities": tfgnn.NodeSetSpec.from_field_specs(
                    features_spec={
                        "features": tf.TensorSpec(
                            shape=(None, node_feature_dim),
                            dtype=tf.float32),
                        "node_type": tf.TensorSpec(shape=(None,),
                                                   dtype=tf.int32)
                    },
                    sizes_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32)
                )
            },
            edge_sets_spec={
                "contracts": tfgnn.EdgeSetSpec.from_field_specs(
                    features_spec={
                        "features": tf.TensorSpec(
                            shape=(None, edge_feature_dim),
                            dtype=tf.float32)
                    },
                    sizes_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                        source_node_set="entities",
                        target_node_set="entities"
                    )
                )
            }
        )
        
        # Input layer
        input_graph = tf.keras.layers.Input(type_spec=input_spec)
        # Merge batch to components for proper processing
        graph = input_graph.merge_batch_to_components()
        
        # Initialize hidden states for both nodes and edges
        def set_initial_node_state(node_set, *, node_set_name):
            return tf.keras.layers.Dense(self.hidden_dim)(node_set["features"])
            
        def set_initial_edge_state(edge_set, *, edge_set_name):
            return tf.keras.layers.Dense(self.hidden_dim)(edge_set["features"])
            
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=set_initial_node_state,
            edge_sets_fn=set_initial_edge_state
        )(graph)
        
        # GNN message passing layers with regularization
        for i in range(self.num_layers):
            graph = tfgnn.keras.layers.GraphUpdate(
                edge_sets={
                    "contracts": tfgnn.keras.layers.EdgeSetUpdate(
                        next_state=tfgnn.keras.layers.NextStateFromConcat(
                            dense_with_regularization(self.hidden_dim)))},
                node_sets={
                    "entities": tfgnn.keras.layers.NodeSetUpdate(
                        {"contracts": tfgnn.keras.layers.SimpleConv(
                            sender_edge_feature=tfgnn.HIDDEN_STATE,
                            message_fn=dense_with_regularization(
                                self.hidden_dim),
                            reduce_type="sum",
                            receiver_tag=tfgnn.TARGET)},
                        tfgnn.keras.layers.NextStateFromConcat(
                            dense_with_regularization(self.hidden_dim)))}
            )(graph)
        
        # Extract final edge representations
        edge_features = graph.edge_sets["contracts"][tfgnn.HIDDEN_STATE]
        
        # Create edge embeddings with regularization
        edge_embeddings = dense_with_regularization(
            self.output_dim, activation="tanh")(edge_features)
        edge_embeddings = tf.keras.layers.Lambda(
            lambda x: x, name="edge_embeddings")(edge_embeddings)
        
        # Edge reconstruction pathway for anomaly detection
        edge_reconstructed = dense_with_regularization(
            self.hidden_dim * 2, activation="relu")(edge_embeddings)
        edge_reconstructed = dense_with_regularization(
            self.hidden_dim, activation="relu")(edge_reconstructed)
        # Final reconstruction layer without dropout to preserve quality
        edge_reconstructed = tf.keras.layers.Dense(
            edge_feature_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
            name="edge_reconstructed")(edge_reconstructed)
        
        model = tf.keras.Model(
            inputs=input_graph,
            outputs={
                'edge_embeddings': edge_embeddings,
                'edge_reconstructed': edge_reconstructed
            }
        )
        
        return model
    
    def train_node_model(self, graph_tensor: tfgnn.GraphTensor,
                         validation_graph_tensor: tfgnn.GraphTensor = None,
                         epochs: int = 50,
                         use_huber_loss: bool = False) -> Dict:
        """Train the node anomaly detection model."""
        logger.info(f"Training node GNN model for {epochs} epochs...")
        
        if self.node_model is None:
            raise ValueError("Node model must be built before training")
        
        self.graph_tensor_train = graph_tensor
        
        # Use provided validation graph or stored validation graph
        if validation_graph_tensor is None:
            validation_graph_tensor = self.graph_tensor_val
            if validation_graph_tensor is None:
                logger.warning("No validation data provided and no stored "
                              "validation graph")
        else:
            self.graph_tensor_val = validation_graph_tensor
        
        # Create training data directly from graph tensor
        node_target_features = graph_tensor.node_sets['entities']['features']
        num_nodes = tf.shape(node_target_features)[0]
        dummy_node_embeddings = tf.zeros((num_nodes, self.output_dim))
        
        # Create training targets dictionary
        train_targets = {
            'node_embeddings': dummy_node_embeddings,
            'node_reconstructed': node_target_features
        }
        
        # Create validation data if provided
        validation_data = None
        if validation_graph_tensor is not None:
            val_node_target_features = (validation_graph_tensor.node_sets
                                       ['entities']['features'])
            val_num_nodes = tf.shape(val_node_target_features)[0]
            val_dummy_node_embeddings = tf.zeros((val_num_nodes, self.output_dim))
            
            val_targets = {
                'node_embeddings': val_dummy_node_embeddings,
                'node_reconstructed': val_node_target_features
            }
            
            # Create validation tuple
            validation_data = (validation_graph_tensor, val_targets)
        
        # Create learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=epochs//3,
            decay_rate=0.8,
            staircase=True
        )
        
        # Choose loss functions
        if use_huber_loss:
            reconstruction_loss = tf.keras.losses.Huber(delta=1.0)
            embedding_loss = tf.keras.losses.MeanSquaredError()
        else:
            reconstruction_loss = tf.keras.losses.MeanSquaredError()
            embedding_loss = tf.keras.losses.MeanSquaredError()
        
        # Compile model
        self.node_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={
                'node_embeddings': embedding_loss,
                'node_reconstructed': reconstruction_loss
            },
            loss_weights={'node_embeddings': 0.1, 'node_reconstructed': 0.9}
        )
        
        # Create batched dataset for training
        train_dataset = tf.data.Dataset.from_tensors((graph_tensor, train_targets))
        train_dataset = train_dataset.repeat()
        
        # Create batched validation dataset if provided
        validation_dataset = None
        if validation_data is not None:
            validation_dataset = tf.data.Dataset.from_tensors(validation_data)
            validation_dataset = validation_dataset.repeat()
        
        # Train using model.fit with properly batched data
        history = self.node_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            steps_per_epoch=1,
            validation_steps=1 if validation_dataset is not None else None,
            epochs=epochs,
            verbose=1
        )
        
        # Save the trained model
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        os.makedirs(data_dir, exist_ok=True)
        model_path = os.path.join(data_dir, 'gnn_node_anomaly_model')
        tf.saved_model.save(self.node_model, model_path)
        logger.info(f"Node model saved to {model_path}")
        
        return history.history

    def train_edge_model(self, graph_tensor: tfgnn.GraphTensor,
                         validation_graph_tensor: tfgnn.GraphTensor = None,
                         epochs: int = 50,
                         use_huber_loss: bool = False) -> Dict:
        """Train the edge anomaly detection model."""
        logger.info(f"Training edge GNN model for {epochs} epochs...")
        
        if self.edge_model is None:
            raise ValueError("Edge model must be built before training")
        
        self.graph_tensor_train = graph_tensor
        
        # Use provided validation graph or stored validation graph
        if validation_graph_tensor is None:
            validation_graph_tensor = self.graph_tensor_val
            if validation_graph_tensor is None:
                logger.warning("No validation data provided and no stored "
                              "validation graph")
        else:
            self.graph_tensor_val = validation_graph_tensor
        
        # Create training data directly from graph tensor
        edge_target_features = graph_tensor.edge_sets['contracts']['features']
        num_edges = tf.shape(edge_target_features)[0]
        dummy_edge_embeddings = tf.zeros((num_edges, self.output_dim))
        
        # Create training targets dictionary
        train_targets = {
            'edge_embeddings': dummy_edge_embeddings,
            'edge_reconstructed': edge_target_features
        }
        
        # Create validation data if provided
        validation_data = None
        if validation_graph_tensor is not None:
            val_edge_target_features = (validation_graph_tensor.edge_sets
                                       ['contracts']['features'])
            val_num_edges = tf.shape(val_edge_target_features)[0]
            val_dummy_edge_embeddings = tf.zeros((val_num_edges, self.output_dim))
            
            val_targets = {
                'edge_embeddings': val_dummy_edge_embeddings,
                'edge_reconstructed': val_edge_target_features
            }
            
            # Create validation tuple
            validation_data = (validation_graph_tensor, val_targets)
        
        # Create learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=epochs//3,
            decay_rate=0.8,
            staircase=True
        )
        
        # Choose loss functions
        if use_huber_loss:
            reconstruction_loss = tf.keras.losses.Huber(delta=1.0)
            embedding_loss = tf.keras.losses.MeanSquaredError()
        else:
            reconstruction_loss = tf.keras.losses.MeanSquaredError()
            embedding_loss = tf.keras.losses.MeanSquaredError()
        
        # Compile model
        self.edge_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={
                'edge_embeddings': embedding_loss,
                'edge_reconstructed': reconstruction_loss
            },
            loss_weights={'edge_embeddings': 0.1, 'edge_reconstructed': 0.9}
        )
        
        # Create batched dataset for training
        train_dataset = tf.data.Dataset.from_tensors((graph_tensor, train_targets))
        train_dataset = train_dataset.repeat()
        
        # Create batched validation dataset if provided
        validation_dataset = None
        if validation_data is not None:
            validation_dataset = tf.data.Dataset.from_tensors(validation_data)
            validation_dataset = validation_dataset.repeat()
        
        # Train using model.fit with properly batched data
        history = self.edge_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            steps_per_epoch=1,
            validation_steps=1 if validation_dataset is not None else None,
            epochs=epochs,
            verbose=1
        )
        
        # Save the trained model
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        os.makedirs(data_dir, exist_ok=True)
        model_path = os.path.join(data_dir, 'gnn_edge_anomaly_model')
        tf.saved_model.save(self.edge_model, model_path)
        logger.info(f"Edge model saved to {model_path}")
        
        return history.history
    

    

    
    def plot_node_training_history(self, history: Dict, save_path: str = None):
        """Plot node model training and validation losses over epochs.
        
        Args:
            history: Node training history dictionary from model.fit()
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Node Model Training and Validation Losses', fontsize=16)
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot 1: Node Reconstruction Loss
        axes[0, 0].plot(epochs, history['node_reconstructed_loss'], 
                       'b-', label='Training', linewidth=2)
        if 'val_node_reconstructed_loss' in history:
            axes[0, 0].plot(epochs, history['val_node_reconstructed_loss'], 
                           'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Node Reconstruction Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Node Embeddings Loss
        if 'node_embeddings_loss' in history:
            axes[0, 1].plot(epochs, history['node_embeddings_loss'], 
                           'b-', label='Training', linewidth=2)
            if 'val_node_embeddings_loss' in history:
                axes[0, 1].plot(epochs, history['val_node_embeddings_loss'], 
                               'r-', label='Validation', linewidth=2)
            axes[0, 1].set_title('Node Embeddings Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Node Embeddings Loss\nNot Available', 
                           ha='center', va='center', 
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Node Embeddings Loss')
        
        # Plot 3: Total Weighted Loss
        axes[1, 0].plot(epochs, history['loss'], 
                       'b-', label='Training', linewidth=2)
        if 'val_loss' in history:
            axes[1, 0].plot(epochs, history['val_loss'], 
                           'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('Total Weighted Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss Comparison (All node losses on one plot)
        axes[1, 1].plot(epochs, history['node_reconstructed_loss'], 
                       'b-', label='Node Recon (Train)', linewidth=2)
        if 'node_embeddings_loss' in history:
            axes[1, 1].plot(epochs, history['node_embeddings_loss'], 
                           'g-', label='Node Embed (Train)', alpha=0.7)
        axes[1, 1].plot(epochs, history['loss'], 
                       'k-', label='Total (Train)', linewidth=2)
        
        if 'val_node_reconstructed_loss' in history:
            axes[1, 1].plot(epochs, history['val_node_reconstructed_loss'], 
                           'b--', label='Node Recon (Val)', linewidth=2)
        if 'val_node_embeddings_loss' in history:
            axes[1, 1].plot(epochs, history['val_node_embeddings_loss'], 
                           'g--', label='Node Embed (Val)', alpha=0.7)
        if 'val_loss' in history:
            axes[1, 1].plot(epochs, history['val_loss'], 
                           'k--', label='Total (Val)', linewidth=2)
        
        axes[1, 1].set_title('All Node Losses Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Node training history plot saved to {save_path}")
        
        plt.show()
        
        # Print final loss values
        print("\n" + "="*50)
        print("NODE MODEL - FINAL LOSS VALUES")
        print("="*50)
        print("Training Losses (Final Epoch):")
        print(f"  - Node Reconstruction: "
              f"{history['node_reconstructed_loss'][-1]:.6f}")
        if 'node_embeddings_loss' in history:
            print(f"  - Node Embeddings: "
                  f"{history['node_embeddings_loss'][-1]:.6f}")
        print(f"  - Total Weighted Loss: {history['loss'][-1]:.6f}")
        
        if 'val_loss' in history:
            print("\nValidation Losses (Final Epoch):")
            print(f"  - Node Reconstruction: "
                  f"{history['val_node_reconstructed_loss'][-1]:.6f}")
            if 'val_node_embeddings_loss' in history:
                print(f"  - Node Embeddings: "
                      f"{history['val_node_embeddings_loss'][-1]:.6f}")
            print(f"  - Total Weighted Loss: {history['val_loss'][-1]:.6f}")
            
            # Calculate improvement/overfitting indicators
            train_val_diff = history['loss'][-1] - history['val_loss'][-1]
            print("\nTraining vs Validation Analysis:")
            if train_val_diff > 0.01:
                print("  - ⚠️  Training loss >> Validation loss")
                print("      Likely causes: Dropout/regularization effects")
                print("      This is NOT overfitting - model performs better "
                      "on validation!")
            elif train_val_diff < -0.01:
                print("  - ⚠️  Potential overfitting (val loss >> train loss)")
                print("      Model memorizing training data, poor "
                      "generalization")
            else:
                print("  - ✅ Good generalization (train ≈ val loss)")

    def plot_edge_training_history(self, history: Dict, save_path: str = None):
        """Plot edge model training and validation losses over epochs.
        
        Args:
            history: Edge training history dictionary from model.fit()
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Edge Model Training and Validation Losses', fontsize=16)
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot 1: Edge Reconstruction Loss
        axes[0, 0].plot(epochs, history['edge_reconstructed_loss'], 
                       'b-', label='Training', linewidth=2)
        if 'val_edge_reconstructed_loss' in history:
            axes[0, 0].plot(epochs, history['val_edge_reconstructed_loss'], 
                           'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Edge Reconstruction Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Edge Embeddings Loss
        if 'edge_embeddings_loss' in history:
            axes[0, 1].plot(epochs, history['edge_embeddings_loss'], 
                           'b-', label='Training', linewidth=2)
            if 'val_edge_embeddings_loss' in history:
                axes[0, 1].plot(epochs, history['val_edge_embeddings_loss'], 
                               'r-', label='Validation', linewidth=2)
            axes[0, 1].set_title('Edge Embeddings Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Edge Embeddings Loss\nNot Available', 
                           ha='center', va='center', 
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Edge Embeddings Loss')
        
        # Plot 3: Total Weighted Loss
        axes[1, 0].plot(epochs, history['loss'], 
                       'b-', label='Training', linewidth=2)
        if 'val_loss' in history:
            axes[1, 0].plot(epochs, history['val_loss'], 
                           'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('Total Weighted Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss Comparison (All edge losses on one plot)
        axes[1, 1].plot(epochs, history['edge_reconstructed_loss'], 
                       'b-', label='Edge Recon (Train)', linewidth=2)
        if 'edge_embeddings_loss' in history:
            axes[1, 1].plot(epochs, history['edge_embeddings_loss'], 
                           'g-', label='Edge Embed (Train)', alpha=0.7)
        axes[1, 1].plot(epochs, history['loss'], 
                       'k-', label='Total (Train)', linewidth=2)
        
        if 'val_edge_reconstructed_loss' in history:
            axes[1, 1].plot(epochs, history['val_edge_reconstructed_loss'], 
                           'b--', label='Edge Recon (Val)', linewidth=2)
        if 'val_edge_embeddings_loss' in history:
            axes[1, 1].plot(epochs, history['val_edge_embeddings_loss'], 
                           'g--', label='Edge Embed (Val)', alpha=0.7)
        if 'val_loss' in history:
            axes[1, 1].plot(epochs, history['val_loss'], 
                           'k--', label='Total (Val)', linewidth=2)
        
        axes[1, 1].set_title('All Edge Losses Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Edge training history plot saved to {save_path}")
        
        plt.show()
        
        # Print final loss values
        print("\n" + "="*50)
        print("EDGE MODEL - FINAL LOSS VALUES")
        print("="*50)
        print("Training Losses (Final Epoch):")
        print(f"  - Edge Reconstruction: "
              f"{history['edge_reconstructed_loss'][-1]:.6f}")
        if 'edge_embeddings_loss' in history:
            print(f"  - Edge Embeddings: "
                  f"{history['edge_embeddings_loss'][-1]:.6f}")
        print(f"  - Total Weighted Loss: {history['loss'][-1]:.6f}")
        
        if 'val_loss' in history:
            print("\nValidation Losses (Final Epoch):")
            print(f"  - Edge Reconstruction: "
                  f"{history['val_edge_reconstructed_loss'][-1]:.6f}")
            if 'val_edge_embeddings_loss' in history:
                print(f"  - Edge Embeddings: "
                      f"{history['val_edge_embeddings_loss'][-1]:.6f}")
            print(f"  - Total Weighted Loss: {history['val_loss'][-1]:.6f}")
            
            # Calculate improvement/overfitting indicators
            train_val_diff = history['loss'][-1] - history['val_loss'][-1]
            print("\nTraining vs Validation Analysis:")
            if train_val_diff > 0.01:
                print("  - ⚠️  Training loss >> Validation loss")
                print("      Likely causes: Dropout/regularization effects")
                print("      This is NOT overfitting - model performs better "
                      "on validation!")
            elif train_val_diff < -0.01:
                print("  - ⚠️  Potential overfitting (val loss >> train loss)")
                print("      Model memorizing training data, poor "
                      "generalization")
            else:
                print("  - ✅ Good generalization (train ≈ val loss)")

    def plot_combined_training_history(self, node_history: Dict, 
                                      edge_history: Dict, 
                                      save_path: str = None):
        """Plot both node and edge model training histories for comparison.
        
        Args:
            node_history: Node training history dictionary
            edge_history: Edge training history dictionary
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Node and Edge Models Training Comparison', fontsize=16)
        
        node_epochs = range(1, len(node_history['loss']) + 1)
        edge_epochs = range(1, len(edge_history['loss']) + 1)
        
        # Plot 1: Node Reconstruction Loss
        axes[0, 0].plot(node_epochs, node_history['node_reconstructed_loss'], 
                       'b-', label='Training', linewidth=2)
        if 'val_node_reconstructed_loss' in node_history:
            axes[0, 0].plot(node_epochs, 
                           node_history['val_node_reconstructed_loss'], 
                           'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Node Reconstruction Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Edge Reconstruction Loss
        axes[0, 1].plot(edge_epochs, edge_history['edge_reconstructed_loss'], 
                       'b-', label='Training', linewidth=2)
        if 'val_edge_reconstructed_loss' in edge_history:
            axes[0, 1].plot(edge_epochs, 
                           edge_history['val_edge_reconstructed_loss'], 
                           'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Edge Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Total Losses Comparison
        axes[0, 2].plot(node_epochs, node_history['loss'], 
                       'b-', label='Node Model', linewidth=2)
        axes[0, 2].plot(edge_epochs, edge_history['loss'], 
                       'r-', label='Edge Model', linewidth=2)
        if 'val_loss' in node_history:
            axes[0, 2].plot(node_epochs, node_history['val_loss'], 
                           'b--', label='Node Model (Val)', linewidth=2)
        if 'val_loss' in edge_history:
            axes[0, 2].plot(edge_epochs, edge_history['val_loss'], 
                           'r--', label='Edge Model (Val)', linewidth=2)
        axes[0, 2].set_title('Total Weighted Loss Comparison')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Node Model All Losses
        axes[1, 0].plot(node_epochs, node_history['node_reconstructed_loss'], 
                       'b-', label='Reconstruction', linewidth=2)
        if 'node_embeddings_loss' in node_history:
            axes[1, 0].plot(node_epochs, node_history['node_embeddings_loss'], 
                           'g-', label='Embeddings', alpha=0.7)
        axes[1, 0].plot(node_epochs, node_history['loss'], 
                       'k-', label='Total', linewidth=2)
        axes[1, 0].set_title('Node Model - All Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Edge Model All Losses
        axes[1, 1].plot(edge_epochs, edge_history['edge_reconstructed_loss'], 
                       'b-', label='Reconstruction', linewidth=2)
        if 'edge_embeddings_loss' in edge_history:
            axes[1, 1].plot(edge_epochs, edge_history['edge_embeddings_loss'], 
                           'g-', label='Embeddings', alpha=0.7)
        axes[1, 1].plot(edge_epochs, edge_history['loss'], 
                       'k-', label='Total', linewidth=2)
        axes[1, 1].set_title('Edge Model - All Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Final Performance Comparison (Bar Chart)
        categories = ['Node\nReconstruction', 'Edge\nReconstruction', 
                     'Node\nTotal', 'Edge\nTotal']
        train_values = [
            node_history['node_reconstructed_loss'][-1],
            edge_history['edge_reconstructed_loss'][-1],
            node_history['loss'][-1],
            edge_history['loss'][-1]
        ]
        
        x_pos = range(len(categories))
        bars = axes[1, 2].bar(x_pos, train_values, alpha=0.7, 
                             color=['skyblue', 'lightcoral', 'lightgreen', 
                                   'orange'])
        axes[1, 2].set_title('Final Training Loss Comparison')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(categories)
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, train_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height(),
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Combined training history plot saved to {save_path}")
        
        plt.show()
        
        # Print comparison summary
        print("\n" + "="*60)
        print("MODELS COMPARISON - FINAL LOSS VALUES")
        print("="*60)
        print("Node Model Final Losses:")
        print(f"  - Reconstruction: "
              f"{node_history['node_reconstructed_loss'][-1]:.6f}")
        print(f"  - Total: {node_history['loss'][-1]:.6f}")
        
        print("\nEdge Model Final Losses:")
        print(f"  - Reconstruction: "
              f"{edge_history['edge_reconstructed_loss'][-1]:.6f}")
        print(f"  - Total: {edge_history['loss'][-1]:.6f}")
        
        # Determine which model performed better
        node_final = node_history['loss'][-1]
        edge_final = edge_history['loss'][-1]
        
        print("\nModel Performance Comparison:")
        if node_final < edge_final:
            diff = ((edge_final - node_final) / edge_final) * 100
            print(f"  - 🏆 Node model performed better "
                  f"({diff:.1f}% lower loss)")
        elif edge_final < node_final:
            diff = ((node_final - edge_final) / node_final) * 100
            print(f"  - 🏆 Edge model performed better "
                  f"({diff:.1f}% lower loss)")
        else:
            print("  - 🤝 Both models achieved similar performance")



    def detect_node_anomalies(self, graph_tensor: tfgnn.GraphTensor = None,
                              threshold_percentile: float = 10
                              ) -> Tuple[np.ndarray, float]:
        """Detect node anomalies based on reconstruction error.
        
        Args:
            graph_tensor: Graph tensor to analyze (uses test tensor if None)
            threshold_percentile: Percentage of nodes to flag as anomalies 
                                (e.g., 10 means top 10% will be anomalies)
        """
        logger.info("Detecting node anomalies...")
        
        if self.node_model is None:
            raise ValueError("Node model must be trained before detecting "
                             "anomalies")
        
        # Use provided graph_tensor or default to test tensor
        if graph_tensor is None:
            if self.graph_tensor_test is None:
                raise ValueError("No graph tensor provided and no test "
                               "tensor available")
            graph_tensor = self.graph_tensor_test
        
        # Get predictions by creating a dataset and batching properly
        def data_generator():
            yield graph_tensor
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=graph_tensor.spec
        )
        dataset = dataset.batch(1)
        
        predictions = self.node_model.predict(dataset)
        node_reconstructed = predictions['node_reconstructed']
        
        # Calculate reconstruction errors
        original_node_features = (graph_tensor.node_sets['entities']
                                ['features'].numpy())
        
        node_reconstruction_error = np.mean((original_node_features - 
                                           node_reconstructed) ** 2, axis=1)
        
        # Determine thresholds and anomalies
        # Convert anomaly percentage to percentile threshold
        actual_percentile = 100 - threshold_percentile
        node_threshold = np.percentile(node_reconstruction_error, 
                                     actual_percentile)
        node_anomalies = node_reconstruction_error > node_threshold
        
        logger.info(f"Detected {np.sum(node_anomalies)} node anomalies "
                   f"({np.sum(node_anomalies)/len(node_anomalies)*100:.1f}%)")
        
        return node_reconstruction_error, node_threshold

    def detect_edge_anomalies(self, graph_tensor: tfgnn.GraphTensor = None,
                              threshold_percentile: float = 10
                              ) -> Tuple[np.ndarray, float]:
        """Detect edge anomalies based on reconstruction error.
        
        Args:
            graph_tensor: Graph tensor to analyze (uses test tensor if None)
            threshold_percentile: Percentage of edges to flag as anomalies 
                                (e.g., 10 means top 10% will be anomalies)
        """
        logger.info("Detecting edge anomalies...")
        
        if self.edge_model is None:
            raise ValueError("Edge model must be trained before detecting "
                             "anomalies")
        
        # Use provided graph_tensor or default to test tensor
        if graph_tensor is None:
            if self.graph_tensor_test is None:
                raise ValueError("No graph tensor provided and no test "
                               "tensor available")
            graph_tensor = self.graph_tensor_test
        
        # DEBUG: Check graph tensor edge features
        original_edge_features = (graph_tensor.edge_sets['contracts']
                                ['features'].numpy())
        logger.info(f"Original edge features shape: {original_edge_features.shape}")
        logger.info(f"Original edge features - NaN count: {np.sum(np.isnan(original_edge_features))}")
        logger.info(f"Original edge features - Inf count: {np.sum(np.isinf(original_edge_features))}")
        logger.info(f"Original edge features stats: min={np.nanmin(original_edge_features):.6f}, "
                   f"max={np.nanmax(original_edge_features):.6f}, "
                   f"mean={np.nanmean(original_edge_features):.6f}")
        
        # Get predictions by creating a dataset and batching properly
        def data_generator():
            yield graph_tensor
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=graph_tensor.spec
        )
        dataset = dataset.batch(1)
        
        predictions = self.edge_model.predict(dataset)
        edge_reconstructed = predictions['edge_reconstructed']
        
        # DEBUG: Check reconstructed features
        logger.info(f"Reconstructed edge features shape: {edge_reconstructed.shape}")
        logger.info(f"Reconstructed edge features - NaN count: {np.sum(np.isnan(edge_reconstructed))}")
        logger.info(f"Reconstructed edge features - Inf count: {np.sum(np.isinf(edge_reconstructed))}")
        logger.info(f"Reconstructed edge features stats: min={np.nanmin(edge_reconstructed):.6f}, "
                   f"max={np.nanmax(edge_reconstructed):.6f}, "
                   f"mean={np.nanmean(edge_reconstructed):.6f}")
        
        # DEBUG: Check if shapes match
        if original_edge_features.shape != edge_reconstructed.shape:
            logger.error(f"Shape mismatch! Original: {original_edge_features.shape}, "
                        f"Reconstructed: {edge_reconstructed.shape}")
            # Try to reshape if possible
            if original_edge_features.size == edge_reconstructed.size:
                logger.info("Attempting to reshape to match...")
                edge_reconstructed = edge_reconstructed.reshape(original_edge_features.shape)
            else:
                raise ValueError("Cannot resolve shape mismatch between original and reconstructed features")
        
        # Calculate reconstruction errors
        diff = original_edge_features - edge_reconstructed
        logger.info(f"Difference - NaN count: {np.sum(np.isnan(diff))}")
        logger.info(f"Difference - Inf count: {np.sum(np.isinf(diff))}")
        
        # Calculate squared differences
        squared_diff = diff ** 2
        logger.info(f"Squared difference - NaN count: {np.sum(np.isnan(squared_diff))}")
        logger.info(f"Squared difference - Inf count: {np.sum(np.isinf(squared_diff))}")
        
        # Calculate mean across features (axis=1)
        edge_reconstruction_error = np.mean(squared_diff, axis=1)
        
        # DEBUG: Check reconstruction errors
        logger.info(f"Reconstruction errors shape: {edge_reconstruction_error.shape}")
        logger.info(f"Reconstruction errors - NaN count: {np.sum(np.isnan(edge_reconstruction_error))}")
        logger.info(f"Reconstruction errors - Inf count: {np.sum(np.isinf(edge_reconstruction_error))}")
        logger.info(f"Reconstruction errors - % NaN: {np.sum(np.isnan(edge_reconstruction_error))/len(edge_reconstruction_error)*100:.1f}%")
        
        # Filter out NaN values for threshold calculation
        valid_errors = edge_reconstruction_error[~np.isnan(edge_reconstruction_error)]
        logger.info(f"Valid errors count: {len(valid_errors)} out of {len(edge_reconstruction_error)}")
        
        if len(valid_errors) == 0:
            logger.error("All reconstruction errors are NaN! Cannot determine threshold.")
            return edge_reconstruction_error, 0.0
        
        if len(valid_errors) < len(edge_reconstruction_error):
            logger.warning(f"Found {len(edge_reconstruction_error) - len(valid_errors)} NaN errors, using only valid errors for threshold")
        
        # Determine thresholds and anomalies using valid errors
        # Convert anomaly percentage to percentile threshold
        actual_percentile = 100 - threshold_percentile
        edge_threshold = np.percentile(valid_errors, actual_percentile)
        logger.info(f"Edge threshold (from valid errors): {edge_threshold:.6f}")
        
        # For anomaly detection, treat NaN as non-anomalous (conservative approach)
        edge_anomalies = np.where(np.isnan(edge_reconstruction_error), 
                                 False, 
                                 edge_reconstruction_error > edge_threshold)
        
        logger.info(f"Detected {np.sum(edge_anomalies)} edge anomalies "
                   f"({np.sum(edge_anomalies)/len(edge_anomalies)*100:.1f}%)")
        
        return edge_reconstruction_error, edge_threshold

    def diagnose_edge_model(self, graph_tensor: tfgnn.GraphTensor = None) -> Dict:
        """Diagnose potential issues with the edge model and features.
        
        Returns:
            Dictionary with diagnostic information
        """
        logger.info("Starting edge model diagnostics...")
        
        if graph_tensor is None:
            graph_tensor = self.graph_tensor_test
            
        if graph_tensor is None:
            return {"error": "No graph tensor available for diagnosis"}
        
        diagnostics = {}
        
        # Check graph tensor structure
        edge_features = graph_tensor.edge_sets['contracts']['features'].numpy()
        diagnostics['edge_features_shape'] = edge_features.shape
        diagnostics['edge_features_nan_count'] = int(np.sum(np.isnan(edge_features)))
        diagnostics['edge_features_inf_count'] = int(np.sum(np.isinf(edge_features)))
        diagnostics['edge_features_finite_count'] = int(np.sum(np.isfinite(edge_features)))
        
        # Check if model exists and is compiled
        if self.edge_model is None:
            diagnostics['edge_model_status'] = "Not built"
            return diagnostics
        
        diagnostics['edge_model_status'] = "Built"
        diagnostics['edge_model_trainable_params'] = self.edge_model.count_params()
        
        # Try a simple prediction
        try:
            def data_generator():
                yield graph_tensor
            
            dataset = tf.data.Dataset.from_generator(
                data_generator, output_signature=graph_tensor.spec
            )
            dataset = dataset.batch(1)
            
            predictions = self.edge_model.predict(dataset, verbose=0)
            edge_reconstructed = predictions['edge_reconstructed']
            
            diagnostics['prediction_shape'] = edge_reconstructed.shape
            diagnostics['prediction_nan_count'] = int(np.sum(np.isnan(edge_reconstructed)))
            diagnostics['prediction_inf_count'] = int(np.sum(np.isinf(edge_reconstructed)))
            diagnostics['prediction_finite_count'] = int(np.sum(np.isfinite(edge_reconstructed)))
            
            if np.all(np.isfinite(edge_reconstructed)):
                diagnostics['prediction_min'] = float(np.min(edge_reconstructed))
                diagnostics['prediction_max'] = float(np.max(edge_reconstructed))
                diagnostics['prediction_mean'] = float(np.mean(edge_reconstructed))
                diagnostics['prediction_std'] = float(np.std(edge_reconstructed))
            
        except Exception as e:
            diagnostics['prediction_error'] = str(e)
        
        # Check feature scaling
        if hasattr(self, 'edge_scaler') and hasattr(self.edge_scaler, 'scale_'):
            diagnostics['edge_scaler_scale_shape'] = self.edge_scaler.scale_.shape
            diagnostics['edge_scaler_scale_nan_count'] = int(np.sum(np.isnan(self.edge_scaler.scale_)))
            diagnostics['edge_scaler_mean_shape'] = self.edge_scaler.mean_.shape
            diagnostics['edge_scaler_mean_nan_count'] = int(np.sum(np.isnan(self.edge_scaler.mean_)))
        
        logger.info("Edge model diagnostics completed")
        return diagnostics


class AnomalyAnalyzer:
    """Analyze and visualize anomaly detection results."""
    
    def __init__(self):
        pass
    
    def create_node_results_dataframe(self, graph_data: Dict,
                                 node_reconstruction_error: np.ndarray,
                                 node_anomalies: np.ndarray) -> pd.DataFrame:
        """Create a comprehensive results DataFrame for nodes."""
        # Create basic results dataframe
        results = {
            'entity_id': graph_data['nodes'],
            'entity_type': ['Buyer' if t == 0 else 'Supplier'
                           for t in graph_data['node_types']],
            'node_reconstruction_error': node_reconstruction_error,
            'is_node_anomaly': node_anomalies
        }
        
        # Add node features if available (handle variable structure)
        node_features = graph_data['node_features']
        if node_features.shape[1] >= 3:
            # Extract basic statistics from the end of feature vector
            # These are added by the preprocessing function
            results.update({
                'num_contracts': node_features[:, -3],
                'num_partners': node_features[:, -2], 
                'contracts_per_partner': node_features[:, -1]
            })
            
            # If we have more features, add some aggregated measures
            if node_features.shape[1] > 3:
                results.update({
                    'mean_feature_value': np.mean(node_features[:, :-3], axis=1),
                    'std_feature_value': np.std(node_features[:, :-3], axis=1)
                })
        
        return pd.DataFrame(results).sort_values('node_reconstruction_error', 
                                                ascending=False)
    
    def create_edge_results_dataframe(self, graph_data: Dict,
                                     edge_reconstruction_error: np.ndarray,
                                     edge_anomalies: np.ndarray) -> pd.DataFrame:
        """Create a comprehensive results DataFrame for edges (contracts)."""
        contract_data = graph_data['contract_data']
        
        # Base data dictionary with required fields
        results_dict = {
            'contract_id': graph_data['contract_ids'],
            'edge_reconstruction_error': edge_reconstruction_error,
            'is_edge_anomaly': edge_anomalies
        }
        
        # Add required contract data fields with safe access
        required_fields = ['acheteur_id', 'titulaire_id', 'montant']
        
        for contract_col in required_fields:
            if contract_col in contract_data.columns:
                results_dict[contract_col] = contract_data[contract_col].values
            else:
                # Fill with placeholder if required field is missing
                results_dict[contract_col] = ['Unknown'] * len(edge_reconstruction_error)
                logger.warning(f"Required column '{contract_col}' not found in contract_data")
        
        # Add optional contract data fields with safe access
        optional_fields = ['codeCPV_3', 'procedure', 'dateNotification']
        
        for contract_col in optional_fields:
            if contract_col in contract_data.columns:
                results_dict[contract_col] = contract_data[contract_col].values
            else:
                # Fill with None/placeholder if optional field is missing
                results_dict[contract_col] = [None] * len(edge_reconstruction_error)
                logger.info(f"Optional column '{contract_col}' not found in contract_data, using None")
        
        # Add edge features with safe indexing
        edge_features = graph_data['edge_features']
        edge_feature_names = ['log_amount', 'cpv_hash', 'procedure_hash', 'duration_months']
        
        for i, feature_name in enumerate(edge_feature_names):
            if i < edge_features.shape[1]:
                results_dict[feature_name] = edge_features[:, i]
            else:
                results_dict[feature_name] = [None] * len(edge_reconstruction_error)
                logger.warning(f"Edge feature column {i} ('{feature_name}') not available in edge_features")
        
        return pd.DataFrame(results_dict).sort_values('edge_reconstruction_error', ascending=False)
    
    def plot_results(self, results_df: pd.DataFrame,
                    node_reconstruction_error: np.ndarray,
                    edge_reconstruction_error: np.ndarray,
                    node_threshold: float, edge_threshold: float):
        """Create comprehensive visualization of results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Node reconstruction error distribution
        axes[0, 0].hist(node_reconstruction_error, bins=50, alpha=0.7,
                       color='skyblue')
        axes[0, 0].axvline(node_threshold, color='red', linestyle='--',
                          label=f'Threshold ({node_threshold:.4f})')
        axes[0, 0].set_xlabel('Node Reconstruction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Node Reconstruction Errors')
        axes[0, 0].legend()
        
        # Plot 2: Edge reconstruction error distribution
        axes[0, 1].hist(edge_reconstruction_error, bins=50, alpha=0.7,
                       color='skyblue')
        axes[0, 1].axvline(edge_threshold, color='red', linestyle='--',
                          label=f'Threshold ({edge_threshold:.4f})')
        axes[0, 1].set_xlabel('Edge Reconstruction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Edge Reconstruction Errors')
        axes[0, 1].legend()
        
        # Plot 3: Error by entity type
        buyer_errors = node_reconstruction_error[
            results_df['entity_type'] == 'Buyer']
        supplier_errors = node_reconstruction_error[
            results_df['entity_type'] == 'Supplier']
        axes[0, 2].boxplot([buyer_errors, supplier_errors],
                          labels=['Buyers', 'Suppliers'])
        axes[0, 2].set_ylabel('Node Reconstruction Error')
        axes[0, 2].set_title('Node Reconstruction Error by Entity Type')
        
        # Plot 4: Scatter plot - contracts vs feature value
        normal_mask = ~results_df['is_node_anomaly']
        if 'num_contracts' in results_df.columns and 'mean_feature_value' in results_df.columns:
            y_col = 'mean_feature_value'
            y_label = 'Mean Feature Value'
        elif 'num_contracts' in results_df.columns:
            y_col = 'contracts_per_partner'
            y_label = 'Contracts per Partner'
        else:
            # Fallback: use reconstruction error
            y_col = 'node_reconstruction_error'
            y_label = 'Reconstruction Error'
            
        x_col = 'num_contracts' if 'num_contracts' in results_df.columns else 'node_reconstruction_error'
        x_label = 'Number of Contracts' if x_col == 'num_contracts' else 'Reconstruction Error'
        
        axes[1, 0].scatter(results_df[normal_mask][x_col],
                          results_df[normal_mask][y_col],
                          alpha=0.6, s=30, label='Normal', color='lightblue')
        axes[1, 0].scatter(results_df[~normal_mask][x_col],
                          results_df[~normal_mask][y_col],
                          alpha=0.8, s=60, label='Anomaly', color='red',
                          marker='x')
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel(y_label)
        axes[1, 0].set_title(f'{x_label} vs {y_label}')
        axes[1, 0].legend()
        
        # Plot 5: Partners vs contracts
        axes[1, 1].scatter(results_df[normal_mask]['num_partners'],
                          results_df[normal_mask]['num_contracts'],
                          alpha=0.6, s=30, label='Normal', color='lightblue')
        axes[1, 1].scatter(results_df[~normal_mask]['num_partners'],
                          results_df[~normal_mask]['num_contracts'],
                          alpha=0.8, s=60, label='Anomaly', color='red',
                          marker='x')
        axes[1, 1].set_xlabel('Number of Partners')
        axes[1, 1].set_ylabel('Number of Contracts')
        axes[1, 1].set_title('Partners vs Contracts')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_anomalous_communities(self, graph_data: Dict,
                                     results_df: pd.DataFrame,
                                     node_embeddings: np.ndarray,
                                     edge_embeddings: np.ndarray) -> Dict:
        """Analyze communities among anomalous entities."""
        logger.info("Analyzing anomalous communities...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i, (name, node_type, is_anom) in enumerate(zip(
            graph_data['nodes'], graph_data['node_types'],
            results_df['is_node_anomaly'])):
            G.add_node(i, name=name,
                      type='buyer' if node_type == 0 else 'supplier',
                      anomaly=is_anom)
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge[0], edge[1])
        
        # Extract anomalous subgraph
        anomalous_nodes = results_df[results_df['is_node_anomaly']].index.tolist()
        anomalous_neighbors = set(anomalous_nodes)
        
        for node in anomalous_nodes:
            anomalous_neighbors.update(G.neighbors(node))
        
        subgraph = G.subgraph(anomalous_neighbors)
        
        # Community detection using embeddings
        communities_info = {}
        if len(anomalous_neighbors) > 3:
            anomalous_node_embeddings = node_embeddings[list(anomalous_neighbors)]
            anomalous_edge_embeddings = edge_embeddings[list(anomalous_neighbors)]
            
            # Use DBSCAN for community detection
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            node_communities = dbscan.fit_predict(anomalous_node_embeddings)
            edge_communities = dbscan.fit_predict(anomalous_edge_embeddings)
            
            # Add community information to nodes
            for node, community in zip(anomalous_neighbors, node_communities):
                G.nodes[node]['community'] = int(community)
            
            communities_info = {
                'num_communities': len(set(node_communities)) - \
                    (1 if -1 in node_communities else 0),
                'noise_points': np.sum(node_communities == -1),
                'subgraph_size': subgraph.number_of_nodes(),
                'subgraph_edges': subgraph.number_of_edges()
            }
            
            # Create interactive visualization
            self.visualize_graph(G, anomalous_nodes, communities_info)
        
        return communities_info

    def visualize_graph(self, G: nx.Graph, anomalous_nodes: List[int],
                       communities_info: Dict):
        """Create an interactive visualization of the graph."""
        try:
            from pyvis.network import Network
            import webbrowser
            from tempfile import NamedTemporaryFile
            
            # Create a new network
            net = Network(height="750px", width="100%", bgcolor="#ffffff",
                        font_color="black")
            
            # Add nodes
            for node in G.nodes():
                node_data = G.nodes[node]
                color = "red" if node in anomalous_nodes else "blue"
                shape = "diamond" if node_data['type'] == 'buyer' else "dot"
                
                # Add node with properties
                net.add_node(
                    node,
                    label=node_data['name'],
                    title=f"Type: {node_data['type']}\n"
                          f"Anomaly: {node_data['anomaly']}\n"
                          f"Community: {node_data.get('community', 'N/A')}",
                    color=color,
                    shape=shape
                )
            
            # Add edges
            for edge in G.edges():
                net.add_edge(edge[0], edge[1])
            
            # Configure physics layout
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000
                    }
                }
            }
            """)
            
            # Save and open in browser
            with NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                webbrowser.open('file://' + tmp.name)
                
        except ImportError:
            logger.warning("pyvis not installed. Install with: pip install pyvis")
            # Fallback to static visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=[n for n in G.nodes()
                                         if n in anomalous_nodes],
                                 node_color='red',
                                 node_size=100,
                                 alpha=0.6)
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=[n for n in G.nodes()
                                         if n not in anomalous_nodes],
                                 node_color='blue',
                                 node_size=50,
                                 alpha=0.4)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            
            # Add labels
            labels = {n: G.nodes[n]['name'] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title("Anomalous Entities Network\n"
                     f"Communities: {communities_info.get('num_communities', 0)}")
            plt.axis('off')
            plt.show()

    def analyze_synthetic_anomaly_detection(self, 
                                          test_graph_data: Dict,
                                          edge_reconstruction_error: np.ndarray,
                                          edge_threshold: float,
                                          threshold_percentile: float = 10,
                                          threshold_percentiles: List[float] = None,
                                          show_plots: bool = True
                                          ) -> Dict:
        """Analyze how well the model detects different types of synthetic 
        anomalies.
        
        Args:
            test_graph_data: Graph data containing contract information
            edge_reconstruction_error: Reconstruction errors from edge model
            edge_threshold: Threshold used for anomaly detection
            threshold_percentiles: Different percentiles to test (optional)
            show_plots: Whether to display plots (default: True)
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing synthetic anomaly detection performance...")
        
        if threshold_percentiles is None:
            threshold_percentiles = [95, 97, 99, 99.5]
        
        # Load the test data with synthetic anomalies
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        test_anomalies_path = os.path.join(data_dir, 'X_test_anomalies.csv')
        
        if not os.path.exists(test_anomalies_path):
            logger.error("Test anomalies file not found. Make sure you've run "
                        "the preprocessing step.")
            return {}
        
        test_data = pd.read_csv(test_anomalies_path, index_col=0)
        
        # Ensure we have the same number of edges as reconstruction errors
        if len(edge_reconstruction_error) != len(test_data):
            logger.warning(f"Mismatch in lengths: "
                          f"{len(edge_reconstruction_error)} errors vs "
                          f"{len(test_data)} contracts. Taking intersection.")
            min_len = min(len(edge_reconstruction_error), len(test_data))
            edge_reconstruction_error = edge_reconstruction_error[:min_len]
            test_data = test_data.iloc[:min_len]
        
        # Get synthetic anomaly labels
        synthetic_labels = test_data['is_synthetic_anomaly'].values
        
        # Define anomaly type mapping (from synthetic generator)
        anomaly_type_names = {
            0: 'Normal',
            1: 'Single Bid Competitive',
            2: 'Price Inflation', 
            3: 'Price Deflation',
            4: 'Procedure Manipulation',
            5: 'Suspicious Modifications',
            6: 'High Market Concentration',
            7: 'Temporal Clustering',
            8: 'Excessive Subcontracting',
            9: 'Short Contract Duration',
            10: 'Suspicious Buyer-Supplier Pairs'
        }
        
        results = {
            'anomaly_type_names': anomaly_type_names,
            'synthetic_labels': synthetic_labels,
            'reconstruction_errors': edge_reconstruction_error,
            'performance_by_threshold': {},
            'performance_by_type': {}
        }
        
        # Analyze performance at different thresholds
        for percentile in threshold_percentiles:
            threshold = np.percentile(edge_reconstruction_error, percentile)
            detected_anomalies = edge_reconstruction_error > threshold
            
            # Overall performance metrics
            from sklearn.metrics import (precision_score, recall_score, 
                                       f1_score, classification_report,
                                       confusion_matrix)
            
            # Convert synthetic labels to binary (0 = normal, 1 = any anomaly)
            true_anomalies = (synthetic_labels > 0).astype(int)
            
            precision = precision_score(true_anomalies, detected_anomalies)
            recall = recall_score(true_anomalies, detected_anomalies)
            f1 = f1_score(true_anomalies, detected_anomalies)
            
            results['performance_by_threshold'][percentile] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'detected_count': np.sum(detected_anomalies),
                'true_anomaly_count': np.sum(true_anomalies)
            }
        
        # Analyze performance by anomaly type using the main threshold
        detected_anomalies = edge_reconstruction_error > edge_threshold
        
        for anomaly_type in range(11):  # 0-10
            type_mask = (synthetic_labels == anomaly_type)
            type_count = np.sum(type_mask)
            
            if type_count == 0:
                continue
                
            type_errors = edge_reconstruction_error[type_mask]
            type_detected = detected_anomalies[type_mask]
            
            # For normal contracts (type 0), we want low detection rate
            if anomaly_type == 0:
                detection_rate = np.sum(type_detected) / type_count
                avg_error = np.mean(type_errors)
                results['performance_by_type'][anomaly_type] = {
                    'name': anomaly_type_names[anomaly_type],
                    'count': type_count,
                    'false_positive_rate': detection_rate,
                    'avg_reconstruction_error': avg_error,
                    'median_reconstruction_error': np.median(type_errors),
                    'std_reconstruction_error': np.std(type_errors)
                }
            else:
                # For anomalous contracts, we want high detection rate
                detection_rate = np.sum(type_detected) / type_count
                avg_error = np.mean(type_errors)
                results['performance_by_type'][anomaly_type] = {
                    'name': anomaly_type_names[anomaly_type],
                    'count': type_count,
                    'detection_rate': detection_rate,
                    'avg_reconstruction_error': avg_error,
                    'median_reconstruction_error': np.median(type_errors),
                    'std_reconstruction_error': np.std(type_errors)
                }
        
        # Create visualizations if requested
        if show_plots:
            self._plot_anomaly_type_analysis(results, edge_threshold)
        
        # Print summary
        self._print_anomaly_analysis_summary(results, threshold_percentile)
        
        return results
    
    def _plot_anomaly_type_analysis(self, results: Dict, 
                                   edge_threshold: float):
        """Create comprehensive visualizations for anomaly type analysis."""
        
        fig = plt.figure(figsize=(20, 15))
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Reconstruction error distribution by anomaly type
        ax1 = fig.add_subplot(gs[0, 0])
        
        synthetic_labels = results['synthetic_labels']
        reconstruction_errors = results['reconstruction_errors']
        anomaly_type_names = results['anomaly_type_names']
        
        # Create box plot data
        plot_data = []
        plot_labels = []
        colors = []
        
        for anomaly_type in sorted(results['performance_by_type'].keys()):
            type_mask = (synthetic_labels == anomaly_type)
            if np.sum(type_mask) > 0:
                type_errors = reconstruction_errors[type_mask]
                plot_data.append(type_errors)
                # Truncate long names for better display
                name = anomaly_type_names[anomaly_type]
                if len(name) > 15:
                    name = name[:12] + "..."
                plot_labels.append(f"{name}\n(n={np.sum(type_mask)})")
                colors.append('lightblue' if anomaly_type == 0 else 'lightcoral')
        
        bp = ax1.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.axhline(y=edge_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({edge_threshold:.4f})')
        ax1.set_ylabel('Reconstruction Error')
        ax1.set_title('Reconstruction Error by Anomaly Type')
        ax1.legend()
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Plot 2: Detection rates by anomaly type
        ax2 = fig.add_subplot(gs[0, 1])
        
        types = []
        detection_rates = []
        counts = []
        
        for anomaly_type in sorted(results['performance_by_type'].keys()):
            if anomaly_type == 0:  # Skip normal for this plot
                continue
            perf = results['performance_by_type'][anomaly_type]
            types.append(perf['name'])
            detection_rates.append(perf['detection_rate'])
            counts.append(perf['count'])
        
        bars = ax2.bar(range(len(types)), detection_rates, 
                      color='skyblue', alpha=0.7)
        ax2.set_xticks(range(len(types)))
        ax2.set_xticklabels([t[:10] + "..." if len(t) > 10 else t 
                           for t in types], rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Detection Rate')
        ax2.set_title('Detection Rate by Anomaly Type')
        ax2.set_ylim(0, 1)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Performance by threshold
        ax3 = fig.add_subplot(gs[0, 2])
        
        thresholds = sorted(results['performance_by_threshold'].keys())
        precisions = [results['performance_by_threshold'][t]['precision'] 
                     for t in thresholds]
        recalls = [results['performance_by_threshold'][t]['recall'] 
                  for t in thresholds]
        f1_scores = [results['performance_by_threshold'][t]['f1_score'] 
                    for t in thresholds]
        
        ax3.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2)
        ax3.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2)
        ax3.plot(thresholds, f1_scores, 'g-^', label='F1-Score', linewidth=2)
        ax3.set_xlabel('Threshold Percentile')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Confusion matrix heatmap (using main threshold)
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Get the main threshold performance (99th percentile by default)
        main_threshold = 99
        if main_threshold in results['performance_by_threshold']:
            threshold_val = results['performance_by_threshold'][main_threshold]['threshold']
        else:
            threshold_val = edge_threshold
            
        detected_anomalies = reconstruction_errors > threshold_val
        true_anomalies = (synthetic_labels > 0).astype(int)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_anomalies, detected_anomalies)
        
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                   yticklabels=['True Normal', 'True Anomaly'])
        ax4.set_title('Confusion Matrix')
        
        # Plot 5: Average reconstruction error by type
        ax5 = fig.add_subplot(gs[1, 1])
        
        all_types = []
        avg_errors = []
        std_errors = []
        
        for anomaly_type in sorted(results['performance_by_type'].keys()):
            perf = results['performance_by_type'][anomaly_type]
            all_types.append(perf['name'])
            avg_errors.append(perf['avg_reconstruction_error'])
            std_errors.append(perf['std_reconstruction_error'])
        
        bars = ax5.bar(range(len(all_types)), avg_errors, 
                      yerr=std_errors, capsize=5, 
                      color=['lightblue' if 'Normal' in t else 'lightcoral' 
                            for t in all_types], alpha=0.7)
        ax5.axhline(y=edge_threshold, color='red', linestyle='--', 
                   label=f'Threshold')
        ax5.set_xticks(range(len(all_types)))
        ax5.set_xticklabels([t[:8] + "..." if len(t) > 8 else t 
                           for t in all_types], rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('Avg Reconstruction Error')
        ax5.set_title('Average Reconstruction Error by Type')
        ax5.legend()
        
        # Plot 6: Detection rate vs average error scatter
        ax6 = fig.add_subplot(gs[1, 2])
        
        detection_rates_all = []
        avg_errors_all = []
        type_names_all = []
        sizes = []
        
        for anomaly_type in sorted(results['performance_by_type'].keys()):
            perf = results['performance_by_type'][anomaly_type]
            if anomaly_type == 0:
                detection_rates_all.append(perf['false_positive_rate'])
            else:
                detection_rates_all.append(perf['detection_rate'])
            avg_errors_all.append(perf['avg_reconstruction_error'])
            type_names_all.append(perf['name'])
            sizes.append(perf['count'] / 10)  # Scale for visibility
        
        scatter = ax6.scatter(avg_errors_all, detection_rates_all, 
                            s=sizes, alpha=0.6, c=range(len(avg_errors_all)), 
                            cmap='viridis')
        
        # Add labels for each point
        for i, name in enumerate(type_names_all):
            ax6.annotate(name[:8] + "..." if len(name) > 8 else name, 
                        (avg_errors_all[i], detection_rates_all[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.axvline(x=edge_threshold, color='red', linestyle='--', alpha=0.7)
        ax6.set_xlabel('Average Reconstruction Error')
        ax6.set_ylabel('Detection Rate')
        ax6.set_title('Detection Rate vs Avg Error')
        
        # Plot 7-9: Histograms for different anomaly types
        selected_types = [0, 2, 6]  # Normal, Price Inflation, Market Concentration
        type_titles = ['Normal Contracts', 'Price Inflation', 'Market Concentration']
        
        for i, (anomaly_type, title) in enumerate(zip(selected_types, type_titles)):
            ax = fig.add_subplot(gs[2, i])
            
            if anomaly_type in results['performance_by_type']:
                type_mask = (synthetic_labels == anomaly_type)
                type_errors = reconstruction_errors[type_mask]
                
                if len(type_errors) > 0:
                    ax.hist(type_errors, bins=30, alpha=0.7, 
                           color='lightblue' if anomaly_type == 0 else 'lightcoral',
                           density=True)
                    ax.axvline(x=edge_threshold, color='red', linestyle='--', 
                              label=f'Threshold')
                    ax.axvline(x=np.mean(type_errors), color='green', linestyle='-', 
                              label=f'Mean')
                    ax.set_xlabel('Reconstruction Error')
                    ax.set_ylabel('Density')
                    ax.set_title(f'{title}\n(n={len(type_errors)})')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(title)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(title)
        
        plt.suptitle('Synthetic Anomaly Detection Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'anomaly_type_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Anomaly analysis plot saved to {save_path}")
        
        plt.show()
    
    def _print_anomaly_analysis_summary(self, results: Dict, 
                                        threshold_percentile: float):
        """Print a comprehensive summary of the anomaly analysis."""
        
        print("\n" + "="*80)
        print("SYNTHETIC ANOMALY DETECTION ANALYSIS")
        print("="*80)
        
        # Overall statistics
        synthetic_labels = results['synthetic_labels']
        total_contracts = len(synthetic_labels)
        normal_contracts = np.sum(synthetic_labels == 0)
        anomalous_contracts = total_contracts - normal_contracts
        
        print(f"Dataset Overview:")
        print(f"  - Total contracts: {total_contracts:,}")
        print(f"  - Normal contracts: {normal_contracts:,} "
              f"({normal_contracts/total_contracts*100:.1f}%)")
        print(f"  - Synthetic anomalies: {anomalous_contracts:,} "
              f"({anomalous_contracts/total_contracts*100:.1f}%)")
        
        # Performance by threshold
        print(f"\nPerformance by Threshold Percentile:")
        print(f"{'Percentile':<12} {'Threshold':<12} {'Precision':<10} "
              f"{'Recall':<8} {'F1-Score':<8} {'Detected':<10}")
        print("-" * 70)
        
        for percentile in sorted(results['performance_by_threshold'].keys()):
            perf = results['performance_by_threshold'][percentile]
            print(f"{percentile:<12} {perf['threshold']:<12.6f} "
                  f"{perf['precision']:<10.3f} {perf['recall']:<8.3f} "
                  f"{perf['f1_score']:<8.3f} {perf['detected_count']:<10}")
        
        # Performance by anomaly type
        print(f"\nPerformance by Anomaly Type:")
        print(f"{'Type':<25} {'Count':<8} {'% Dataset':<10} {'Detection Rate %':<15} "
              f"{'vs Threshold':<12} {'Avg Error':<12} {'Status':<10}")
        print("-" * 97)
        
        for anomaly_type in sorted(results['performance_by_type'].keys()):
            perf = results['performance_by_type'][anomaly_type]
            
            # Calculate percentage of this anomaly type in the full dataset
            dataset_percentage = (perf['count'] / total_contracts) * 100
            
            if anomaly_type == 0:
                rate = perf['false_positive_rate']
                # For normal contracts, compare false positive rate to threshold
                threshold_rate = threshold_percentile / 100
                vs_threshold = f"{rate/threshold_rate:.1f}x" if threshold_rate > 0 else "N/A"
                status = "✅ Good" if rate < threshold_rate else "⚠️ High FP" if rate < threshold_rate * 2 else "❌ Very High FP"
            else:
                rate = perf['detection_rate']
                # Compare detection rate to threshold percentage 
                threshold_rate = threshold_percentile / 100
                improvement_factor = rate / threshold_rate if threshold_rate > 0 else float('inf')
                vs_threshold = f"{improvement_factor:.1f}x"
                
                # Update status to consider threshold performance
                if improvement_factor > 2.0:
                    status = "✅ Excellent"
                elif improvement_factor > 1.5:
                    status = "⚠️ Good"
                elif improvement_factor > 0.8:
                    status = "⚠️ Weak"
                else:
                    status = "❌ Poor"
            
            type_name = perf['name'][:23] + "..." if len(perf['name']) > 23 else perf['name']
            print(f"{type_name:<25} {perf['count']:<8} {dataset_percentage:<10.1f} {rate*100:<15.1f} "
                  f"{vs_threshold:<12} {perf['avg_reconstruction_error']:<12.6f} {status:<10}")
        
        # Best and worst performing anomaly types
        anomaly_performances = []
        for anomaly_type, perf in results['performance_by_type'].items():
            if anomaly_type != 0:  # Skip normal
                anomaly_performances.append((perf['name'], perf['detection_rate'], perf['count']))
        
        if anomaly_performances:
            anomaly_performances.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nBest Detected Anomaly Types:")
            for i, (name, rate, count) in enumerate(anomaly_performances[:3]):
                print(f"  {i+1}. {name}: {rate:.1%} detection rate (n={count})")
            
            print(f"\nWorst Detected Anomaly Types:")
            for i, (name, rate, count) in enumerate(anomaly_performances[-3:]):
                print(f"  {len(anomaly_performances)-i}. {name}: {rate:.1%} detection rate (n={count})")
        
        # Interpretation guide
        print(f"\nInterpretation Guide:")
        print(f"  - '% Dataset': Percentage of this anomaly type in the full dataset")
        print(f"  - 'vs Threshold': How many times better than threshold percentage ({threshold_percentile}%)")
        print(f"    • 1.0x = Same as threshold (random performance)")
        print(f"    • 1.5x = 1.5 times better than threshold")
        print(f"    • >2.0x = Strong anomaly detection")
        
        # Recommendations
        print(f"\nRecommendations:")
        
        # Check false positive rate
        normal_perf = results['performance_by_type'].get(0)
        if normal_perf:
            fp_rate = normal_perf['false_positive_rate']
            if fp_rate > 0.1:
                print(f"  ⚠️  High false positive rate ({fp_rate:.1%}). "
                      f"Consider increasing threshold.")
            elif fp_rate < 0.02:
                print(f"  ✅ Low false positive rate ({fp_rate:.1%}). "
                      f"Good threshold setting.")
        
        # Check overall detection rates vs threshold
        improvement_factors = []
        threshold_rate = threshold_percentile / 100
        for anomaly_type, perf in results['performance_by_type'].items():
            if anomaly_type != 0:  # Skip normal
                if threshold_rate > 0:
                    improvement_factor = perf['detection_rate'] / threshold_rate
                    improvement_factors.append(improvement_factor)
        
        if improvement_factors:
            avg_improvement = np.mean(improvement_factors)
            if avg_improvement > 2.0:
                print(f"  ✅ Strong performance vs threshold "
                      f"({avg_improvement:.1f}x better). Model learns well.")
            elif avg_improvement > 1.5:
                print(f"  ⚠️  Moderate performance vs threshold "
                      f"({avg_improvement:.1f}x better). Some learning.")
            else:
                print(f"  ❌ Weak performance vs threshold "
                      f"({avg_improvement:.1f}x better). Limited learning.")
        
        # Check individual types performing worse than threshold
        poor_performers = []
        for anomaly_type, perf in results['performance_by_type'].items():
            if anomaly_type != 0:  # Skip normal
                if threshold_rate > 0:
                    improvement_factor = perf['detection_rate'] / threshold_rate
                    if improvement_factor < 1.2:  # Less than 1.2x threshold
                        poor_performers.append((perf['name'], improvement_factor))
        
        if poor_performers:
            print(f"  ⚠️  Anomaly types performing poorly vs threshold:")
            for name, factor in poor_performers:
                print(f"    • {name}: {factor:.1f}x threshold")
        
        print("\n" + "="*95)



def main():
    """Main execution function."""
    logger.info("Starting GNN Anomaly Detection Pipeline...")
    
    # Configuration
    DATA_PATH = os.path.join(os.path.dirname(__file__),
                            'data')
    MODEL_PATH = os.path.join(os.path.dirname(__file__),
                             'models', 'anomalies')
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Initialize components
    graph_builder = ProcurementGraphBuilder()
    gnn_detector = GNNAnomalyDetector(hidden_dim=64, output_dim=32,
                                     num_layers=3)
    analyzer = AnomalyAnalyzer()
    
    try:
        # Load and preprocess data
        X = graph_builder.load_data(DATA_PATH)
        # Now using all three splits: train, val, test (with synthetic anomalies)
        (X_train_preproc, X_val_preproc, X_test_preproc, 
         X_train, X_val, X_test) = graph_builder.preprocess_data(X)
        
        # Create graphs for all three splits
        X_train_graph = graph_builder.create_graph(X_train_preproc, X_train, type='train')
        X_val_graph = graph_builder.create_graph(X_val_preproc, X_val, type='val')
        X_test_graph = graph_builder.create_graph(X_test_preproc, X_test, type='test')
        
        # Scale derived node/edge features using training data
        logger.info("Scaling derived node and edge features...")
        node_features_train = X_train_graph['node_features']
        edge_features_train = X_train_graph['edge_features']
        
        # Fit scalers on training data (derived features)
        node_features_train_scaled = graph_builder.node_scaler.fit_transform(
            node_features_train)
        edge_features_train_scaled = graph_builder.edge_scaler.fit_transform(
            edge_features_train)
        
        # Transform validation features using training scalers
        node_features_val = X_val_graph['node_features']
        edge_features_val = X_val_graph['edge_features']
        node_features_val_scaled = graph_builder.node_scaler.transform(
            node_features_val)
        edge_features_val_scaled = graph_builder.edge_scaler.transform(
            edge_features_val)
        
        # Transform test features using training scalers
        node_features_test = X_test_graph['node_features']
        edge_features_test = X_test_graph['edge_features']
        node_features_test_scaled = graph_builder.node_scaler.transform(
            node_features_test)
        edge_features_test_scaled = graph_builder.edge_scaler.transform(
            edge_features_test)
        
        # Create TensorFlow graphs for all three splits
        X_train_tf_graph = gnn_detector.create_tensorflow_graph(
            X_train_graph, node_features_train_scaled, edge_features_train_scaled)
        X_val_tf_graph = gnn_detector.create_tensorflow_graph(
            X_val_graph, node_features_val_scaled, edge_features_val_scaled)
        X_test_tf_graph = gnn_detector.create_tensorflow_graph(
            X_test_graph, node_features_test_scaled, edge_features_test_scaled)
        
        # Store graph tensors for later use
        gnn_detector.graph_tensor_train = X_train_tf_graph
        gnn_detector.graph_tensor_val = X_val_tf_graph
        gnn_detector.graph_tensor_test = X_test_tf_graph
        
        # Build both models
        gnn_detector.node_model = gnn_detector.build_node_model(
            node_features_train_scaled.shape[1], 
            edge_features_train_scaled.shape[1])
        gnn_detector.edge_model = gnn_detector.build_edge_model(
            node_features_train_scaled.shape[1], 
            edge_features_train_scaled.shape[1])
        
        # Train both models with proper validation data
        logger.info("Training models using validation data for monitoring...")
        node_history = gnn_detector.train_node_model(
            X_train_tf_graph, 
            validation_graph_tensor=X_val_tf_graph,
            epochs=100)
        edge_history = gnn_detector.train_edge_model(
            X_train_tf_graph, 
            validation_graph_tensor=X_val_tf_graph,
            epochs=100)
        
        # Plot training histories
        node_plot_save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'node_training_history.png')
        edge_plot_save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'edge_training_history.png')
        
        # Note: We'll need to update plot_training_history for single model
        # For now, we'll skip the plotting and just log the results
        logger.info("Node training completed. Final losses:")
        logger.info(f"Node reconstruction loss: "
                   f"{node_history['node_reconstructed_loss'][-1]:.4f}")
        logger.info("Edge training completed. Final losses:")
        logger.info(f"Edge reconstruction loss: "
                   f"{edge_history['edge_reconstructed_loss'][-1]:.4f}")
        
        # Add training history plots
        gnn_detector.plot_node_training_history(node_history)
        gnn_detector.plot_edge_training_history(edge_history)
        
        # Detect anomalies on TEST data (contains synthetic anomalies)
        logger.info("Detecting anomalies on test data with synthetic anomalies...")
        node_reconstruction_error, node_threshold = (
            gnn_detector.detect_node_anomalies())
        edge_reconstruction_error, edge_threshold = (
            gnn_detector.detect_edge_anomalies())
        
        # Calculate anomaly masks
        node_anomalies = node_reconstruction_error > node_threshold
        edge_anomalies = edge_reconstruction_error > edge_threshold
        
        # Analyze synthetic anomaly detection performance
        logger.info("Analyzing synthetic anomaly detection performance...")
        synthetic_analysis = analyzer.analyze_synthetic_anomaly_detection(
            X_test_graph, edge_reconstruction_error, edge_threshold, 
            show_plots=False)  # Set to True if you want to see plots
        
        # OPTIONAL: Also evaluate on training data for comparison
        logger.info("Evaluating models on training data for comparison...")
        train_node_error, _ = gnn_detector.detect_node_anomalies(
            graph_tensor=X_train_tf_graph, threshold_percentile=95)
        train_edge_error, _ = gnn_detector.detect_edge_anomalies(
            graph_tensor=X_train_tf_graph, threshold_percentile=95)
        train_node_anomalies = train_node_error > node_threshold
        train_edge_anomalies = train_edge_error > edge_threshold
        
        print(f"Training data anomalies: {np.sum(train_node_anomalies)} nodes "
              f"({np.sum(train_node_anomalies)/len(train_node_anomalies)*100:.1f}%), "
              f"{np.sum(train_edge_anomalies)} edges "
              f"({np.sum(train_edge_anomalies)/len(train_edge_anomalies)*100:.1f}%)")
        
        # Get embeddings for analysis from TEST data
        node_predictions = gnn_detector.node_model.predict(
            gnn_detector.graph_tensor_test)
        edge_predictions = gnn_detector.edge_model.predict(
            gnn_detector.graph_tensor_test)
        node_embeddings = node_predictions['node_embeddings']
        edge_embeddings = edge_predictions['edge_embeddings']
        
        # Create results DataFrames using TEST data
        results_df = analyzer.create_node_results_dataframe(
            X_test_graph, node_reconstruction_error, node_anomalies)
        
        edge_results_df = analyzer.create_edge_results_dataframe(
            X_test_graph, edge_reconstruction_error, edge_anomalies)
        
        # Analyze communities
        communities_info = analyzer.analyze_anomalous_communities(
            X_test_graph, results_df, node_embeddings, edge_embeddings)
        
        # Visualize results
        analyzer.plot_results(results_df, node_reconstruction_error, 
                             edge_reconstruction_error,
                             node_threshold, edge_threshold)
        
        # Save results including synthetic analysis
        gnn_detector.node_model.save(os.path.join(MODEL_PATH, 
                                                 'gnn_node_anomaly_model'))
        gnn_detector.edge_model.save(os.path.join(MODEL_PATH, 
                                                 'gnn_edge_anomaly_model'))
        results_df.to_csv(os.path.join(MODEL_PATH, 
                                      'gnn_node_anomaly_results.csv'),
                         index=False)
        edge_results_df.to_csv(os.path.join(MODEL_PATH, 
                                           'gnn_edge_anomaly_results.csv'),
                              index=False)
        np.save(os.path.join(MODEL_PATH, 'gnn_node_embeddings.npy'),
               node_embeddings)
        np.save(os.path.join(MODEL_PATH, 'gnn_edge_embeddings.npy'),
               edge_embeddings)
        
        # Save synthetic anomaly analysis results
        import pickle
        with open(os.path.join(MODEL_PATH, 'synthetic_analysis.pkl'), 'wb') as f:
            pickle.dump(synthetic_analysis, f)
        
        # Print summary
        print("\n" + "="*60)
        print("GNN ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Training entities: {len(X_train_graph['nodes'])}")
        print(f"- Training buyers: {np.sum(X_train_graph['node_types'] == 0)}")
        print(f"- Training suppliers: {np.sum(X_train_graph['node_types'] == 1)}")
        print(f"Training contracts: {len(X_train_graph['edges'])}")
        print(f"\nTest entities analyzed: {len(X_test_graph['nodes'])}")
        print(f"- Test buyers: {np.sum(X_test_graph['node_types'] == 0)}")
        print(f"- Test suppliers: {np.sum(X_test_graph['node_types'] == 1)}")
        print(f"Test contracts analyzed: {len(X_test_graph['edges'])}")
        
        print(f"\nNode anomalies detected: {np.sum(node_anomalies)} "
              f"({np.sum(node_anomalies)/len(node_anomalies)*100:.1f}%)")
        print(f"Edge anomalies detected: {np.sum(edge_anomalies)} "
              f"({np.sum(edge_anomalies)/len(edge_anomalies)*100:.1f}%)")
        
        anomalous_buyers = results_df[
            (results_df['is_node_anomaly']) &
            (results_df['entity_type'] == 'Buyer')]
        anomalous_suppliers = results_df[
            (results_df['is_node_anomaly']) &
            (results_df['entity_type'] == 'Supplier')]
        
        print(f"- Anomalous buyers: {len(anomalous_buyers)}")
        print(f"- Anomalous suppliers: {len(anomalous_suppliers)}")
        print(f"\nModel performance:")
        print(f"- Final node reconstruction loss: "
              f"{node_history['node_reconstructed_loss'][-1]:.4f}")
        print(f"- Final edge reconstruction loss: "
              f"{edge_history['edge_reconstructed_loss'][-1]:.4f}")
        print(f"- Node anomaly threshold: {node_threshold:.4f}")
        print(f"- Edge anomaly threshold: {edge_threshold:.4f}")
        
        # Print synthetic anomaly detection summary
        if synthetic_analysis:
            print(f"\nSynthetic Anomaly Detection Performance:")
            if 99 in synthetic_analysis['performance_by_threshold']:
                perf = synthetic_analysis['performance_by_threshold'][99]
                print(f"- Precision: {perf['precision']:.3f}")
                print(f"- Recall: {perf['recall']:.3f}")
                print(f"- F1-Score: {perf['f1_score']:.3f}")
            
            # Show best and worst detected anomaly types
            anomaly_rates = []
            for anomaly_type, perf in synthetic_analysis['performance_by_type'].items():
                if anomaly_type != 0:  # Skip normal
                    anomaly_rates.append((perf['name'], perf['detection_rate']))
            
            if anomaly_rates:
                anomaly_rates.sort(key=lambda x: x[1], reverse=True)
                print(f"- Best detected: {anomaly_rates[0][0]} "
                      f"({anomaly_rates[0][1]:.1%})")
                print(f"- Worst detected: {anomaly_rates[-1][0]} "
                      f"({anomaly_rates[-1][1]:.1%})")
        
        if communities_info:
            print(f"\nCommunity analysis:")
            print(f"- Communities detected: "
                  f"{communities_info['num_communities']}")
            print(f"- Anomalous subgraph size: "
                  f"{communities_info['subgraph_size']} nodes")
        
        print(f"\nTop 5 most anomalous entities:")
        top_anomalies = results_df.head(5)[
            ['entity_name', 'entity_type', 'node_reconstruction_error']]
        print(top_anomalies.to_string(index=False))
        
        print(f"\nTop 5 most anomalous contracts:")
        top_edge_anomalies = edge_results_df.head(5)[
            ['buyer_name', 'supplier_name', 'amount', 'edge_reconstruction_error']]
        print(top_edge_anomalies.to_string(index=False))
        
        # Visualize the test procurement graph
        graph_builder.visualize_procurement_graph(X_test_graph, 
            "French Public Procurement Network (Test Set)")
        
        logger.info("GNN Anomaly Detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in GNN pipeline: {str(e)}")
        raise





def analyze_edge_anomaly_types(test_graph_data: Dict, 
                               edge_reconstruction_error: np.ndarray, 
                               edge_threshold: float,
                               show_plots: bool = True) -> Dict:
    """
    Convenience function to analyze what types of synthetic anomalies 
    the edge model can detect.
    
    Args:
        test_graph_data: Graph data from test set
        edge_reconstruction_error: Reconstruction errors from edge model
        edge_threshold: Threshold used for anomaly detection
        show_plots: Whether to display plots (default: True)
        
    Returns:
        Dictionary with analysis results
        
    Usage:
        # After training your edge model and getting reconstruction errors:
        results = analyze_edge_anomaly_types(
            test_graph_data, 
            edge_reconstruction_error, 
            edge_threshold,
            show_plots=False  # Add this to suppress plots
        )
    """
    analyzer = AnomalyAnalyzer()
    return analyzer.analyze_synthetic_anomaly_detection(
        test_graph_data, 
        edge_reconstruction_error, 
        edge_threshold,
        show_plots=show_plots
    )


if __name__ == "__main__":
    main()


# USAGE EXAMPLE FOR NOTEBOOK:
"""
# Correct sequence for using this module in a notebook:

# 1. Initialize components
graph_builder = ProcurementGraphBuilder()
gnn_detector = GNNAnomalyDetector(hidden_dim=64, output_dim=32, num_layers=3)

# 2. Load and preprocess data
X = graph_builder.load_data(DATA_PATH)  # Replace DATA_PATH with your path
X_train_preproc, X_test_preproc, X_train, X_test = graph_builder.preprocess_data(X)

# 3. Create train graph from preprocessed data
train_graph_data = graph_builder.create_graph(X_train_preproc, X_train, type='train')

# 4. Scale the training features
train_node_features = train_graph_data['node_features']
train_edge_features = train_graph_data['edge_features']
train_node_features_scaled = graph_builder.node_scaler.fit_transform(train_node_features)
train_edge_features_scaled = graph_builder.edge_scaler.fit_transform(train_edge_features)

# 5. Create TensorFlow training graph
train_graph_tensor = gnn_detector.create_tensorflow_graph(
    train_graph_data, train_node_features_scaled, train_edge_features_scaled)

# 6. Build both models
gnn_detector.node_model = gnn_detector.build_node_model(
    train_node_features_scaled.shape[1], train_edge_features_scaled.shape[1])
gnn_detector.edge_model = gnn_detector.build_edge_model(
    train_node_features_scaled.shape[1], train_edge_features_scaled.shape[1])

# 7. Train both models
node_history = gnn_detector.train_node_model(train_graph_tensor, epochs=50)
edge_history = gnn_detector.train_edge_model(train_graph_tensor, epochs=50)

# 8. Create test graph
test_graph_data = graph_builder.create_graph(X_test_preproc, X_test, type='test')
test_node_features_scaled = graph_builder.node_scaler.transform(test_graph_data['node_features'])
test_edge_features_scaled = graph_builder.edge_scaler.transform(test_graph_data['edge_features'])
test_graph_tensor = gnn_detector.create_tensorflow_graph(
    test_graph_data, test_node_features_scaled, test_edge_features_scaled)
gnn_detector.graph_tensor_test = test_graph_tensor

# 9. Detect anomalies on test data using both models
node_reconstruction_error, node_threshold = gnn_detector.detect_node_anomalies()
edge_reconstruction_error, edge_threshold = gnn_detector.detect_edge_anomalies()

# 10. Optionally detect anomalies on training data for comparison
train_node_error, _ = gnn_detector.detect_node_anomalies(
    graph_tensor=train_graph_tensor, threshold_percentile=95)
train_edge_error, _ = gnn_detector.detect_edge_anomalies(
    graph_tensor=train_graph_tensor, threshold_percentile=95)
""" 