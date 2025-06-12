import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class IdAccordCadreEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that converts idAccordCadre to a binary indicator.

    Transforms the idAccordCadre column by checking if it contains a value (1)
    or is null (0), indicating whether the contract is part of a framework
    agreement.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['idAccordCadre'] = (
            X_transformed['idAccordCadre'].notnull().astype(int))
        return X_transformed


class TauxAvanceCategorizer(BaseEstimator, TransformerMixin):
    """
    Transformer that categorizes the tauxAvance (advance payment rate) into
    bins.

    Converts continuous advance payment rates into discrete categories based on
    the provided bins and labels.

    Parameters
    ----------
    bins : list, default=[-0.001, 0.001, 0.05, 0.15, 1.0]
        Bin edges for categorizing advance payment rates.
    labels : list, default=['no_advance', 'small_advance', 'medium_advance',
        'large_advance']
        Labels for the resulting categories.
    """
    def __init__(self, bins=[-0.001, 0.001, 0.05, 0.15, 1.0],
                 labels=['no_advance', 'small_advance', 'medium_advance',
                         'large_advance']):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['tauxAvance_cat'] = pd.cut(
            X_transformed['tauxAvance'],
            bins=self.bins,
            labels=self.labels)
        return X_transformed.drop(columns=['tauxAvance'])


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies log1p transformation to numerical features.

    Applies np.log1p (natural logarithm of 1 + x) to specified numerical
    columns or to all numerical columns if none are specified. Handles both
    DataFrame and numpy array inputs.

    Parameters
    ----------
    columns : list or None, default=None
        List of column names to transform. If None, transforms all numerical
        columns.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like or None, default=None
            Input feature names

        Returns
        -------
        feature_names_out : ndarray of str objects
            Same as input_features if provided, empty array otherwise.
        """
        return (input_features if input_features is not None
                else np.array([]))

    def transform(self, X):
        """
        Apply log1p transformation to numerical columns.

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            Data to transform

        Returns
        -------
        X_transformed : pandas DataFrame or numpy array
            Log-transformed data in the same format as the input
        """
        # Convert to DataFrame if input is an array
        is_array = not hasattr(X, 'columns')
        if is_array:
            X = pd.DataFrame(X, columns=[f'col_{i}'
                                         for i in range(X.shape[1])])

        X_transformed = X.copy()
        # Apply log transform to all numeric columns
        if self.columns:
            for col in self.columns:
                if (col in X_transformed.columns and
                        np.issubdtype(X_transformed[col].dtype, np.number)):
                    X_transformed[col] = np.log1p(X_transformed[col])
        else:
            # If no columns specified, transform all numeric columns
            for col in X_transformed.select_dtypes(
                    include=[np.number]).columns:
                X_transformed[col] = np.log1p(X_transformed[col])

        # Return array if input was array
        if is_array:
            return X_transformed.values
        return X_transformed


class InitTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for initial data preprocessing and feature selection.

    Applies the columns_selection function to filter and transform the data
    based on the specified column lists.

    Parameters
    ----------
    numerical_columns : list
        List of numerical column names
    binary_columns : list
        List of binary column names
    categorical_columns : list
        List of categorical column names
    """
    def __init__(self, numerical_columns, binary_columns, categorical_columns):
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Apply initial data transformations and selections.

        Parameters
        ----------
        X : pandas DataFrame
            Input data

        Returns
        -------
        pandas DataFrame
            Transformed and filtered data
        """
        return columns_selection(X, self.numerical_columns,
                                 self.binary_columns, self.categorical_columns)


class StringConverter(BaseEstimator, TransformerMixin):
    """
    Transformer that converts all values to strings.

    Ensures categorical data is uniformly represented as strings before
    encoding.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Convert all values in X to strings.

        Parameters
        ----------
        X : array-like
            Input data

        Returns
        -------
        array-like
            Data with all values converted to strings
        """
        return X.astype(str)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like or None, default=None
            Input feature names

        Returns
        -------
        feature_names_out : ndarray of str objects
            Same as input_features if provided, empty array otherwise.
        """
        return (input_features if input_features is not None
                else np.array([]))


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts array output to DataFrame with feature names.
    
    This transformer is needed to ensure the final output is a pandas DataFrame
    with proper column names, making it pickleable unlike lambda functions.
    """
    def __init__(self, column_transformer=None):
        self.column_transformer = column_transformer
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
    def fit(self, X, y=None):
        # Store input feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self
        
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like or None, default=None
            Input feature names from the previous transformer
            
        Returns
        -------
        feature_names_out : ndarray of str objects
            Feature names for the output DataFrame
        """
        if input_features is not None:
            return np.array(input_features)
        elif self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_)
        elif (self.column_transformer is not None and
              hasattr(self.column_transformer, 'get_feature_names_out')):
            return self.column_transformer.get_feature_names_out()
        elif self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_)
        else:
            # Fallback to generic names
            return np.array([f'feature_{i}' for i in range(self.n_features_)])
        
    def transform(self, X):
        """
        Convert array to DataFrame with feature names.
        
        Parameters
        ----------
        X : array-like
            Input data (typically from column transformer output)
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with proper column names
        """
        # If X is already a DataFrame, return as is
        if hasattr(X, 'columns'):
            return X
            
        # Determine number of features
        if hasattr(X, 'shape'):
            self.n_features_ = X.shape[1] if len(X.shape) > 1 else len(X)
        else:
            self.n_features_ = len(X[0]) if len(X) > 0 else 0
            
        # Try to get proper feature names from column transformer
        if (self.column_transformer is not None and
            hasattr(self.column_transformer, 'get_feature_names_out')):
            try:
                feature_names = self.column_transformer.get_feature_names_out()
            except Exception:
                feature_names = ([f'feature_{i}'
                                  for i in range(self.n_features_)])
        elif self.feature_names_out_ is not None:
            feature_names = self.feature_names_out_
        else:
            # Generate generic column names as fallback
            feature_names = [f'feature_{i}' for i in range(self.n_features_)]
            
        return pd.DataFrame(X, columns=feature_names)
        
    def set_output_columns(self, feature_names):
        """
        Set the feature names to use for output DataFrame.
        
        Parameters
        ----------
        feature_names : array-like
            Feature names to use for DataFrame columns
        """
        self.feature_names_out_ = list(feature_names)
        return self


def columns_selection(df, numerical_columns, binary_columns,
                      categorical_columns):
    """
    Select and filter columns based on provided column lists.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    numerical_columns : list
        List of numerical column names
    binary_columns : list
        List of binary column names  
    categorical_columns : list
        List of categorical column names

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with only the specified columns
    """
    all_columns = numerical_columns + binary_columns + categorical_columns
    df = df[all_columns]
    return df


def create_pipeline(numerical_columns, binary_columns, categorical_columns):
    """
    Create a complete preprocessing pipeline for public contract data.
    
    This pipeline handles initial preprocessing (feature selection,
    categorical encoding) and follow-up preprocessing (imputation, scaling,
    one-hot encoding).
    
    Parameters
    ----------
    numerical_columns : list
        List of numerical column names to process
    binary_columns : list  
        List of binary column names to process
    categorical_columns : list
        List of categorical column names to process
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing pipeline that outputs a pandas DataFrame
    """
    
    # Initial preprocessing
    init_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector',
         InitTransformer(numerical_columns, binary_columns,
                         categorical_columns))
    ])
    
    # Column transformer
    column_transformer = ColumnTransformer([
        ('offres_recues_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), (['offresRecues'] if 'offresRecues' in numerical_columns
             else [])),
        
        ('other_num_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), [col for col in numerical_columns if col != 'offresRecues']),
        
        ('binary_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ]), binary_columns),
        
        ('cat_pipeline', Pipeline([
            ('imputer',
             SimpleImputer(strategy='constant', fill_value='missing')),
            ('string_converter', StringConverter()),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                     sparse_output=False))
        ]), categorical_columns)
    ], remainder='drop', verbose_feature_names_out=True)
    
    # Follow-up processing
    follow_pipeline = Pipeline([
        ('column_transformer', column_transformer)
    ])
    
    # Create DataFrameTransformer for proper feature names
    df_transformer = DataFrameTransformer(column_transformer)
    
    # Complete pipeline
    complete_pipeline = Pipeline([
        ('init', init_pipeline),
        ('follow', follow_pipeline),
        ('to_dataframe', df_transformer)
    ])
    
    return complete_pipeline


def create_pipeline_cat(cat):

    if cat == 'pred_montant':

        numerical_columns = ['dureeMois', 'offresRecues', 'annee']
        binary_columns = ['sousTraitanceDeclaree', 'origineFrance', 
                          'marcheInnovant', 'idAccordCadre']
        categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'tauxAvance_cat',
                               'codeCPV_2_3'
                               ]

    elif cat == 'marche_sim':
        numerical_columns = ['montant', 'dureeMois', 'offresRecues']
        binary_columns = ['sousTraitanceDeclaree', 'origineFrance', 
                          'marcheInnovant', 'idAccordCadre']
        categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'tauxAvance_cat',
                               'codeCPV_2'
                               ]

    elif cat == 'anomalie':
        numerical_columns = ['montant', 'dureeMois', 'offresRecues']
        binary_columns = ['sousTraitanceDeclaree', 'origineFrance', 
                          'marcheInnovant', 'idAccordCadre']
        categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'tauxAvance_cat',
                               'codeCPV_3'
                               ]
    
    else:
        error_msg = ("Error, cat not in 'pred_montant', 'marche_sim', "
                     "'anomalie'.")
        return error_msg
    
    return create_pipeline(numerical_columns, binary_columns,
                           categorical_columns)