'''
This script is used to add CPV descriptions to the raw data by matching CPV codes.
'''

# Import libraries
import pandas as pd
import os
import sqlite3

def codeCPV_description(data_raw):
    """
    Add CPV descriptions to the raw data by matching CPV codes.
    
    Args:
        data_raw (pd.DataFrame): Raw data containing 'codeCPV' column
        
    Returns:
        pd.DataFrame: Data with added 'codeCPV_FR' column containing 
                     CPV descriptions in French
    """
    # Load CPV reference data
    cpv_path = os.path.join(os.path.dirname(__file__), '..', 
                            'docs', 'cpv_2008_ver_2013_FR.csv')
    df_cpv = pd.read_csv(cpv_path)
    
    # Find missing CPV codes (not in reference data)
    missing_cpv_codes = data_raw[
        ~data_raw['codeCPV'].isin(df_cpv['CODE'])
    ]['codeCPV'].unique()
    missing_cpv = pd.DataFrame(missing_cpv_codes, columns=['codeCPV'])
    
    # Try to find similar CPV codes for missing ones
    missing_cpv.loc[:, 'count_similar'] = (
        missing_cpv['codeCPV'].astype(str).apply(
            lambda cpv: df_cpv['CODE'].str.startswith(cpv).sum()
        )
    )
    
    missing_cpv.loc[:, 'new_CPV'] = (
        missing_cpv['codeCPV'].astype(str).apply(
            lambda cpv: (
                df_cpv[df_cpv['CODE'].str.startswith(cpv)]['CODE'].values[0] 
                if df_cpv[df_cpv['CODE'].str.startswith(cpv)].shape[0] > 0 
                else None
            )
        )
    )
    
    # Merge missing CPV codes with their descriptions
    missing_cpv = pd.merge(
        missing_cpv, df_cpv[['CODE', 'FR']], 
        left_on='new_CPV', right_on='CODE', how='left'
    )
    missing_cpv.rename(columns={'FR': 'codeCPV_FR'}, inplace=True)
    
    # Handle correct CPV codes (already in reference data)
    correct_cpv_codes = data_raw[
        data_raw['codeCPV'].isin(df_cpv['CODE'])
    ]['codeCPV'].unique()
    correct_cpv = pd.DataFrame(correct_cpv_codes, columns=['codeCPV'])
    
    correct_cpv = pd.merge(
        correct_cpv, df_cpv[['CODE', 'FR']], 
        left_on='codeCPV', right_on='CODE', how='left'
    )
    correct_cpv.rename(columns={'FR': 'codeCPV_FR'}, inplace=True)
    
    # Combine both correct and missing CPV mappings
    cpvFR = pd.concat([
        correct_cpv[['codeCPV', 'codeCPV_FR']], 
        missing_cpv[['codeCPV', 'codeCPV_FR']]
    ])
    
    # Merge original data with CPV descriptions
    data_cpv = pd.merge(
        data_raw, cpvFR, left_on='codeCPV', 
        right_on='codeCPV', how='left'
    )
    
    return data_cpv


def extract_cpv_hierarchy_level(cpv_code, level=2):
    """
    Extract higher-level hierarchy code from a CPV code.
    
    Args:
        cpv_code (str): Original CPV code (e.g., '03111900-1')
        level (int): Hierarchy level to extract (2-5):
                    - 2: Division (XX000000): First 2 digits + 6 zeros
                    - 3: Group (XXX00000): First 3 digits + 5 zeros  
                    - 4: Class (XXXX0000): First 4 digits + 4 zeros
                    - 5: Category (XXXXX000): First 5 digits + 3 zeros
    
    Returns:
        str: Higher-level CPV code (e.g., '03000000')
        
    Raises:
        ValueError: If level is not between 2 and 5
        TypeError: If cpv_code is None or cannot be converted to string
    """
    if cpv_code is None:
        raise TypeError("CPV code cannot be None")
    
    if not 2 <= level <= 5:
        raise ValueError("Level must be between 2 and 5")
    
    # Remove any whitespace and convert to string
    cpv_str = str(cpv_code).strip()
    
    if not cpv_str:
        raise ValueError("CPV code cannot be empty")
    
    # Extract the numeric part before the dash
    if '-' in cpv_str:
        numeric_part = cpv_str.split('-')[0]
    else:
        numeric_part = cpv_str
    
    # Ensure we have at least 8 digits, pad with zeros if needed
    numeric_part = numeric_part.ljust(8, '0')
    
    # Extract based on hierarchy level
    zeros_to_add = 8 - level
    return numeric_part[:level] + '0' * zeros_to_add


def add_cpv_hierarchy_column(df, cpv_column='codeCPV', level=2, 
                             new_column_name=None):
    """
    Add a new column with higher-level CPV hierarchy codes to a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing CPV codes
        cpv_column (str): Name of the column containing CPV codes 
                        (default: 'codeCPV')
        level (int): Hierarchy level to extract (2-5, default: 2)
        new_column_name (str): Name for the new column. If None, 
                             will be auto-generated as 'codeCPV_{level}'
    
    Returns:
        pd.DataFrame: DataFrame with added hierarchy column
        
    Raises:
        KeyError: If cpv_column doesn't exist in DataFrame
        ValueError: If level is not between 2 and 5
    """
    if cpv_column not in df.columns:
        raise KeyError(f"Column '{cpv_column}' not found in DataFrame")
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Auto-generate column name if not provided
    if new_column_name is None:
        new_column_name = f'codeCPV_{level}'
    
    # Apply the hierarchy extraction function
    df_copy[new_column_name] = df_copy[cpv_column].apply(
        lambda x: extract_cpv_hierarchy_level(x, level=level)
    )
    
    return df_copy


def codeCPV_group(data_cpv, levels=None, save_csv=True, output_path=None,
                  cpv_column='codeCPV'):
    """
    Add CPV hierarchy columns to the data and optionally save to CSV.
    
    This function creates hierarchical CPV code columns based on different
    levels of the CPV classification system. Each level represents a 
    different granularity of classification.
    
    Args:
        data_cpv (pd.DataFrame): DataFrame containing CPV codes with 
                               descriptions
        levels (list of int, optional): List of hierarchy levels to add 
                                      (2-5). Defaults to [2, 3, 4, 5]
        save_csv (bool, optional): Whether to save the result to CSV. 
                                 Defaults to True
        output_path (str, optional): Path where to save CSV file. If None,
                                   saves to '../data/data_cpv.csv'
        cpv_column (str, optional): Name of the column containing CPV codes.
                                  Defaults to 'codeCPV'
    
    Returns:
        pd.DataFrame: DataFrame with added CPV hierarchy columns
        
    Raises:
        KeyError: If cpv_column doesn't exist in DataFrame
        ValueError: If any level in levels is not between 2 and 5
        FileNotFoundError: If output directory doesn't exist and cannot 
                         be created
    
    Example:
        >>> data = pd.DataFrame({'codeCPV': ['03111900-1', '45000000-7']})
        >>> result = codeCPV_group(data, levels=[2, 3], save_csv=False)
        >>> print(result.columns)
        Index(['codeCPV', 'codeCPV_2', 'codeCPV_3'], dtype='object')
    """
    # Input validation
    if not isinstance(data_cpv, pd.DataFrame):
        raise TypeError("data_cpv must be a pandas DataFrame")
    
    if cpv_column not in data_cpv.columns:
        raise KeyError(f"Column '{cpv_column}' not found in DataFrame")
    
    if levels is None:
        levels = [2, 3, 4, 5]
    
    # Validate levels
    for level in levels:
        if not isinstance(level, int) or not 2 <= level <= 5:
            level_msg = f"Level {level} must be an integer between 2 and 5"
            raise ValueError(level_msg)
    
    # Create a copy to avoid modifying the original DataFrame
    data_cpv_new = data_cpv.copy()
    
    # Add hierarchy columns for each specified level
    for level in levels:
        try:
            data_cpv_new = add_cpv_hierarchy_column(
                data_cpv_new, 
                cpv_column=cpv_column, 
                level=level
            )
        except Exception as e:
            print(f"Warning: Failed to add hierarchy level {level}: {e}")
            continue
    
    # Save to CSV if requested
    if save_csv:
        try:
            if output_path is None:
                data_path = os.path.join(
                    os.path.dirname(__file__), '..', 'data'
                )
                output_path = os.path.join(data_path, 'data_cpv.csv')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with index=False to avoid saving row indices
            data_cpv_new.to_csv(output_path, index=False)
            print(f"Data saved successfully to: {output_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save CSV file: {e}")
    
    return data_cpv_new
