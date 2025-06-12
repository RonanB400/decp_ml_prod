import pandas as pd
import os
import sqlite3
import sys

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.preprocess_cpv import codeCPV_description, codeCPV_group
except ImportError:
    from preprocess_cpv import codeCPV_description, codeCPV_group


def process_cpv_data(data_raw, levels=None, save_csv=True, output_path=None):
    """
    Complete CPV processing pipeline: adds descriptions and hierarchy columns.

    This function combines codeCPV_description and codeCPV_group to provide
    a complete CPV data processing pipeline. It first adds CPV descriptions
    by matching codes, then adds hierarchical CPV classification columns.

    Args:
        data_raw (pd.DataFrame): Raw data containing 'codeCPV' column
        levels (list of int, optional): List of hierarchy levels to add
                                      (2-5). Defaults to [2, 3, 4, 5]
        save_csv (bool, optional): Whether to save the result to CSV.
                                 Defaults to True
        output_path (str, optional): Path where to save CSV file. If None,
                                   saves to '../data/data_cpv.csv'

    Returns:
        pd.DataFrame: DataFrame with CPV descriptions and hierarchy columns

    Raises:
        KeyError: If 'codeCPV' column doesn't exist in DataFrame
        ValueError: If any level in levels is not between 2 and 5
        FileNotFoundError: If output directory doesn't exist and cannot
                         be created

    Example:
        >>> data = pd.DataFrame({'codeCPV': ['03111900-1', '45000000-7']})
        >>> result = process_cpv_data(data, levels=[2, 3], save_csv=False)
        >>> print(result.columns)
        Index(['codeCPV', 'codeCPV_FR', 'codeCPV_2', 'codeCPV_3'],
              dtype='object')
    """
    print("Step 1: Adding CPV descriptions...")
    data_with_descriptions = codeCPV_description(data_raw)

    print("Step 2: Adding CPV hierarchy columns...")
    data_with_hierarchy = codeCPV_group(
        data_with_descriptions,
        levels=levels,
        save_csv=save_csv,
        output_path=output_path
    )

    print("CPV processing completed successfully!")
    return data_with_hierarchy


def drop_outliers(df, min=20_000, max=50_000_000):
    """
    Remove rows with outlier values in montant and dureeMois columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    min : int, default=20000
        Minimum value threshold for montant
    max : int, default=50000000
        Maximum value threshold for montant

    Returns
    -------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    try:
        # Check if 'montant' column exists before filtering
        if 'montant' in df.columns:
            df = df.drop(df[df['montant'] > max].index)
            df = df.drop(df[df['montant'] < min].index)

        # Check if 'dureeMois' column exists before filtering
        if 'dureeMois' in df.columns:
            df = df.drop(df[df['dureeMois'] > 900].index)
            df = df.dropna(subset=['dureeMois'])

    except Exception as e:
        print(f"Error in drop_outliers: {e}")
        return df
    return df


def filter_top_cpv_categories(df, top_n=40, cpv_column='codeCPV_2'):
    """
    Filter DataFrame to keep only rows with the top N most frequent CPV
    categories.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    top_n : int, default=40
        Number of top CPV categories to keep
    cpv_column : str, default='codeCPV_2'
        Name of the CPV column to filter on

    Returns
    -------
    pandas.DataFrame
        DataFrame filtered to top N CPV categories
    """
    try:
        # Check if CPV column exists
        if cpv_column not in df.columns:
            print(f"Warning: Column '{cpv_column}' not found in DataFrame")
            return df

        # Count occurrences of each CPV category
        cpv_group_counts = df[cpv_column].value_counts()

        # Get the top N categories
        top_groups = cpv_group_counts.nlargest(top_n)

        # Filter DataFrame to keep only top categories
        df_filtered = df[df[cpv_column].isin(top_groups.index)]

        print(f"Filtered from {len(cpv_group_counts)} to {len(top_groups)} "
              f"CPV categories, keeping {len(df_filtered)} rows out of "
              f"{len(df)}")

        return df_filtered

    except Exception as e:
        print(f"Error in filter_top_cpv_categories: {e}")
        return df


def cpv_2_3(df):
    """
    Add column with first 2 digits of CPV, and first 3 if it's 45 or 71.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        DataFrame with codeCPV_2_3 column
    """

    df.loc[:, 'codeCPV_2_3'] = df.apply(
        lambda row: (row['codeCPV_3'] if row['codeCPV_2'] in ['45000000',
                                                              '71000000'] else row['codeCPV_2']),
        axis=1
    )

    # Drop columns that exist
    # columns_to_drop = ['codeCPV_3', 'codeCPV_4', 'codeCPV_5']
    # existing_columns_to_drop = [col for col in columns_to_drop
    #                            if col in df.columns]
    # if existing_columns_to_drop:
    #    df.drop(columns=existing_columns_to_drop, inplace=True)

    return df


def annee(df):
    """
    Create an 'annee' column in datetime format from dateNotification.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        DataFrame with 'annee' column and filtered for years > 2018
    """
    try:
        if 'annee' not in df.columns:
            df = df[
                (df['dateNotification'].str[:4].astype(int) > 2018) &
                (df['dateNotification'].str[:4].astype(int) <= 2025)
            ]

            df.loc[:, 'annee'] = df['dateNotification'].str[:4].astype(int)
            # df['annee'] = pd.to_datetime(df['annee'], errors='ignore')

    except Exception as e:
        print(f"Error in annee: {e}")
        return df
    return df


def create_columns(df):
    """
    Run both cpv_2_3 and annee functions to create all necessary columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        DataFrame with CPV hierarchy and annee columns created
    """
    df = cpv_2_3(df)
    df = annee(df)
    return df

def sirene_features(data):
    """
    Ajoute au DataFrame `data` les colonnes :
    - 'acheteur_tranche_effectif' et 'acheteur_categorie' (via acheteur_siren)
    - 'titulaire_tranche_effectif' et 'titulaire_categorie' (via titulaire_siren)
    en utilisant le fichier sirene.csv.
    """
    sirene_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'sirene.csv'
        )

    # sirene_path = os.path.join(os.path.dirname(__file__, 'data', 'sirene.csv'))
    colonnes_a_charger = ['categorieEntreprise', 'siren', 'trancheEffectifsUniteLegale']

    sirene_df = pd.read_csv(sirene_path, usecols=colonnes_a_charger)
    sirene_df['siren'] = sirene_df['siren'].astype(str)

    # Pour les acheteurs
    data['acheteur_siren'] = data['acheteur_siren'].astype(str)
    data = data.merge(
        sirene_df.rename(columns={
            'trancheEffectifsUniteLegale': 'acheteur_tranche_effectif',
            'categorieEntreprise': 'acheteur_categorie'
        }),
        left_on='acheteur_siren',
        right_on='siren',
        how='left'
    )
    data = data.drop(columns=['siren'])

    # Pour les titulaires
    data['titulaire_siren'] = data['titulaire_siren'].astype(str)
    data = data.merge(
        sirene_df.rename(columns={
            'trancheEffectifsUniteLegale': 'titulaire_tranche_effectif',
            'categorieEntreprise': 'titulaire_categorie'
        }),
        left_on='titulaire_siren',
        right_on='siren',
        how='left'
    )
    data = data.drop(columns=['siren'])

    return data



def clean_data(save_csv=True, output_path=None):
    """
    Complete data cleaning pipeline that runs all cleaning functions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    save_csv : bool, default=True
        Whether to save the result to CSV
    output_path : str, optional
        Path where to save CSV file. If None, saves to
        '../data/cleaned_data.csv'

    Returns
    -------
    pandas.DataFrame
        Fully cleaned DataFrame
    """

    try:
        db_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'datalab.sqlite'
        )

        conn = sqlite3.connect(db_path)
        query = """
                SELECT *
                FROM "data.gouv.fr.2022.clean"
                """

        print(f"Loading data from database: {db_path}")
        df = pd.read_sql_query(query, conn)
        conn.close()

    except Exception as e:
        print(f"Error loading data from database: {e}")
        print("Please ensure the database file exists and is accessible.")
        raise

    print("Starting data cleaning pipeline...")

    # Process CPV data first (add descriptions and hierarchy)
    print("Processing CPV data...")
    df = process_cpv_data(df, levels=[2, 3, 4, 5])

    # Drop outliers
    print("Dropping outliers...")
    df = drop_outliers(df)

    # Create necessary columns
    print("Creating additional columns...")
    df = create_columns(df)

    # Create necessary columns
    print('add features workforce and category')
    df = sirene_features(df)

    # Filter to top CPV categories
    # print("Filtering to top CPV categories...")
    # df = filter_top_cpv_categories(df)

    # Save to CSV if requested
    if save_csv:
        try:
            if output_path is None:
                data_path = os.path.join(
                    os.path.dirname(__file__), '..', 'data'
                )
                output_path = os.path.join(data_path, 'data_clean.csv')

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save with index=False to avoid saving row indices
            df.to_csv(output_path, index=False)
            print(f"Data saved successfully to: {output_path}")

        except Exception as e:
            print(f"Warning: Failed to save CSV file: {e}")

    print("Data cleaning pipeline completed!")
    return df


if __name__ == "__main__":
    print("Starting data cleaning pipeline...")
    # Clean data with automatic CSV saving
    df_cleaned = clean_data()

    print(f"Processing completed. Final data shape: "
          f"{df_cleaned.shape}")
    print(f"Columns in final data: "
          f"{list(df_cleaned.columns)}")
