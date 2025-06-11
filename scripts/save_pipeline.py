import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

from scripts.preprocess_pipeline import create_pipeline
from scripts.data_cleaner import filter_top_cpv_categories

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

def save_montant_prediction_pipeline():
    """Create and save the pipeline for montant prediction."""
    # Estimation du montant
    numerical_columns = ['dureeMois', 'offresRecues', 'annee']

    binary_columns = ['sousTraitanceDeclaree', 'origineFrance',
                      'marcheInnovant', 'idAccordCadre']

    categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                           'typeGroupementOperateurs', 'tauxAvance_cat',
                           'codeCPV_3', 'acheteur_tranche_effectif',
                           'acheteur_categorie']

    pipeline_pred_montant = create_pipeline(numerical_columns,
                                            binary_columns,
                                            categorical_columns)

    #on fit le model
    df = pd.read_csv(os.path.join(data_path,'data_clean.csv'))
    df = filter_top_cpv_categories(df, top_n=150, cpv_column='codeCPV_3')
    df.drop(df[df['montant'] > 999_999].index, inplace=True)
    X = df
    X[['acheteur_tranche_effectif', 'acheteur_categorie']] = X[['acheteur_tranche_effectif', 'acheteur_categorie']].fillna('null')
    X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=0, stratify=X['codeCPV_3'])

    pipeline_pred_montant.fit(X_train)

    # Save the pipeline to a file
    with open(os.path.join(data_path, 'pipeline_pred_montant.pkl'),
              'wb') as f:
        pickle.dump(pipeline_pred_montant, f)

    print("Pipeline for montant prediction saved successfully.")


def save_marche_similaire_pipeline():
    """Create and save the pipeline for similar marches."""
    # Marchés similaires
    numerical_columns = ['montant', 'dureeMois', 'offresRecues']
    binary_columns = ['sousTraitanceDeclaree', 'origineFrance',
                      'marcheInnovant', 'idAccordCadre']
    categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                           'typeGroupementOperateurs', 'tauxAvance_cat',
                           'codeCPV_2_3']

    pipeline_marche_sim = create_pipeline(numerical_columns,
                                          binary_columns,
                                          categorical_columns)

    #fit
    df = pd.read_csv(os.path.join(data_path,'data_clean.csv'))
    df_cpv = filter_top_cpv_categories(df, top_n=50, cpv_column='codeCPV_2_3')

    pipeline_marche_sim.fit(df_cpv)
    # Save the pipeline to a file
    with open(os.path.join(data_path, 'pipeline_marche_sim.pkl'),
              'wb') as f:
        pickle.dump(pipeline_marche_sim, f)

    print("Pipeline for similar marches saved successfully.")


if __name__ == "__main__":
    print("Creating and saving pipelines...")
    save_montant_prediction_pipeline()
    save_marche_similaire_pipeline()
    print("All pipelines saved successfully!")
