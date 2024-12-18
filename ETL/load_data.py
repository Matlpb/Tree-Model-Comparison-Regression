import os
import sys
from pathlib import Path
import zipfile
import pandas as pd
from typing import Optional, Tuple

sys.path.append(str(Path(os.path.abspath('')).resolve().parents[0]))

def ensure_directory_exists(directory: str) -> None:
    """
    Assure que le répertoire spécifié existe. Le crée s'il n'existe pas.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_and_extract_kaggle_data(competition: str, output_dir: str) -> None:
    """
    Télécharge et extrait les fichiers de compétition dans le répertoire spécifié.
    """
    os.system(f"kaggle competitions download -c {competition} -p {output_dir}")
    
    zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
    for zip_file in zip_files:
        zip_path = os.path.join(output_dir, zip_file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV et retourne un DataFrame.
    """
    return pd.read_csv(file_path)

def load_dataframes(base_dir: str, competition_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Télécharge, extrait les données de la compétition Kaggle et charge 
    les fichiers 'train.csv' et 'test.csv' dans des DataFrames.
    """
    data_dir = os.path.join(base_dir, "data")  # Sous-dossier "data"
    ensure_directory_exists(data_dir)
    download_and_extract_kaggle_data(competition_name, data_dir)

    train_csv_path = os.path.join(data_dir, "train.csv")
    test_csv_path = os.path.join(data_dir, "test.csv")

    df_train, df_test = None, None
    if os.path.exists(train_csv_path):
        df_train = load_csv(train_csv_path)
    if os.path.exists(test_csv_path):
        df_test = load_csv(test_csv_path)

    return df_train, df_test

def one_hot_encode_and_fill_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique un encodage one-hot sur les colonnes catégoriques, remplit les valeurs manquantes avec 0
    et transforme les colonnes booléennes en 0 et 1.
    """
    df_copy = df.copy()
    df_copy = pd.get_dummies(df_copy, drop_first=True)
    bool_columns = df_copy.select_dtypes(include=[bool]).columns
    df_copy[bool_columns] = df_copy[bool_columns].astype(int)
    df_copy = df_copy.fillna(0)
    return df_copy
