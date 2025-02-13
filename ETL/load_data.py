import os
import sys
from pathlib import Path
import zipfile
import pandas as pd
from typing import Optional, Tuple

sys.path.append(str(Path(os.path.abspath('')).resolve().parents[0]))

def ensure_directory_exists(directory: str) -> None:
    """
    Ensures that the specified directory exists. Creates it if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_and_extract_kaggle_data(competition: str, output_dir: str) -> None:
    """
    Downloads and extracts Kaggle competition files into the specified directory.
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
    Loads a CSV file and returns a DataFrame.
    """
    return pd.read_csv(file_path)

def load_dataframes(base_dir: str, competition_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Downloads, extracts Kaggle competition data and loads 'train.csv' and 'test.csv' files into DataFrames.
    """
    data_dir = os.path.join(base_dir, "data")
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
