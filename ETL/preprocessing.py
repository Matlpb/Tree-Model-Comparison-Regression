from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from ETL.transformers import (
    MinMaxScalerTransformer,
    FrequencyEncoderTransformer,
    OneHotEncoderTransformer,
)
import pickle
import os
import pandas as pd
from typing import Dict, List


def find_transforms(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Determine which transformation rules to apply based on the columns' types and cardinalities in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    Dict[str, List[str]]: A dictionary with transformation rules for quantitative and qualitative columns.
    """
    colonnes_quantitatives = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    colonnes_qualitatives = df.select_dtypes(include=["object"]).columns.tolist()

    colonnes_quantitatives_min = [
        col for col in colonnes_quantitatives if df[col].nunique() <= 130
    ]
    colonnes_quantitatives_max = [
        col for col in colonnes_quantitatives if df[col].nunique() > 130
    ]
    colonnes_qualitatives_min = [
        col for col in colonnes_qualitatives if df[col].nunique() <= 5
    ]
    colonnes_qualitatives_max = [
        col for col in colonnes_qualitatives if df[col].nunique() > 5
    ]

    transformation_rules = {
        "quantitatives_min": colonnes_quantitatives_min,
        "quantitatives_max": colonnes_quantitatives_max,
        "qualitatives_min": colonnes_qualitatives_min,
        "qualitatives_max": colonnes_qualitatives_max,
    }
    return transformation_rules


def apply_transforms(df: pd.DataFrame, transformation_rules: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Apply transformations based on the provided rules for the quantitative and qualitative columns.

    Parameters:
    df (pd.DataFrame): The input dataframe to be transformed.
    transformation_rules (Dict[str, List[str]]): The transformation rules to be applied.

    Returns:
    pd.DataFrame: The transformed dataframe.
    """
    transformers = [
        (
            "quantitative_max",
            MinMaxScalerTransformer(cols=transformation_rules["quantitatives_max"]),
        ),
        (
            "quantitative_min",
            FrequencyEncoderTransformer(cols=transformation_rules["quantitatives_min"]),
        ),
        (
            "qualitative_min",
            OneHotEncoderTransformer(cols=transformation_rules["qualitatives_min"]),
        ),
        (
            "qualitative_max",
            FrequencyEncoderTransformer(cols=transformation_rules["qualitatives_max"]),
        ),
    ]

    column_transformer = Pipeline(steps=transformers)

    return column_transformer.fit_transform(df)


class ApplyTransforms(BaseEstimator, TransformerMixin):
    """
    A custom transformer to apply various transformations to the input data based on predefined rules.
    Supports saving and loading transformation parameters for reuse.

    Attributes:
    saving_mode (bool): If True, saves the transformation parameters after fitting.
    save_dir (str): Directory to save or load transformation parameters.
    transformation_rules (Dict[str, List[str]]): The transformation rules.
    transformers_ (List[tuple]): A list of transformers for the transformations.
    """
    def __init__(
        self, 
        saving_mode: bool = False, 
        save_dir: str = "./ETL/transformer_params"
    ):
        """
        Initializes the ApplyTransforms class with saving_mode and directory for saving transformer parameters.

        Parameters:
        saving_mode (bool): If True, saves the 
        transformation parameters If False it load transformer_params and encode X test with respect to the rules.
        save_dir (str): Directory to save or load transformation parameters.
        """
        self.transformation_rules: Dict[str, List[str]] = None
        self.quantitative_min_params = None
        self.quantitative_max_params = None
        self.qualitative_min_params = None
        self.qualitative_max_params = None
        self.transformers_: List[tuple] = []
        self.saving_mode = saving_mode
        self.save_dir = save_dir

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "ApplyTransforms":
        """
        Fit the transformer by saving or loading transformation parameters.

        Parameters:
        X (pd.DataFrame): The input dataframe for fitting.
        y (pd.Series, optional): Target variable (not used).

        Returns:
        ApplyTransforms: The fitted transformer.
        """
        if self.saving_mode:
            self.save_transformation_params(X)
        else:
            self.load_transformation_params()

        self.transformers_ = [
            ("quantitative_max", MinMaxScalerTransformer(cols=self.transformation_rules["quantitatives_max"])),
            ("quantitative_min", FrequencyEncoderTransformer(cols=self.transformation_rules["quantitatives_min"])),
            ("qualitative_min", OneHotEncoderTransformer(cols=self.transformation_rules["qualitatives_min"])),
            ("qualitative_max", FrequencyEncoderTransformer(cols=self.transformation_rules["qualitatives_max"]))
        ]
        return self

    def save_transformation_params(self, X: pd.DataFrame) -> None:
        """
        Save the transformation parameters to files.

        Parameters:
        X (pd.DataFrame): The input dataframe to derive transformation rules.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.transformation_rules = find_transforms(X)
        rules_path = os.path.join(self.save_dir, "transformation_rules.pkl")
        with open(rules_path, "wb") as f:
            pickle.dump(self.transformation_rules, f)
        print(f"Transformation rules saved at: {rules_path}")
        
        self.transformers_ = [
            ("quantitative_max", MinMaxScalerTransformer(cols=self.transformation_rules["quantitatives_max"])),
            ("quantitative_min", FrequencyEncoderTransformer(cols=self.transformation_rules["quantitatives_min"])),
            ("qualitative_min", OneHotEncoderTransformer(cols=self.transformation_rules["qualitatives_min"])),
            ("qualitative_max", FrequencyEncoderTransformer(cols=self.transformation_rules["qualitatives_max"]))
        ]
        
        for name, transformer in self.transformers_:
            transformer.fit(X)
            transformer.save_params(os.path.join(self.save_dir, f"{name}_params.pkl"))

    def load_transformation_params(self) -> None:
        """
        Load transformation parameters from saved files.

        """
        rules_path = os.path.join(self.save_dir, "transformation_rules.pkl")
        if os.path.exists(rules_path):
            with open(rules_path, "rb") as f:
                self.transformation_rules = pickle.load(f)
            print(f"Transformation rules loaded from: {rules_path}")

        self.quantitative_min_params = self.load_params_from_file("quantitative_min_params.pkl")
        self.quantitative_max_params = self.load_params_from_file("quantitative_max_params.pkl")
        self.qualitative_min_params = self.load_params_from_file("qualitative_min_params.pkl")
        self.qualitative_max_params = self.load_params_from_file("qualitative_max_params.pkl")

    def load_params_from_file(self, filename: str) -> Dict:
        """
        Load transformation parameters from a specified file.

        Parameters:
        filename (str): The name of the parameter file to load.

        Returns:
        Dict: The loaded parameters.
        """
        params_path = os.path.join(self.save_dir, filename)
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            print(f"Loaded parameters from {params_path}")
            return params
        else:
            print(f"Parameter file {params_path} not found.")
            return {}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the appropriate transformations to the input dataframe.

        Parameters:
        X (pd.DataFrame): The input dataframe to be transformed.

        Returns:
        pd.DataFrame: The transformed dataframe.
        If saving_mode is True, the transformed dataframe is returned.
        If saving_mode is False, the transformed dataframe is modified in 
        place with respect to the transformation rules and all transformer_params.
        """
        if self.saving_mode:
            transformed_data = apply_transforms(X, self.transformation_rules)
            return transformed_data
        else:
            transformed_data = X.copy()

            for name in self.transformation_rules:
                cols = self.transformation_rules[name]
                
                if name == "quantitatives_max":
                    for col in cols:
                        min_val = self.quantitative_max_params.get("X_min", {}).get(col, None)
                        max_val = self.quantitative_max_params.get("X_max", {}).get(col, None)
        
                        if min_val is not None and max_val is not None:

                            scaler = MinMaxScalerTransformer(cols=[col], feature_range=(0, 1), X_min={col: min_val}, X_max={col: max_val})
                            transformed_data[col] = scaler.fit_transform(X[[col]])[col]
                        else:
                            transformed_data[col]=0

                elif name == "qualitatives_min":
                    transformed_one_hot = pd.DataFrame()

                    for col in cols:
                        categories = self.qualitative_min_params.get(col, [])
                        
                        if categories:
                            temp_df = pd.DataFrame(0, index=X.index, columns=[categories])
                            value = X[col].values[0]
                            
                            if pd.notna(value) and f"{col}_{value}" in temp_df.columns:
                                temp_df[f"{col}_{value}"] = 1
                            
                            transformed_one_hot = pd.concat([transformed_one_hot, temp_df], axis=1)
                            
                            transformed_data.drop(columns=[col], inplace=True)


                elif name == "quantitatives_min" or name == "qualitatives_max":
                    for col in cols:
                        encoding_map = self.quantitative_min_params.get(col, None)
                        value = X[col].values[0]

                        if encoding_map and not pd.isna(value):
                            try:
                                value = int(value)
                                transformed_value = encoding_map.get(value, 0)
                            except ValueError:
                                transformed_value = 0
                        else:
                            transformed_value = 0

                        transformed_data[col] = transformed_value

            transformed_one_hot.columns = [col[0] if isinstance(col, tuple) else col for col in transformed_one_hot.columns]
            final_transformed_X = pd.concat([transformed_data, transformed_one_hot], axis=1)
            
            return final_transformed_X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
       
        self.fit(X, y)
        return self.transform(X)
