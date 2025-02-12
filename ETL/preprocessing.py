from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from ETL.transformers import (
    MinMaxScalerTransformer,
    FrequencyEncoderTransformer,
    OneHotEncoderTransformer,
)
import pickle
import os
import numpy as np 
import pandas as pd


def find_transforms(df):
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


def apply_transforms(df, transformation_rules):
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
    def __init__(
        self, 
        saving_mode=False, 
        save_dir: str = "./transformer_params"
    ):
        self.transformation_rules = None
        self.quantitative_min_params = None
        self.quantitative_max_params = None
        self.qualitative_min_params = None
        self.qualitative_max_params = None
        self.transformers_ = []
        self.saving_mode = saving_mode 
        self.save_dir = save_dir

    def fit(self, X, y=None):
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

    def save_transformation_params(self, X):
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
        # Save transformer parameters
        for name, transformer in self.transformers_:
            transformer.fit(X)  # Entraînement du transformateur
            transformer.save_params(os.path.join(self.save_dir, f"{name}_params.pkl"))

    def load_transformation_params(self):
        rules_path = os.path.join(self.save_dir, "transformation_rules.pkl")
        if os.path.exists(rules_path):
            with open(rules_path, "rb") as f:
                self.transformation_rules = pickle.load(f)
            print(f"Transformation rules loaded from: {rules_path}")

        # Load parameters for transformers
        self.quantitative_min_params = self.load_params_from_file("quantitative_min_params.pkl")
        self.quantitative_max_params = self.load_params_from_file("quantitative_max_params.pkl")
        self.qualitative_min_params = self.load_params_from_file("qualitative_min_params.pkl")
        self.qualitative_max_params = self.load_params_from_file("qualitative_max_params.pkl")

    def load_params_from_file(self, filename):
        params_path = os.path.join(self.save_dir, filename)
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            print(f"Loaded parameters from {params_path}")
            return params
        else:
            print(f"Parameter file {params_path} not found.")
            return None

    def transform(self, X):

        # If in saving_mode, just apply the transformation using the rules
        if self.saving_mode:
            transformed_data = apply_transforms(X, self.transformation_rules)
        else:
            transformed_data = X.copy()

            # For each transformation rule, apply transformations accordingly
            for name in self.transformation_rules:
                cols = self.transformation_rules[name]
                
                if name == "quantitatives_max":
    # Appliquer MinMaxScaler pour chaque colonne
                    for col in cols:
                        min_val = self.quantitative_max_params.get("X_min", {}).get(col, None)
                        max_val = self.quantitative_max_params.get("X_max", {}).get(col, None)
        
                        if min_val is not None and max_val is not None:
                            print(f"Applying MinMaxScaler to column: {col} with min: {min_val},{type(min_val)}, max: {max_val},{type(max_val)}")
            
            # Créer un scaler pour une seule colonne
                            scaler = MinMaxScalerTransformer(cols=[col], feature_range=(0, 1), X_min={col: min_val}, X_max={col: max_val})
            
            # Transformer la colonne seule
                            transformed_data[col] = scaler.fit_transform(X[[col]])[col]
                        else:
                            print(f"MinMaxScaler params for column {col} not found.")
                            transformed_data[col]=0
                        print(f"Applying MinMaxScaler to column: {col} endoded from {X[col].values[0]} to {transformed_data[col]}")


                elif name == "qualitatives_min":
                    # Apply OneHotEncoder for qualitative columns
                    for col in cols:
                        categories = self.qualitative_min_params.get(col, [])
                        if categories is not None:
                            print(f"OneHotEncoding column: {col} with categories: {categories}")
                            onehot = OneHotEncoderTransformer(categories=[categories], sparse=False)
                            onehot_transformed = onehot.fit_transform(X[[col]])
                            transformed_data.append(onehot_transformed)
                        else:
                            print(f"Categories for column {col} not found.")
                elif name == "quantitatives_min" or name == "qualitatives_max":
                    for col in cols:
                        print(f"🔄 Applying FrequencyEncoder to column: {col}")

                        encoding_map = self.quantitative_min_params.get(col, None)
                        print(encoding_map)
                        if encoding_map is not None:
                            print(f"📌 Encoding map for {col}: {encoding_map}")

                            value = X[col].values[0]

            # Vérifier si la valeur est NaN
                            if pd.isna(value):
                                transformed_value = 0  # Si la valeur est NaN, on la remplace par 0
                                print(f"⚠️ La valeur est NaN, {col} encodée en 0.")
                            else:
                                try:
                                    value = int(value)  # Convertir la valeur en entier
                                except ValueError:
                    # Si la conversion échoue, on remplace également par 0
                                    transformed_value = 0
                                    print(f"⚠️ La valeur {value} n'a pas pu être convertie en entier, {col} encodée en 0.")
                                else:
                                    if value in encoding_map:
                                        transformed_value = encoding_map[value]
                                        print(f"✅ {col} transformed: Original Value = {value} → Encoded Value = {transformed_value}")
                                    else:
                                        transformed_value = 0  # Si la valeur n'est pas dans le dictionnaire
                                        print(f"⚠️ No mapping found for {col}, replacing {value} with 0.")

            # Ajouter la valeur transformée dans le DataFrame final
                            transformed_data[col] = transformed_value

                        else:
                            print(f"⚠️ No mapping found for {col}, replacing with 0.")
                            transformed_data[col] = 0


        return transformed_data
                

        
     

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
