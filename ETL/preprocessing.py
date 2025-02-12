from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from ETL.transformers import MinMaxScalerTransformer, FrequencyEncoderTransformer, OneHotEncoderTransformer
import pickle
import os

def find_transforms(df):
    colonnes_quantitatives = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    colonnes_qualitatives = df.select_dtypes(include=['object']).columns.tolist()

    colonnes_quantitatives_min = [col for col in colonnes_quantitatives if df[col].nunique() <= 130]
    colonnes_quantitatives_max = [col for col in colonnes_quantitatives if df[col].nunique() > 130]
    colonnes_qualitatives_min = [col for col in colonnes_qualitatives if df[col].nunique() <= 5]
    colonnes_qualitatives_max = [col for col in colonnes_qualitatives if df[col].nunique() > 5]

    transformation_rules = {
        "quantitatives_min": colonnes_quantitatives_min,
        "quantitatives_max": colonnes_quantitatives_max,
        "qualitatives_min": colonnes_qualitatives_min,
        "qualitatives_max": colonnes_qualitatives_max
    }
    return transformation_rules

def apply_transforms(df, transformation_rules):
    transformers = [
        ('quantitative_max', MinMaxScalerTransformer(cols=transformation_rules["quantitatives_max"])),
        ('quantitative_min', FrequencyEncoderTransformer(cols=transformation_rules["quantitatives_min"])),
        ('qualitative_min', OneHotEncoderTransformer(cols=transformation_rules["qualitatives_min"])),
        ('qualitative_max', FrequencyEncoderTransformer(cols=transformation_rules["qualitatives_max"]))
    ]
    
    column_transformer = Pipeline(steps=transformers)
    
    return column_transformer.fit_transform(df)

class ApplyTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_rules=None, save_dir="./transformer_params"):
        self.transformation_rules = transformation_rules
        self.transformers_ = []
        self.save_dir = save_dir  # Répertoire pour sauvegarder les paramètres et les règles

    def fit(self, X, y=None):
        if self.transformation_rules is None:
            self.transformation_rules = find_transforms(X)
        
        self.transformers_ = [
            ('quantitative_max', MinMaxScalerTransformer(cols=self.transformation_rules["quantitatives_max"])),
            ('quantitative_min', FrequencyEncoderTransformer(cols=self.transformation_rules["quantitatives_min"])),
            ('qualitative_min', OneHotEncoderTransformer(cols=self.transformation_rules["qualitatives_min"])),
            ('qualitative_max', FrequencyEncoderTransformer(cols=self.transformation_rules["qualitatives_max"]))
        ]

        # Créer le dossier de sauvegarde s'il n'existe pas
        os.makedirs(self.save_dir, exist_ok=True)

        # Sauvegarder les règles de transformation
        rules_path = os.path.join(self.save_dir, "transformation_rules.pkl")
        with open(rules_path, "wb") as f:
            pickle.dump(self.transformation_rules, f)
        print(f"Transformation rules saved at: {rules_path}")

        return self

    def transform(self, X):
        if self.transformation_rules is None:
            raise ValueError("Les règles de transformation ne sont pas définies.")

        transformed_data = X.copy()
        for name, transformer in self.transformers_:
            transformed_data = transformer.fit_transform(transformed_data)
            transformer.save_params(os.path.join(self.save_dir, f"{name}_params.pkl"))

        return transformed_data

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
