from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from ETL.transformers import (
    MinMaxScalerTransformer,
    FrequencyEncoderTransformer,
    OneHotEncoderTransformer,
)
import pickle
import os


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
        transformation_rules=None, 
        saving_mode=False, 
        save_dir: str = "./transformer_params"
    ):
        self.transformation_rules = transformation_rules
        self.quantitative_min_params = None  # Variable pour les paramètres quantitatifs_min
        self.quantitative_max_params = None  # Variable pour les paramètres quantitatifs_max
        self.qualitative_min_params = None  # Variable pour les paramètres qualitatifs_min
        self.qualitative_max_params = None  # Variable pour les paramètres qualitatifs_max
        self.transformers_ = []
        self.saving_mode = saving_mode 
        self.save_dir = save_dir

    def fit(self, X, y=None):
        if self.saving_mode:
            self.save_transformation_params()
        else:
            self.load_transformation_params()

        self.transformers_ = [
            ("quantitative_max", MinMaxScalerTransformer(cols=self.transformation_rules["quantitatives_max"])),
            ("quantitative_min", FrequencyEncoderTransformer(cols=self.transformation_rules["quantitatives_min"])),
            ("qualitative_min", OneHotEncoderTransformer(cols=self.transformation_rules["qualitatives_min"])),
            ("qualitative_max", FrequencyEncoderTransformer(cols=self.transformation_rules["qualitatives_max"]))
        ]
        return self

    def save_transformation_params(self):
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Sauvegarder les règles de transformation
        rules_path = os.path.join(self.save_dir, "transformation_rules.pkl")
        with open(rules_path, "wb") as f:
            pickle.dump(self.transformation_rules, f)
        print(f"Transformation rules saved at: {rules_path}")
        
        # Sauvegarder les paramètres des transformateurs
        for name, transformer in self.transformers_:
            transformer.save_params(os.path.join(self.save_dir, f"{name}_params.pkl"))

    def load_transformation_params(self):
        rules_path = os.path.join(self.save_dir, "transformation_rules.pkl")
        if os.path.exists(rules_path):
            with open(rules_path, "rb") as f:
                self.transformation_rules = pickle.load(f)
            print(f"Transformation rules loaded from: {rules_path}")

        # Charger les paramètres des transformateurs dans des variables spécifiques
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
        if self.transformation_rules is None:
            raise ValueError("Les règles de transformation ne sont pas définies.")

        transformed_data = X.copy()
        for name, transformer in self.transformers_:
            if self.saving_mode:
                transformed_data = transformer.fit_transform(transformed_data)
                transformer.save_params(os.path.join(self.save_dir, f"{name}_params.pkl"))
            else:
                params_path = os.path.join(self.save_dir, f"{name}_params.pkl")
                if os.path.exists(params_path):
                    params=transformer.load_params(params_path)
                    print(f"Loaded parameters for {name} from {params_path}")

                if name == "quantitative_max":
                    for col in self.transformation_rules["quantitatives_max"]:

                        min_val = params.get("X_min", {}).get(col, None)
                        max_val = params.get("X_max", {}).get(col, None)

                        if min_val is not None and max_val is not None:
                            transformed_data[col] = (transformed_data[col] - min_val) / (max_val - min_val)
                    
                elif name == "qualitative_min":
                    for col in self.transformation_rules["qualitatives_min"]:
                        if col in params:
                            categories = params[col]
                            for category in categories:
                            # Ajouter une colonne pour chaque catégorie possible
                                transformed_data[f"{col}_{category}"] = (transformed_data[col] == category).astype(int)
                            transformed_data.drop(columns=[col], inplace=True)
                        
                    
                    
                

                transformed_data = transformer.transform(transformed_data)

        return transformed_data

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)