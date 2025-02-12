import os
import joblib
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ETL.preprocessing import ApplyTransforms
from engine.load_models import load_model  # Assurez-vous que cette fonction est dans engine.processing
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from typing import Optional
import pickle
from pathlib import Path  # Importation de pathlib pour gérer les chemins de manière plus propre

# Importer les transformers nécessaires depuis votre dossier de transformation
from ETL.transformers import MinMaxScalerTransformer, FrequencyEncoderTransformer, OneHotEncoderTransformer

# Charger les modèles sauvegardés
app = FastAPI()

# Configurer Jinja2 pour le rendu des templates HTML
templates = Jinja2Templates(directory="API/templates")

# Route pour afficher la page HTML
@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("interface.html", {"request": request})

# Charger les modèles au démarrage
extra_trees_model = load_model('extra_trees')
gradient_boosting_model = load_model('gradient_boosting')

# Définir la racine du chemin pour les fichiers
root_dir = Path("/Users/matthieu/Downloads/Tree-Model-Comparison-Regression-main")

# Charger les règles de transformation et les paramètres
def load_transformation_rules_and_params():
    # Construire le chemin du fichier pour les règles de transformation
    transformation_rules_path = root_dir / "transformer_params" / "transformation_rules.pkl"
    
    # Charger les règles de transformation
    with open(transformation_rules_path, "rb") as f:
        transformation_rules = pickle.load(f)

    # Charger les paramètres sauvegardés
    params = {
        'quantitative_max': joblib.load(root_dir / 'transformer_params' / 'quantitative_max_params.pkl'),
        'quantitative_min': joblib.load(root_dir / 'transformer_params' / 'quantitative_min_params.pkl'),
        'qualitative_max': joblib.load(root_dir / 'transformer_params' / 'qualitative_max_params.pkl'),
        'qualitative_min': joblib.load(root_dir / 'transformer_params' / 'qualitative_min_params.pkl')
    }

    return transformation_rules, params

transformation_rules, transformer_params = load_transformation_rules_and_params()
print(transformation_rules)
print(transformer_params)

class InputFeatures(BaseModel):
    MSSubClass: Optional[float] = None
    MSZoning: Optional[str] = None
    LotFrontage: Optional[float] = None
    LotArea: Optional[float] = None
    Street: Optional[str] = None
    Alley: Optional[str] = None
    LotShape: Optional[str] = None
    # Ajoutez toutes les autres caractéristiques ici

# Fonction pour transformer les données d'entrée avant prédiction en utilisant les règles d'entraînement et les paramètres sauvegardés
def transform_input_data(input_data: dict, df: pd.DataFrame, transformers: list, transformation_rules: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([input_data])  # Convertir l'entrée en DataFrame

    # Appliquer les transformations selon les règles et les paramètres sauvegardés
    try:
        # Initialisation des transformateurs
        transformers_dict = {name: transformer for name, transformer in transformers}

        print("\nTransformation des colonnes quantitatives_max (MinMax Scaling)...")
        # Appliquer les transformations pour les colonnes quantitatives_max (MinMax Scaling)
        for column in transformation_rules['quantitatives_max']:
            if column in input_data:
                print(f"Avant MinMaxScaler pour la colonne {column}: {input_data[column]}")
                transformer = transformers_dict['quantitative_max']
                input_df[column] = transformer.transform([[input_data[column]]])[0][0]
                print(f"Après MinMaxScaler pour la colonne {column}: {input_df[column]}")

        print("\nTransformation des colonnes quantitatives_min (Frequency Encoding)...")
        # Appliquer les transformations pour les colonnes quantitatives_min (Frequency Encoding)
        for column in transformation_rules['quantitatives_min']:
            if column in input_data:
                print(f"Avant Frequency Encoding pour la colonne {column}: {input_data[column]}")
                transformer = transformers_dict['quantitative_min']
                input_df = transformer.transform(input_df)
                print(f"Après Frequency Encoding pour la colonne {column}: {input_df[column]}")

        print("\nTransformation des colonnes qualitatives_min (One-Hot Encoding)...")
        # Appliquer One-Hot Encoding pour les colonnes qualitatives_min
        for column in transformation_rules['qualitatives_min']:
            if column in input_data:
                print(f"Avant One-Hot Encoding pour la colonne {column}: {input_data[column]}")
                transformer = transformers_dict['qualitative_min']
                input_df = transformer.transform(input_df)
                print(f"Après One-Hot Encoding pour la colonne {column}: {input_df.head()}")

        print("\nTransformation des colonnes qualitatives_max (Frequency Encoding)...")
        # Appliquer Frequency Encoding pour les colonnes qualitatives_max
        for column in transformation_rules['qualitatives_max']:
            if column in input_data:
                print(f"Avant Frequency Encoding pour la colonne {column}: {input_data[column]}")
                transformer = transformers_dict['qualitative_max']
                input_df = transformer.transform(input_df)
                print(f"Après Frequency Encoding pour la colonne {column}: {input_df[column]}")

        # Retourner les données transformées
        return input_df

    except Exception as e:
        print(f"Transformation Error: {e}")
        raise HTTPException(status_code=500, detail="Error during transformation")

# Route pour la prédiction avec ExtraTrees
expected_columns = [
    "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", 
    "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
    "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", 
    "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond", "Foundation", 
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", 
    "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC", "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", 
    "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", 
    "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "FireplaceQu", "GarageType", 
    "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond", "PavedDrive", 
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "PoolQC", "Fence", 
    "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition"
]

@app.post("/predict/")
async def predict(input_data: dict):
    try:
        print("\nDonnées d'entrée:", input_data)

        # Compléter les données d'entrée avec NaN pour les colonnes manquantes
        input_data = {key: value for key, value in input_data.items() if key in expected_columns}
        for col in expected_columns:
            if col not in input_data:
                input_data[col] = None  # Remplir avec None pour la transformation suivante

        # Transformer les données reçues en DataFrame
        input_df = pd.DataFrame([input_data])

        # Remplacer les valeurs None par NaN pour chaque colonne
        input_df = input_df.apply(lambda x: x.map(lambda val: np.nan if val is None else val))

        print("\nDonnées après préparation (avant transformation):", input_df)

        # Initialiser les transformateurs
        transformers = [
            ('quantitative_max', MinMaxScalerTransformer(cols=transformation_rules["quantitatives_max"])),
            ('quantitative_min', FrequencyEncoderTransformer(cols=transformation_rules["quantitatives_min"])),
            ('qualitative_min', OneHotEncoderTransformer(cols=transformation_rules["qualitatives_min"])),
            ('qualitative_max', FrequencyEncoderTransformer(cols=transformation_rules["qualitatives_max"]))
        ]

        # Appliquer les transformations
        transformed_data = transform_input_data(input_data, input_df, transformers, transformation_rules)

        # Vérifier que les données transformées sont prêtes pour la prédiction
        if transformed_data.isnull().any().any():
            print("Avertissement: Certaines valeurs manquantes restent après transformation.")

        # Choisir le modèle en fonction de l'option sélectionnée
        model_selection = input_data.get('model_selection')
        if model_selection == 'extra_trees':
            print("\nUtilisation du modèle ExtraTrees pour la prédiction.")
            prediction = extra_trees_model.predict(transformed_data)
        elif model_selection == 'gradient_boosting':
            print("\nUtilisation du modèle GradientBoosting pour la prédiction.")
            prediction = gradient_boosting_model.predict(transformed_data)
        else:
            raise HTTPException(status_code=400, detail="Modèle inconnu")

        # Retourner la prédiction
        print(f"Prédiction: {prediction[0]}")
        return {"prediction": prediction[0]}

    except Exception as e:
        print(f"Erreur survenue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
