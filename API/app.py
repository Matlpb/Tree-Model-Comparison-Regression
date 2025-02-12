import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ETL.preprocessing import ApplyTransforms
from engine.load_models import load_model
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from typing import Optional
from pathlib import Path  

from ETL.transformers import MinMaxScalerTransformer, FrequencyEncoderTransformer, OneHotEncoderTransformer

# Initialiser l'application FastAPI
app = FastAPI()

# Configurer Jinja2 pour le rendu des templates HTML
templates = Jinja2Templates(directory="API/templates")

@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("interface.html", {"request": request})

# Charger les modèles au démarrage
extra_trees_model = load_model('extra_trees')
gradient_boosting_model = load_model('gradient_boosting')

# Définir le chemin des fichiers
root_dir = Path("/Users/matthieu/Downloads/Tree-Model-Comparison-Regression-main")

# Initialiser le transformateur en mode lecture seule
transformer = ApplyTransforms(
    save_dir=str(root_dir / "transformer_params"),
    saving_mode=False  
)

class InputFeatures(BaseModel):
    MSSubClass: Optional[float] = None
    MSZoning: Optional[str] = None
    LotFrontage: Optional[float] = None
    LotArea: Optional[float] = None
    Street: Optional[str] = None
    Alley: Optional[str] = None
    LotShape: Optional[str] = None
    # Ajouter les autres caractéristiques ici
train_columns = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
    'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
    'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 
    'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
    'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
    'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 
    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 
    'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
    'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 
    'SaleType', 'SaleCondition'
]
# Fonction pour transformer les données d'entrée
def transform_input_data(input_data: dict) -> pd.DataFrame:
    try:
        print("\nÉtape 1: Conversion en DataFrame")
        input_df = pd.DataFrame([input_data], columns=train_columns)
        print("Données originales:", input_df)

        # Remplacer les valeurs None ou NaN par 0
        print("\nÉtape 2: Remplacement des NaN par 0")
        #input_df = input_df.fillna(0)
        print("Données après remplacement des NaN:", input_df)
        has_nan = input_df.isna().any().any()

        print("Données après remplacement des NaN et chaînes vides:", input_df)
        print(f"Présence de NaN après transformation : {has_nan}")

        # Transformation avec ApplyTransforms
        print("\nÉtape 3: Application des transformations")
        transformed_df = transformer.fit_transform(input_df)
        transformed_df = pd.DataFrame(transformed_df)

        print("Données transformées:", transformed_df)

        return transformed_df

    except Exception as e:
        print(f"\n❌ Erreur lors de la transformation: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la transformation")

# Route pour la prédiction
@app.post("/predict/")
async def predict(input_data: dict):
    try:
        print("\n🔹 Nouvelle requête reçue:", input_data)

        transformed_data = transform_input_data(input_data)

        model_selection = input_data.get('model_selection')
        if model_selection == 'extra_trees':
            print("\nUtilisation du modèle ExtraTrees pour la prédiction.")
            prediction = extra_trees_model.predict(transformed_data)
        elif model_selection == 'gradient_boosting':
            print("\nUtilisation du modèle GradientBoosting pour la prédiction.")
            prediction = gradient_boosting_model.predict(transformed_data)
        else:
            raise HTTPException(status_code=400, detail="Modèle inconnu")

        print(f"\n✅ Prédiction: {prediction[0]}")
        return {"prediction": prediction[0]}
    
    except Exception as e:
        print(f"\n❌ Erreur survenue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
