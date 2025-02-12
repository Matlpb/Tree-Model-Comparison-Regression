import os
import joblib
from fastapi import FastAPI, Form, HTTPException
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

# Route pour afficher la page HTML
@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("interface.html", {"request": request})

# Charger les modèles au démarrage
extra_trees_model = load_model('extra_trees')
gradient_boosting_model = load_model('gradient_boosting')

# Définir le chemin des fichiers
root_dir = Path("/Users/matthieu/Downloads/Tree-Model-Comparison-Regression-main")

# Appliquer ApplyTransforms sans avoir besoin de charger manuellement les règles et paramètres
transformer = ApplyTransforms(
    save_dir=str(root_dir / "transformer_params"),
    saving_mode=False  # Lecture seule des paramètres
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

# Fonction pour transformer les données d'entrée
def transform_input_data(input_data: dict) -> pd.DataFrame:
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df.applymap(lambda val: np.nan if val is None else val)
        transformed_df = transformer.fit_transform(input_df)

        if transformed_df.isnull().any().any():
            print("Attention: Certaines valeurs sont manquantes après transformation.")

        return transformed_df
    except Exception as e:
        print(f"Transformation Error: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la transformation")

# Route pour la prédiction
@app.post("/predict/")
async def predict(input_data: dict):
    try:
        print("\nDonnées d'entrée:", input_data)

        transformed_data = transform_input_data(input_data)

        if transformed_data.isnull().any().any():
            print("Avertissement: Certaines valeurs manquantes restent après transformation.")

        model_selection = input_data.get('model_selection')
        if model_selection == 'extra_trees':
            print("\nUtilisation du modèle ExtraTrees pour la prédiction.")
            prediction = extra_trees_model.predict(transformed_data)
        elif model_selection == 'gradient_boosting':
            print("\nUtilisation du modèle GradientBoosting pour la prédiction.")
            prediction = gradient_boosting_model.predict(transformed_data)
        else:
            raise HTTPException(status_code=400, detail="Modèle inconnu")

        print(f"Prédiction: {prediction[0]}")
        return {"prediction": prediction[0]}
    
    except Exception as e:
        print(f"Erreur survenue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
