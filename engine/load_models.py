import joblib
import os

def load_model(model_name: str):
    model_path = f"engine/models/{model_name}.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Le modèle {model_name} n'a pas été trouvé!")