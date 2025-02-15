import pickle
import os

def load_model(model_name: str):
    model_path = f"engine/models/{model_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError(f"The model {model_name} couldn't be found")
