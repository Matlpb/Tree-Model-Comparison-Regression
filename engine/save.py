import pickle
import os

def save_model(model, model_name: str) -> None:
    """
    Save the model to a pickle file.
    """
    model_dir = 'engine/models'
    os.makedirs(model_dir, exist_ok=True)  
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved to {model_path}")
