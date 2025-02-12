from sklearn.metrics import mean_squared_error, r2_score

from typing import Any, Dict, Tuple
import pandas as pd





def predict_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Prédit les valeurs avec le modèle et affiche le MSE et R² score.
    """
    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Calculer le Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error sur l'ensemble de test: {mse}")
    
    # Calculer le R² score
    r2 = r2_score(y_test, y_pred)
    print(f"R² score sur l'ensemble de test: {r2}")
