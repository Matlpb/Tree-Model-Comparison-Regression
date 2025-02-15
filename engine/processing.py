from sklearn.metrics import mean_squared_error, r2_score

from typing import Any, Dict, Tuple
import pandas as pd


def predict_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Predict values and display MSE and R² score.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on the testing set: {mse}")
    
    r2 = r2_score(y_test, y_pred)
    print(f"R² score on the testing set: {r2}")
