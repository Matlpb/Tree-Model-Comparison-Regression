import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def save(
        df_test: pd.DataFrame,
        graphics: dict,
        predictions: pd.Series,
        method: str,
        best_params: dict,
        best_score: float,
        mse: float,
        r2: float
    ) -> None:
    """
    Saves graphics, predictions, and model information in a structured way under the 'all_save' folder.
    
    Args:
        df_test (pd.DataFrame): The test dataset to save predictions for.
        graphics (dict): Dictionary containing graphics (matplotlib Figures).
        predictions (pd.Series): Predictions made for df_test.
        method (str): Name of the method (e.g., 'random_forest').
        best_params (dict): Best hyperparameters of the model.
        best_score (float): Best score obtained during grid search.
        mse (float): Mean Squared Error from model validation.
        r2 (float): R² Score from model validation.
    """
    base_path = os.path.join("all_save", method)
    
    os.makedirs(base_path, exist_ok=True)
    
    for name, fig in graphics.items():
        fig.savefig(os.path.join(base_path, f"{name}.png"))
    
    predictions_df = pd.DataFrame({"Id": df_test.index, "Predicted_SalePrice": predictions})
    predictions_df.to_csv(os.path.join(base_path, f"{method}_predictions.csv"), index=False)

    model_info_df = pd.DataFrame({
        "Model": [method],
        "Best_Params": [str(best_params)],
        "Best_Score": [best_score],
        "Validation_MSE": [mse],
        "Validation_R2": [r2]
    })
    model_info_df.to_csv(os.path.join(base_path, f"{method}_model_info.csv"), index=False)