from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from typing import Optional
from pathlib import Path  
import os
import pickle


import pandas as pd
from ETL.preprocessing import ApplyTransforms
from engine.load_models import load_model
from API.format_params_app import reformat_bounds, reformat_one_hot




app = FastAPI()

templates = Jinja2Templates(directory="API/templates")

@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request) -> HTMLResponse:
    """
    Renders the homepage template.
    """
    # Passer les données supplémentaires de quantitative_min_params au front-end
    return templates.TemplateResponse("interface.html", {
        "request": request,
        "quantitative_max_params": formatted_bounds,
        "qualitative_max_params": qualitative_max_params,
        "quantitative_min_params": quantitative_min_params,
        "qualitative_min_params": formatted_one_hot  # Ajout des données
    })



@app.get("/load_row/{row_number}")
async def load_row(row_number: int):
    """
    Loads the specified row from the test CSV file and replaces it in the transformed CSV.
    
    Args:
        row_number (int): The row number to load from the test CSV file.
        
    Returns:
        dict: The transformed data for the specified row.
    """
    try:
        # Path to your test CSV file
        test_file_path = "/Users/matthieu/Downloads/Tree-Model-Comparison-Regression-main/house_prices/data/test.csv"
        output_file_path = "/Users/matthieu/Downloads/Tree-Model-Comparison-Regression-main/API/data_test/X_test_transformed.csv"

        # Load the data from the CSV file
        df = pd.read_csv(test_file_path)

        # Ensure the row number is within the valid range
        if row_number < 1 or row_number > len(df):
            raise HTTPException(status_code=400, detail="Row number out of range.")
        
        # Get the row data (adjusting for zero-based indexing)
        row_data = df.iloc[row_number - 1]

        # Sanitize problematic float values (NaN, Infinity)
        for column in row_data.index:
            if isinstance(row_data[column], float):
                if pd.isna(row_data[column]) or row_data[column] == float('inf') or row_data[column] == float('-inf'):
                    row_data[column] = None

        # Convert the row to a DataFrame and apply transformations
        row_df = pd.DataFrame([row_data], columns=df.columns)
        
        # Transform the row (using your transformer logic)
        transformed_data = transformer.fit_transform(row_df)
        
        # Write transformed data to the output CSV, overwriting it
        transformed_df = pd.DataFrame(transformed_data, columns=df.columns)
        transformed_df.to_csv(output_file_path, index=False)
        
        return {"success": True, "message": "CSV has been overwritten with the transformed row."}

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

extra_trees_model = load_model('extra_trees')
gradient_boosting_model = load_model('gradient_boosting')

root_dir = Path(__file__).parent.parent
transformer = ApplyTransforms(
    save_dir=str(root_dir / "ETL/transformer_params"),
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

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "data_test")
os.makedirs(output_dir, exist_ok=True)


transformer.load_transformation_params()

quantitative_min_params = transformer.quantitative_min_params
quantitative_max_params = transformer.quantitative_max_params
qualitative_min_params = transformer.qualitative_min_params
qualitative_max_params = transformer.qualitative_max_params


X_min = quantitative_max_params.get('X_min', {})
X_max = quantitative_max_params.get('X_max', {})

# Réorganiser les bornes
formatted_bounds = reformat_bounds(X_min, X_max)
formatted_one_hot = reformat_one_hot(qualitative_min_params)





def transform_input_data(input_data: dict) -> pd.DataFrame:
    """
    Transforms the input data using the pre-defined transformer.
    
    Args:
        input_data (dict): The input data to be transformed.
        
    Returns:
        pd.DataFrame: The transformed input data.
    """
    try:
        input_df = pd.DataFrame([input_data], columns=train_columns)
        print("Original data:", input_df)


        transformed_df = transformer.fit_transform(input_df)
        transformed_df = pd.DataFrame(transformed_df)

        print("Transformed data:", transformed_df)
        output_file_path = os.path.join(output_dir, "X_test_transformed.csv")
        
        if os.path.exists(output_file_path):
            print(f"\n⚠️ The file {output_file_path} already exists. It will be deleted and rewritten.")
            os.remove(output_file_path)
        
        transformed_df.to_csv(output_file_path, index=False)
        print(f"\n✅ The file {output_file_path} has been successfully saved.")

        return transformed_df

    except Exception as e:
        print(f"\n❌ Error during transformation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during transformation")

@app.post("/predict/")
async def predict(input_data: dict) -> dict:
    """
    Makes a prediction based on the input data.

    Args:
        input_data (dict): The input data for prediction.
        
    Returns:
        dict: The prediction result.
    """
    try:
        print("\n🔹 New request received:", input_data)

        transformed_data = transform_input_data(input_data)

        model_selection = input_data.get('model_selection')
        if model_selection == 'extra_trees':
            print("\nUsing ExtraTrees model for prediction.")
            prediction = extra_trees_model.predict(transformed_data)
        elif model_selection == 'gradient_boosting':
            print("\nUsing GradientBoosting model for prediction.")
            prediction = gradient_boosting_model.predict(transformed_data)
        else:
            raise HTTPException(status_code=400, detail="Unknown model")

        print(f"\n✅ Prediction: {prediction[0]}")
        return {"prediction": prediction[0]}
    
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
