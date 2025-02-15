from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd
from typing import List
from pathlib import Path
import os
from ETL.preprocessing import ApplyTransforms
from engine.load_models import load_model
from API.format_params_app import reformat_bounds, reformat_one_hot

app = FastAPI()
templates = Jinja2Templates(directory="API/templates")

BASE_DIR = Path(__file__).resolve().parent.parent
test_file_path = BASE_DIR / "house_prices/data/test.csv"
output_file_path = BASE_DIR / "API/data_test/X_test_transformed.csv"
os.makedirs(output_file_path.parent, exist_ok=True)

extra_trees_model = load_model("extra_trees")
gradient_boosting_model = load_model("gradient_boosting")

transformer = ApplyTransforms(
    save_dir=str(BASE_DIR / "ETL/transformer_params"), saving_mode=False
)
transformer.load_transformation_params()

quantitative_min_params = transformer.quantitative_min_params
quantitative_max_params = transformer.quantitative_max_params
qualitative_min_params = transformer.qualitative_min_params
qualitative_max_params = transformer.qualitative_max_params

formatted_bounds = reformat_bounds(
    quantitative_max_params.get("X_min", {}), quantitative_max_params.get("X_max", {})
)
formatted_one_hot = reformat_one_hot(qualitative_min_params)


@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    """Returns the main interface page."""
    return templates.TemplateResponse(
        "interface.html",
        {
            "request": request,
            "quantitative_max_params": formatted_bounds,
            "qualitative_max_params": qualitative_max_params,
            "quantitative_min_params": quantitative_min_params,
            "qualitative_min_params": formatted_one_hot,
        },
    )


@app.get("/load_row/{row_number}")
async def load_row(row_number: int):
    """Loads a row from the test dataset, applies transformations, and saves it."""
    try:
        df = pd.read_csv(test_file_path)
        if row_number < 1 or row_number > len(df):
            raise HTTPException(status_code=400, detail="Row number out of range.")

        row_data = df.iloc[row_number - 1]
        row_data = row_data.where(pd.notna(row_data), None)
        row_df = pd.DataFrame([row_data], columns=df.columns)
        transformed_data = transformer.fit_transform(row_df)
        transformed_df = pd.DataFrame(transformed_data, columns=df.columns)
        transformed_df.to_csv(output_file_path, index=False)

        return {
            "success": True,
            "message": "CSV has been overwritten with the transformed row.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


train_df = pd.read_csv(BASE_DIR / "house_prices/data/train.csv")
train_columns: List[str] = [
    col for col in train_df.columns if col not in ["Id", "SalePrice"]
]


def transform_input_data(input_data) -> pd.DataFrame:
    """Transforms input data using pre-defined transformer and saves the result."""
    try:
        if isinstance(input_data, int):
            df = pd.read_csv(test_file_path)
            if input_data < 1 or input_data > len(df):
                raise HTTPException(status_code=400, detail="Row number out of range.")
            row_data = df.iloc[input_data - 1].to_dict()
            input_df = pd.DataFrame([row_data], columns=train_columns)
            print(input_df)
            print(input_data)
        else:
            input_df = pd.DataFrame([input_data], columns=train_columns)
            print(input_df)

        transformed_df = transformer.fit_transform(input_df)
        transformed_df = pd.DataFrame(transformed_df)

        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        transformed_df.to_csv(output_file_path, index=False)

        return transformed_df
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during transformation")


@app.post("/predict/")
async def predict(input_data: dict):
    """Predicts house price based on input data."""
    try:
        if isinstance(input_data.get("row_number"), int):
            transformed_data = transform_input_data(input_data["row_number"])
        else:
            transformed_data = transform_input_data(input_data)

        model_selection = input_data.get("model_selection")
        if model_selection == "extra_trees":
            prediction = extra_trees_model.predict(transformed_data)
        elif model_selection == "gradient_boosting":
            prediction = gradient_boosting_model.predict(transformed_data)
        else:
            raise HTTPException(status_code=400, detail="Unknown model")

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
