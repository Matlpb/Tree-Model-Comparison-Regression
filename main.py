from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from engine.processing import predict_model
from engine.save import save_model
from ETL.load_data import load_dataframes
from ETL.preprocessing import ApplyTransforms

from typing import List, Tuple
import os


def main() -> None:
    """
    Main function to load the data, apply transformations, split the dataset, train models,
    evaluate them, and save the results.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "house_prices")
    competition_name = "house-prices-advanced-regression-techniques"
    df_train, df_test = load_dataframes(project_dir, competition_name)

    X = df_train.drop(columns=['SalePrice', 'Id'])
    y = df_train['SalePrice']

    saving_mode = True
    transformer = ApplyTransforms(save_dir=os.path.join(base_dir, "ETL", "transformer_params"), saving_mode=saving_mode)
    X_encoded = transformer.fit_transform(X)

    if not transformer.saving_mode:
        print("Transformation Rules:")
        print(transformer.transformation_rules)

        print("\nQuantitative Min Params:")
        print(transformer.quantitative_min_params)

        print("\nQuantitative Max Params:")
        print(transformer.quantitative_max_params)

        print("\nQualitative Min Params:")
        print(transformer.qualitative_min_params)

        print("\nQualitative Max Params:")
        print(transformer.qualitative_max_params)

    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    output_dir = os.path.join(base_dir, "house_prices/data")
    os.makedirs(output_dir, exist_ok=True)
    X_encoded.to_csv(os.path.join(output_dir, "X_encoded.csv"), index=False)

    print(f"Files saved to {output_dir}")

    model_types: List[Tuple[str, type]] = [("extra_trees", ExtraTreesRegressor), ("gradient_boosting", GradientBoostingRegressor)]
    for model_name, model_class in model_types:
        pipeline = Pipeline(steps=[('model', model_class(random_state=42))])
        pipeline.fit(X_train_encoded, y_train)

        print(f"\n📊 Evaluating model {model_name}:")
        predict_model(pipeline, X_test_encoded, y_test)

        save_model(pipeline, model_name)
        print(f"\n✅ Model {model_name} saved.\n")

if __name__ == "__main__":
    main()
