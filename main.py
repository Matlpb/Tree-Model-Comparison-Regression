import os
from ETL.load_data import load_dataframes, one_hot_encode_and_fill_na
from visualisations.visualisation import plot_sale_price_distribution
from engine.processing import predict_price
from Save.save import save

base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(base_dir, "house_prices")
competition_name = "house-prices-advanced-regression-techniques"

def main():
    """
    Main function to execute the full process.
    """
    df_train, df_test = load_dataframes(project_dir, competition_name)

    if df_train is not None:
        df_train_encoded = one_hot_encode_and_fill_na(df_train)
    if df_test is not None:
        df_test_encoded = one_hot_encode_and_fill_na(df_test)

    models = ["random_forest", "extra_trees", "xgboost", "adaboost"] 
    for model in models:
        print(f"Testing model: {model}")
        
        train_predictions, test_predictions, best_model, best_params, best_score, mse, r2 = predict_price(model, df_train_encoded, df_test_encoded)

        fig_train, fig_test = plot_sale_price_distribution(train_predictions, test_predictions, df_train_encoded, df_test_encoded)
        graphics = {
            "train_sale_price_distribution": fig_train,
            "test_sale_price_distribution": fig_test
        }
        
        save(df_test_encoded, graphics, test_predictions, method=model, 
             best_params=best_params, best_score=best_score, mse=mse, r2=r2)

if __name__ == "__main__":
    main()
