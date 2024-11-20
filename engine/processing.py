from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor

def prepare_data(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the data for model training.
    """
    X = df_train.drop('SalePrice', axis=1)
    y = df_train['SalePrice']
    return X, y

def train_model(model_class: Any, param_grid: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any], float]:
    """
    Trains a model using grid search with cross-validation.
    """
    model = model_class()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {-grid_search.best_score_}")
    return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_

def get_model_and_params(model_type: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Returns the model class and parameter grid based on the model type.
    """
    tree_params = {
        'n_estimators': [100, 150, 200,250],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10,13]
    }
    boost_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4]
    }
    if model_type == "random_forest":
        return RandomForestRegressor, tree_params
    elif model_type == "extra_trees":
        return ExtraTreesRegressor, tree_params
    elif model_type == "xgboost":
        return XGBRegressor, {**boost_params, 'max_depth': [3, 5, 7]}
    elif model_type == "adaboost":
        return AdaBoostRegressor, boost_params
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def predict_price(
    model_type: str, 
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Any, Dict[str, Any], float, float, float]:
    """
    Full process: Prepare data, train model using grid search with cross-validation, 
    evaluate performance, and predict on the test set.

    Parameters:
    model_type (str): Type of model to train ("random_forest", "extra_trees", "xgboost", "adaboost").
    df_train (pd.DataFrame): Training data including the target column 'SalePrice'.
    df_test (pd.DataFrame): Test data excluding the target column 'SalePrice'.

    """
    X_train, y_train = prepare_data(df_train)
    model_class, param_grid = get_model_and_params(model_type)
    best_model, best_params, best_score = train_model(model_class, param_grid, X_train, y_train)
    train_predictions = best_model.predict(X_train)
    mse = mean_squared_error(y_train, train_predictions)
    r2 = r2_score(y_train, train_predictions)
    X_test = df_test.drop('SalePrice', axis=1, errors='ignore')
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    test_predictions = best_model.predict(X_test)
    return train_predictions, test_predictions, best_model, best_params, best_score, mse, r2
