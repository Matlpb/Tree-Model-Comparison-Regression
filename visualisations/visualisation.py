import matplotlib.pyplot as plt
import seaborn as sns
from engine.processing import prepare_data
import pandas as pd
from typing import Tuple

def plot_sale_price_distribution(predicted_saleprice_train: pd.Series, predicted_saleprice_test: pd.Series, 
                                 df_train_encoded: pd.DataFrame, df_test_encoded: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plots the distribution of actual SalePrice and predicted SalePrice for both df_train and df_test.
    
    Returns:
        tuple: Two matplotlib figures, one for df_train and one for df_test.
    """
    fig_train, ax_train = plt.subplots(figsize=(14, 12))
    sns.histplot(df_train_encoded['SalePrice'], color='g', kde=True, bins=100, label='Actual SalePrice (Train)', stat="density", alpha=0.6, ax=ax_train)
    sns.histplot(predicted_saleprice_train, color='b', kde=True, bins=100, label='Predicted SalePrice (Train)', stat="density", alpha=0.6, ax=ax_train)
    ax_train.set_title('Distribution of Actual vs Predicted SalePrice for df_train')
    ax_train.set_xlabel('SalePrice')
    ax_train.set_ylabel('Density')
    ax_train.legend()

    fig_test, ax_test = plt.subplots(figsize=(14, 12))
    sns.histplot(predicted_saleprice_test, color='r', kde=True, bins=100, label='Predicted SalePrice (Test)', stat="density", alpha=0.6, ax=ax_test)
    ax_test.set_title('Distribution of Predicted SalePrice for df_test')
    ax_test.set_xlabel('SalePrice')
    ax_test.set_ylabel('Density')
    ax_test.legend()

    return fig_train, fig_test
