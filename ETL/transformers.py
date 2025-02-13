import pandas as pd
import pickle
from typing import Dict, Optional

from sklearn.base import BaseEstimator, TransformerMixin


class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    """Applies MinMaxScaler to the specified columns, saving min and max for each column."""
    
    def __init__(self, cols: list, feature_range: tuple = (0, 1), X_min: Optional[Dict[str, float]] = None, X_max: Optional[Dict[str, float]] = None):
        """
        Initializes the transformer.

        Args:
            cols (list): List of columns to apply the transformation to.
            feature_range (tuple): The desired range for the transformed values.
            X_min (Optional[Dict[str, float]]): Predefined minimum values for each column.
            X_max (Optional[Dict[str, float]]): Predefined maximum values for each column.
        """
        self.cols = cols
        self.feature_range = feature_range
        self.X_min = X_min if X_min is not None else {}
        self.X_max = X_max if X_max is not None else {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MinMaxScalerTransformer":
        """
        Fits the transformer by calculating the min and max values for each column if not provided.

        Args:
            X (pd.DataFrame): The input data.
            y (Optional[pd.Series]): The target values (not used).

        Returns:
            MinMaxScalerTransformer: The fitted transformer.
        """
        if not self.X_min or not self.X_max:
            for col in self.cols:
                self.X_min[col] = X[col].min()
                self.X_max[col] = X[col].max()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by applying the MinMaxScaler.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data.
        """
        X_transformed = X.copy()
        
        for col in self.cols:
            X_transformed[col] = pd.to_numeric(X_transformed[col], errors="coerce")
            min_val = self.X_min[col]
            max_val = self.X_max[col]

            if min_val == max_val:
                X_transformed[col] = 0
            else:
                X_transformed[col] = (X_transformed[col] - min_val) / (max_val - min_val)
                X_transformed[col] = X_transformed[col] * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

            X_transformed[col] = X_transformed[col].fillna(0)

        return X_transformed

    def save_params(self, filepath: str) -> None:
        """
        Saves the min and max values for each column to a file.

        Args:
            filepath (str): The file path where the parameters will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'X_min': self.X_min, 'X_max': self.X_max}, f)
        
    def load_params(self, filepath: str) -> Dict[str, Dict[str, float]]:
        """
        Loads the min and max values for each column from a file.

        Args:
            filepath (str): The file path from which the parameters will be loaded.

        Returns:
            Dict[str, Dict[str, float]]: The loaded min and max values.
        """
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            self.X_min = params.get('X_min', {})
            self.X_max = params.get('X_max', {})
        return params


class FrequencyEncoderTransformer(BaseEstimator, TransformerMixin):
    """Encodes specified columns by calculating the frequency of each value."""

    def __init__(self, cols: Optional[list] = None):
        """
        Initializes the transformer.

        Args:
            cols (Optional[list]): List of columns to apply the transformation to.
        """
        self.cols = cols or []
        self.encoding = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FrequencyEncoderTransformer":
        """
        Fits the transformer by calculating the frequency encoding for each column.

        Args:
            X (pd.DataFrame): The input data.
            y (Optional[pd.Series]): The target values (not used).

        Returns:
            FrequencyEncoderTransformer: The fitted transformer.
        """
        for col in self.cols:
            freq = X[col].value_counts(normalize=True).to_dict()
            self.encoding[col] = freq
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by applying frequency encoding.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data.
        """
        X_transformed = X.copy()
        for col in self.cols:
            freq_dict = self.encoding.get(col, {})
            X_transformed[col] = X_transformed[col].map(freq_dict).fillna(0)
        return X_transformed

    def save_params(self, filepath: str) -> None:
        """
        Saves the frequency encoding to a file.

        Args:
            filepath (str): The file path where the encoding will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.encoding, f)

    def load_params(self, filepath: str) -> Dict[str, Dict[str, float]]:
        """
        Loads the frequency encoding from a file.

        Args:
            filepath (str): The file path from which the encoding will be loaded.

        Returns:
            Dict[str, Dict[str, float]]: The loaded frequency encoding.
        """
        with open(filepath, 'rb') as f:
            self.encoding = pickle.load(f)


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """Applies One-Hot Encoding to the specified columns."""
    
    def __init__(self, cols: list):
        """
        Initializes the transformer.

        Args:
            cols (list): List of columns to apply the transformation to.
        """
        self.cols = cols
        self.columns_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OneHotEncoderTransformer":
        """
        Fits the transformer by generating One-Hot Encoding for each column.

        Args:
            X (pd.DataFrame): The input data.
            y (Optional[pd.Series]): The target values (not used).

        Returns:
            OneHotEncoderTransformer: The fitted transformer.
        """
        for col in self.cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=int)
            self.columns_[col] = dummies.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by applying One-Hot Encoding.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data.
        """
        X_transformed = X.copy()

        for col in self.cols:
            dummies = pd.get_dummies(X_transformed[col], prefix=col, drop_first=False, dtype=int)
            dummies = dummies[self.columns_[col]]
            X_transformed = pd.concat([X_transformed, dummies], axis=1)
            X_transformed.drop(columns=[col], inplace=True)

        return X_transformed.fillna(0)

    def save_params(self, filepath: str) -> None:
        """
        Saves the One-Hot Encoding columns to a file.

        Args:
            filepath (str): The file path where the encoding columns will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.columns_, f)

    def load_params(self, filepath: str) -> Dict[str, list]:
        """
        Loads the One-Hot Encoding columns from a file.

        Args:
            filepath (str): The file path from which the columns will be loaded.

        Returns:
            Dict[str, list]: The loaded One-Hot Encoding columns.
        """
        with open(filepath, 'rb') as f:
            self.columns_ = pickle.load(f)
        return self.columns_
