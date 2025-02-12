import pandas as pd
import pickle 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    """Applique MinMaxScaler aux colonnes spécifiées, en sauvegardant les min et max pour chaque colonne."""
    
    def __init__(self, cols, feature_range=(0, 1)):
        self.cols = cols
        self.feature_range = feature_range
        self.X_min = {}
        self.X_max = {}

    def fit(self, X, y=None):
        # Sauvegarder les valeurs minimales et maximales pour chaque colonne spécifiée
        for col in self.cols:
            self.X_min[col] = X[col].min()
            self.X_max[col] = X[col].max()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        # Appliquer la normalisation en utilisant les min et max enregistrés
        for col in self.cols:
            X_transformed[col] = (X_transformed[col] - self.X_min[col]) / (self.X_max[col] - self.X_min[col])
            # Appliquer la mise à l'échelle en fonction de feature_range
            X_transformed[col] = X_transformed[col] * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
            
            # Remplir les valeurs manquantes par 0
            X_transformed[col] = X_transformed[col].fillna(0)

        return X_transformed
    
    def save_params(self, filepath):
        # Sauvegarder les paramètres (min, max) pour chaque colonne
        with open(filepath, 'wb') as f:
            pickle.dump({'X_min': self.X_min, 'X_max': self.X_max}, f)
        
    def load_params(self, filepath):
        """Charge les valeurs min et max à partir du fichier."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            self.min_values = params.get('X_min', {})
            self.max_values = params.get('X_max', {})
        return params 


class FrequencyEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.encoding = {}

    def fit(self, X, y=None):
        for col in self.cols:
            freq = X[col].value_counts(normalize=True).to_dict()  
            self.encoding[col] = freq
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols:
            freq_dict = self.encoding[col]
            X_transformed[col] = X_transformed[col].map(freq_dict).fillna(0)  
        return X_transformed
    
    def save_params(self, filepath):
        # Sauvegarder les encodages de fréquence
        with open(filepath, 'wb') as f:
            pickle.dump(self.encoding, f)

    def load_params(self, filepath):
        # Charger les encodages de fréquence à partir du fichier
        with open(filepath, 'rb') as f:
            self.encoding = pickle.load(f)



class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """Applique un One-Hot Encoding classique aux colonnes spécifiées, en garantissant que les colonnes sont cohérentes entre l'entraînement et les tests."""
    
    def __init__(self, cols):
        self.cols = cols
        self.columns_ = {}  # Dictionnaire pour enregistrer les colonnes One-Hot

    def fit(self, X, y=None):
        """Enregistre les colonnes générées par OneHotEncoder pour chaque catégorie dans les colonnes spécifiées."""
        for col in self.cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
            self.columns_[col] = dummies.columns.tolist()  # Enregistre les noms de colonnes générés pour chaque catégorie
        return self

    def transform(self, X):
        """Applique l'encodage One-Hot aux colonnes spécifiées en respectant les colonnes d'entraînement."""
        X_transformed = X.copy()

        for col in self.cols:
            dummies = pd.get_dummies(X_transformed[col], prefix=col, drop_first=False)
            
            # S'assurer que les colonnes de test sont alignées avec celles de l'entraînement
            missing_cols = set(self.columns_[col]) - set(dummies.columns)
            for missing_col in missing_cols:
                dummies[missing_col] = 0  # Ajouter les colonnes manquantes avec des valeurs à 0

            # Réorganiser les colonnes de manière à correspondre à celles de l'entraînement
            dummies = dummies[self.columns_[col]]

            # Ajouter les colonnes encodées au dataframe transformé
            X_transformed = pd.concat([X_transformed, dummies], axis=1)
            
            # Supprimer la colonne d'origine
            X_transformed.drop(columns=[col], inplace=True)
            
            # Remplir les valeurs manquantes (si applicable) avec des zéros
            X_transformed = X_transformed.fillna(0)

        return X_transformed
    
    def save_params(self, filepath):
        """Sauvegarde les colonnes générées par One-Hot Encoding."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.columns_, f)

    def load_params(self, filepath):
        """Charge les colonnes générées par One-Hot Encoding à partir du fichier."""
        with open(filepath, 'rb') as f:
            self.columns_ = pickle.load(f)

