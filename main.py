import os
import pickle
import pandas as pd
from ETL.load_data import load_dataframes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from engine.processing import predict_model
from engine.save import save_model
from ETL.preprocessing import ApplyTransforms

def main():
    # Charger les données
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "house_prices")
    competition_name = "house-prices-advanced-regression-techniques"
    df_train, df_test = load_dataframes(project_dir, competition_name)

    # Charger le fichier qualitative_max_params.pkl
    params_path = os.path.join(base_dir, "transformer_params", "quantitative_max_params.pkl")
    if os.path.exists(params_path):
        with open(params_path, "rb") as f:
            qualitative_max_params = pickle.load(f)
        print("\n✅ Contenu de qualitative_max_params.pkl :")
        print(qualitative_max_params)
    else:
        print("\n⚠️ Fichier qualitative_max_params.pkl non trouvé !")

    # Séparer les caractéristiques (X) et la cible (y)
    X = df_train.drop(columns=['SalePrice', 'Id'])
    y = df_train['SalePrice']

    # Appliquer ApplyTransforms sur l'ENSEMBLE des données AVANT le split
    transformer = ApplyTransforms(save_dir=os.path.join(base_dir, "transformer_params"))
    X_encoded = transformer.fit_transform(X)

    # Split après transformation
    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Sauvegarde des fichiers transformés
    output_dir = os.path.join(base_dir, "house_prices")
    os.makedirs(output_dir, exist_ok=True)
    X_encoded.to_csv(os.path.join(output_dir, "X_encoded.csv"), index=False)

    print(f"Fichiers enregistrés dans {output_dir}")

    # Entraîner et sauvegarder les modèles
    model_types = [("extra_trees", ExtraTreesRegressor), ("gradient_boosting", GradientBoostingRegressor)]
    for model_name, model_class in model_types:
        print(f"\n🔹 Entraînement du modèle {model_name}...")

        pipeline = Pipeline(steps=[
            ('model', model_class(random_state=42))
        ])

        # Entraîner le modèle
        pipeline.fit(X_train_encoded, y_train)

        print(f"\n📊 Évaluation du modèle {model_name}:")
        predict_model(pipeline, X_test_encoded, y_test)

        # Sauvegarder le modèle
        save_model(pipeline, model_name)
        print(f"\n✅ Le modèle {model_name} a été sauvegardé.\n")

if __name__ == "__main__":
    main()
