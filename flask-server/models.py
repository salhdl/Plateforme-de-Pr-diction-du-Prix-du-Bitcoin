# Fichier pour charger et exécuter les modèles de prédiction (GRU uniquement)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Fonction pour scaler les données
def scale_data(data, feature_column):
    """
    Scaler les données pour les amener dans la plage [0, 1].
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[feature_column]].values)
    return scaled_data, scaler

# Fonction pour créer des séquences
def create_sequences(data, time_steps=50):
    """
    Créer des séquences glissantes à partir des données scalées.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Fonction pour les prédictions avec le modèle GRU
def GRU_predict(data, model_path, time_steps=50):
    """
    Prédit les valeurs futures à l'aide d'un modèle GRU.

    Arguments :
    - data : pandas.DataFrame contenant les données d'entrée (doit inclure la colonne 'Close').
    - model_path : chemin vers le fichier .h5 du modèle GRU.
    - time_steps : nombre de pas de temps utilisés pour créer les séquences.

    Retourne :
    - predictions_original : prédictions dénormalisées sous forme de tableau numpy.
    """
    try:
        # Charger le modèle GRU
        print(f"Tentative de chargement du modèle depuis : {model_path}")
        model = load_model(model_path)
        print("Modèle chargé avec succès.")
        
        # Vérifier si la colonne 'Close' est présente
        if 'Close' not in data.columns:
            raise ValueError("La colonne 'Close' est manquante dans les données d'entrée.")

        # Préparer et scaler les données
        scaled_data, scaler = scale_data(data, 'Close')
        print("Données scalées avec succès.")

        # Créer les séquences pour les prédictions
        X_test, _ = create_sequences(scaled_data, time_steps)
        print(f"Nombre de séquences créées : {len(X_test)}")

        # Ajuster la forme pour l'entrée du modèle
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Effectuer les prédictions
        print("Début des prédictions avec le modèle GRU...")
        predictions = model.predict(X_test)
        print("Prédictions terminées.")

        # Dénormaliser les prédictions
        predictions_original = scaler.inverse_transform(predictions)
        print("Prédictions dénormalisées avec succès.")
        return predictions_original
    
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou des prédictions : {e}")
        raise
