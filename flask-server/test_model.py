#Vérifier si le modèle .h5 est valide
from tensorflow.keras.models import load_model

# Chemin vers le modèle
model_path = "models/gru_model.h5"

try:
    # Charger le modèle
    model = load_model(model_path)
    print("Modèle chargé avec succès.")
    
    # Afficher le résumé du modèle
    model.summary()
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
