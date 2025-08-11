from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import yfinance as yf
from datetime import datetime, timedelta

# Configuration des logs
logging.basicConfig(
    filename='flask_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialiser l'application Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration globale
TIME_STEPS_IN = 10
TIME_STEPS_OUT = 2
MODEL_PATHS = {
    "2days": 'models/gru_model_2days.h5',
    "5days": 'models/gru_model_5days.h5',
    "7days": 'models/gru_model_7days.h5',
    "10days": 'models/gru_model_10days.h5'
}

models = {}
for key, path in MODEL_PATHS.items():
    try:
        models[key] = load_model(path)
        logging.info(f"Modèle {key} chargé avec succès depuis {path}")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {key} : {str(e)}")
        models[key] = None

def create_sequences(data, time_steps_in):
    """Créer des séquences pour la prédiction"""
    X = []
    for i in range(len(data) - time_steps_in + 1):
        X.append(data[i:(i + time_steps_in)].values)
    return np.array(X)

def prepare_data(ticker, start_date, end_date):
    """Préparer les données pour la prédiction"""
    try:
        # Télécharger les données
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        
        # Préparer les données
        data = df[['Volume', 'High', 'Low', 'Close', 'Open']].copy()
        
        # Normalisation
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=['Volume', 'High', 'Low', 'Close', 'Open'])
        
        return data_scaled, scaler, df.index, df
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données : {str(e)}")
        raise e

@app.route('/predictions/<model_key>', methods=['POST'])
def future_predictions_by_model(model_key):
    """Faire des prédictions pour les prochains jours en fonction du modèle choisi"""
    model = models.get(model_key)
    if model is None:
        return jsonify({"error": f"Le modèle {model_key} n'est pas disponible"}), 500

    try:
        # Récupérer les paramètres de la requête
        data = request.get_json()
        ticker = data.get('ticker', 'KO')  # Ticker par défaut
        future_days = int(data.get('days', 2))  # Nombre de jours à prédire

        # Charger les données les plus récentes
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Un an de données historiques
        data_scaled, scaler, dates, raw_data = prepare_data(
            ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )

        # Créer la séquence d'entrée à partir des données les plus récentes
        last_sequence = data_scaled[-TIME_STEPS_IN:].values.reshape(1, TIME_STEPS_IN, -1)

        predictions = []
        for _ in range(future_days):
            # Faire la prédiction pour un jour
            pred = model.predict(last_sequence)

            # Étendre les prédictions pour correspondre aux dimensions des données d'entrée
            extended_pred = np.zeros((1, 1, last_sequence.shape[2]))
            extended_pred[:, :, 3] = pred[:, 0]

            # Mettre à jour la séquence
            last_sequence = np.append(last_sequence[:, 1:, :], extended_pred, axis=1)

            # Stocker uniquement la valeur prédite pour "Close"
            predictions.append(pred[0][0])

        # Dénormaliser les prédictions
        predictions_array = np.zeros((future_days, last_sequence.shape[2]))
        predictions_array[:, 3] = predictions  # La colonne "Close"
        predictions_denorm = scaler.inverse_transform(predictions_array)[:, 3]

        # Générer des dates pour les prédictions futures
        prediction_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=future_days).strftime('%Y-%m-%d').tolist()

        return jsonify({
            "success": True,
            "predictions": predictions_denorm.tolist(),
            "dates": prediction_dates,
            "ticker": ticker,
            "prediction_count": future_days
        })

    except Exception as e:
        logging.error(f"Erreur dans /predictions/{model_key} : {e}")
        return jsonify({"error": str(e)}), 500

AVAILABLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'KO']

# Métriques du modèle (à mettre à jour avec vos vraies métriques)
MODEL_METRICS = {
    "accuracy": 0.95,
    "loss": 0.05,
    "val_accuracy": 0.92,
    "val_loss": 0.08,
    "mae": 1.23,
    "mse": 2.34,
    "rmse": 1.53,
    "r2": 0.89,
    "mape": 3.45
}

def create_sequences(data, time_steps_in):
    """Créer des séquences pour la prédiction"""
    X = []
    for i in range(len(data) - time_steps_in + 1):
        X.append(data[i:(i + time_steps_in)].values)
    return np.array(X)

def prepare_data(ticker, start_date, end_date):
    """Préparer les données pour la prédiction"""
    try:
        # Télécharger les données
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        
        # Préparer les données
        data = df[['Volume', 'High', 'Low', 'Close', 'Open']].copy()
        
        # Normalisation
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=['Volume', 'High', 'Low', 'Close', 'Open'])
        
        return data_scaled, scaler, df.index, df
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données : {str(e)}")
        raise e

# Charger le modèle
try:
    model = load_model(MODEL_PATH)
    logging.info("Modèle GRU chargé avec succès")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def home():
    """Endpoint de base"""
    return jsonify({"message": "Backend is running!"}), 200

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Vérifier le statut du modèle"""
    is_available = model is not None
    return jsonify({
        "model": "GRU",
        "available": is_available,
        "description": "Modèle GRU pour la prédiction des prix d'actions"
    })

@app.route('/data', methods=['GET'])
def get_data():
    """Récupérer les données historiques pour Coca-Cola (KO)"""
    ticker = 'KO'
    start = request.args.get('start', '2010-01-01')
    end = request.args.get('end', datetime.now().strftime('%Y-%m-%d'))

    logging.info(f"Attempting to fetch data for {ticker} from {start} to {end}")

    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start, end=end)

        if historical_data.empty:
            logging.error(f"No data available for ticker {ticker}")
            return jsonify({"error": f"No data available for ticker {ticker}"}), 404

        # Convertir les données en format JSON-compatible
        data_dict = historical_data.reset_index().to_dict('records')
        for record in data_dict:
            record['Date'] = record['Date'].strftime('%Y-%m-%d')

        logging.info(f"Successfully fetched {len(data_dict)} records")
        return jsonify({
            "data": data_dict,
            "ticker": ticker,
            "start": start,
            "end": end,
            "total_records": len(data_dict)
        })
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return jsonify({
            "error": "Failed to fetch stock data",
            "details": str(e)
        }), 500

@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    """Liste des tickers disponibles"""
    return jsonify({"tickers": AVAILABLE_TICKERS})

@app.route('/api/gru-metrics', methods=['GET'])
def gru_metrics():
    """Récupérer les métriques du modèle"""
    return jsonify({
        "model": "GRU",
        "metrics": MODEL_METRICS,
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/test-gru', methods=['POST'])
def test_gru():
    """Tester le modèle avec des données factices"""
    if model is None:
        return jsonify({"error": "Le modèle GRU n'est pas disponible"}), 500

    try:
        # Créer des données factices
        dummy_data = np.random.rand(1, TIME_STEPS_IN, 5)
        
        # Normaliser les données
        scaler = MinMaxScaler(feature_range=(0, 1))
        dummy_flat = dummy_data.reshape(-1, dummy_data.shape[-1])
        dummy_scaled = scaler.fit_transform(dummy_flat)
        dummy_scaled = dummy_scaled.reshape(dummy_data.shape)

        # Faire la prédiction
        predictions = model.predict(dummy_scaled)
        
        # Préparer pour inverse_transform
        pred_full = np.zeros((len(predictions), 5))
        pred_full[:, 3] = predictions[:, -1]
        
        # Dénormaliser
        predictions_final = scaler.inverse_transform(pred_full)[:, 3]

        return jsonify({
            "success": True,
            "predictions": predictions_final.tolist(),
            "test_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        logging.error(f"Erreur lors du test GRU : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/gru', methods=['POST'])
def predict_gru():
    """Faire des prédictions pour Coca-Cola (KO)"""
    if model is None:
        logging.error("Le modèle GRU n'est pas disponible")
        return jsonify({"error": "Le modèle GRU n'est pas disponible"}), 500

    try:
        logging.info("Requête reçue avec les données : %s", request.json)

        # Paramètres fixes
        ticker = 'KO'  # Ticker fixé à KO
        data = request.get_json()
        start_date = data.get('start', '2010-01-01')
        end_date = data.get('end', datetime.now().strftime('%Y-%m-%d'))
        logging.info("Période demandée : %s à %s", start_date, end_date)

        # Préparer les données
        data_scaled, scaler, dates, raw_data = prepare_data(ticker, start_date, end_date)
        logging.info("Données préparées avec succès")

        # Créer la séquence d'entrée
        X = create_sequences(data_scaled, TIME_STEPS_IN)
        logging.info("Séquences créées : %s", X.shape)

        # Faire la prédiction
        predictions = model.predict(X)
        logging.info("Prédictions générées : %s", predictions)

        # Dénormaliser les prédictions
        pred_array = np.zeros((len(predictions), 5))
        pred_array[:, 3] = predictions[:, -1]
        predictions_denorm = scaler.inverse_transform(pred_array)[:, 3]
        logging.info("Prédictions dénormalisées : %s", predictions_denorm)

        # Préparer les dates pour les prédictions
        prediction_dates = pd.date_range(
            start=dates[-len(predictions)],
            periods=len(predictions),
            freq='D'
        ).strftime('%Y-%m-%d').tolist()
        

        return jsonify({
            "success": True,
            "predictions": predictions_denorm.tolist(),
            "dates": prediction_dates,
            "ticker": ticker,
            "actual_prices": raw_data['Close'][-len(predictions):].tolist(),
            "prediction_count": len(predictions)
        })

    except Exception as e:
        logging.error(f"Erreur dans /predict/gru : {e}")   
        return jsonify({"error": str(e)}), 500

    
# Add this debug endpoint to your Flask app to test connectivity
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.after_request
def after_request(response):
    """Ajouter les headers CORS nécessaires"""
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)