# Fichier pour récupérer ou traiter les données historiques nécessaires.
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler  # Nécessaire pour le scaling

# Liste des actions par défaut (incluant Coca-Cola)
DEFAULT_STOCKS = ["KO", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "NVDA"]

# Fonction pour scaler les données
def scale_data(data, feature_column):
    """
    Scale les données d'une colonne spécifique.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[feature_column]].values)
    return scaled_data, scaler

# Récupérer des données historiques pour une période donnée
def fetch_data_for_gru(ticker, start, end):
    """
    Récupère les données historiques pour une action donnée entre les dates start et end.
    """
    print(f"Récupération des données pour {ticker} de {start} à {end}")
    try:
        stock = yf.Ticker(ticker)
        print("Instance de yfinance créée avec succès.")
        
        # Télécharger les données
        data = stock.history(start=start, end=end)
        print(f"Données brutes récupérées :\n{data.head()}")

        if data.empty:
            raise ValueError(f"Pas de données disponibles pour le ticker {ticker} entre {start} et {end}")
        
        # Conserver uniquement les colonnes nécessaires pour le modèle
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Les colonnes suivantes sont manquantes dans les données pour {ticker}: {missing_columns}")
        
        data = data[required_columns].dropna()
        print(f"Données après filtrage :\n{data.head()}")

        # Préparer les données pour le retour
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'date'}, inplace=True)  # Renommer uniquement 'Date' si nécessaire
        
        # Vérification finale pour s'assurer que 'Close' est bien dans les données
        if 'Close' not in data.columns:
            raise ValueError("La colonne 'Close' n'a pas été trouvée après le traitement des données.")

        print(f"Données finales prêtes à retourner :\n{data.head()}")
        return data
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la récupération des données pour {ticker}: {e}")

# Récupérer les données pour un ensemble d'actions par défaut
def fetch_default_stock_data(duration_days=30):
    """
    Récupère les données historiques pour les actions par défaut pour une durée donnée.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=duration_days)).strftime('%Y-%m-%d')

        all_stock_data = {}
        for ticker in DEFAULT_STOCKS:
            data = fetch_data_for_gru(ticker, start_date, end_date)
            all_stock_data[ticker] = data
        
        return all_stock_data
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la récupération des données par défaut: {e}")

# Récupérer les données des derniers jours pour des graphiques
def fetch_latest_close_prices(tickers, days=7):
    """
    Récupère les prix de clôture pour les derniers jours.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        close_prices = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            if not data.empty:
                close_prices[ticker] = data['Close'].tolist()
            else:
                close_prices[ticker] = []

        return close_prices
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la récupération des prix de clôture: {e}")

# Récupérer les données pour un rapport global
def fetch_summary_report(tickers, duration_days=30):
    """
    Récupère un rapport récapitulatif pour les actions données sur une période.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=duration_days)).strftime('%Y-%m-%d')

        report = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if not data.empty:
                volume = data['Volume'].mean()
                latest_close = data['Close'].iloc[-1]
                change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100

                report.append({
                    "Ticker": ticker,
                    "Average Volume": volume,
                    "Latest Close": latest_close,
                    "Change (%)": round(change, 2),
                    "Data Points": len(data)
                })
            else:
                report.append({
                    "Ticker": ticker,
                    "Average Volume": None,
                    "Latest Close": None,
                    "Change (%)": None,
                    "Data Points": 0
                })

        return pd.DataFrame(report)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la récupération du rapport récapitulatif: {e}")

# Tester les fonctions individuellement
if __name__ == "__main__":
    # Test pour Coca-Cola
    ticker = "KO"
    start_date = "2024-01-01"
    end_date = "2024-11-21"

    try:
        print(f"Test : récupération des données pour {ticker} de {start_date} à {end_date}")
        data = fetch_data_for_gru(ticker, start_date, end_date)
        print("Colonnes disponibles dans les données récupérées :", list(data.columns))
        print("Données récupérées avec succès :")
        print(data.head())
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
