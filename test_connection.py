import requests

def test_connection():
    try:
        # Test GET endpoint
        print("Testing GET /data ...")
        response = requests.get('http://127.0.0.1:5000/data')
        if response.status_code == 200:
            print("GET /data Response:", response.json())
        else:
            print(f"GET /data failed with status code {response.status_code}")

        # Test POST endpoint
        print("\nTesting POST /predict/gru ...")
        payload = {"ticker": "KO", "start": "2024-01-01", "end": "2024-12-31"}
        response = requests.post('http://127.0.0.1:5000/predict/gru', json=payload)
        if response.status_code == 200:
            print("POST /predict/gru Response:", response.json())
        else:
            print(f"POST /predict/gru failed with status code {response.status_code}")

    except Exception as e:
        print("Erreur lors de la connexion au backend :", e)

if __name__ == "__main__":
    test_connection()
