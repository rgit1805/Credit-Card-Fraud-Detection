import requests

API_URL = "http://127.0.0.1:8000/predict"

def predict(data):

    response = requests.post(API_URL, json=data)

    return response.json()