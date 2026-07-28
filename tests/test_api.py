import os
import sys
from fastapi.testclient import TestClient

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.app import app

client = TestClient(app)

def test_read_root():
    """
    Tests that the root endpoint responds with status 200 and details version info.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["status"] == "active"

def test_health_check():
    """
    Tests that the health check endpoint returns model load status and SQLite database checks.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["database_connected"] is True

def test_predict_endpoint():
    """
    Tests a single transaction prediction request.
    Verifies output data parameters: prediction, probability, risk level, action.
    """
    payload = {
        "transaction_amount": 1200.0,
        "merchant_category": "luxury_goods",
        "merchant_country": "RU",
        "payment_method": "credit_card",
        "device_type": "desktop",
        "transaction_type": "purchase",
        "customer_age": 28,
        "account_age_days": 15,
        "previous_fraud_history": 1,
        "transactions_last_24h": 5,
        "average_transaction_amount": 100.0,
        "distance_from_home": 4000.0,
        "card_present": 0,
        "international_transaction": 1
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "recommended_action" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_batch_endpoint():
    """
    Tests bulk batch transaction predictions.
    """
    payload = {
        "transactions": [
            {
                "transaction_amount": 15.0,
                "merchant_category": "groceries",
                "merchant_country": "US",
                "payment_method": "debit_card",
                "device_type": "desktop",
                "transaction_type": "purchase",
                "customer_age": 55,
                "account_age_days": 1000,
                "previous_fraud_history": 0,
                "transactions_last_24h": 1,
                "average_transaction_amount": 25.0,
                "distance_from_home": 1.2,
                "card_present": 1,
                "international_transaction": 0
            },
            {
                "transaction_amount": 5000.0,
                "merchant_category": "gambling",
                "merchant_country": "CN",
                "payment_method": "bank_transfer",
                "device_type": "mobile",
                "transaction_type": "transfer",
                "customer_age": 32,
                "account_age_days": 2,
                "previous_fraud_history": 1,
                "transactions_last_24h": 20,
                "average_transaction_amount": 50.0,
                "distance_from_home": 8500.0,
                "card_present": 0,
                "international_transaction": 1
            }
        ]
    }
    
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "batch_size" in data
    assert data["batch_size"] == 2
    assert "results" in data
    assert len(data["results"]) == 2
    
    # Assert result structures
    for res in data["results"]:
        assert "prediction" in res
        assert "fraud_probability" in res
        assert "risk_level" in res
        assert "recommended_action" in res
