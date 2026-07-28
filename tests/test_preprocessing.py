import os
import sys
import pandas as pd
import numpy as np

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.predict import fill_missing_features, preprocess_single, feature_order

def test_fill_missing_features():
    """
    Verifies that optional fields are correctly populated using our backend heuristics.
    """
    # Sample minimum input parameters entered by dashboard form
    mock_input = {
        "transaction_amount": 450.0,
        "merchant_category": "luxury_goods",
        "merchant_country": "MX",
        "payment_method": "credit_card",
        "device_type": "mobile",
        "transaction_type": "purchase",
        "customer_age": 30,
        "account_age_days": 100,
        "previous_fraud_history": 0,
        "transactions_last_24h": 2,
        "average_transaction_amount": 50.0,
        "distance_from_home": 1500.0,
        "card_present": 0,
        "international_transaction": 1
    }
    
    filled = fill_missing_features(mock_input)
    
    # Assert missing columns are filled
    assert "hour_of_day" in filled
    assert "day_of_week" in filled
    assert "is_weekend" in filled
    assert filled["customer_country"] == "MX"  # Matches merchant country as default
    assert filled["operating_system"] == "Android"  # Heuristic for mobile device
    assert filled["failed_transactions_last_24h"] == 0
    assert filled["ip_risk_score"] is not None
    assert filled["device_trust_score"] is not None
    assert filled["merchant_risk_score"] is not None
    
    # High risk country should boost merchant risk score heuristic
    assert filled["merchant_risk_score"] > 0.5

def test_preprocess_single_dimensions():
    """
    Verifies that preprocessing maps input fields to the exact 85-dimensional scaled/OHE schema.
    """
    mock_input = {
        "transaction_amount": 25.0,
        "merchant_category": "groceries",
        "merchant_country": "US",
        "payment_method": "debit_card",
        "device_type": "desktop",
        "transaction_type": "purchase",
        "customer_age": 45,
        "account_age_days": 365,
        "previous_fraud_history": 0,
        "transactions_last_24h": 1,
        "average_transaction_amount": 30.0,
        "distance_from_home": 2.5,
        "card_present": 1,
        "international_transaction": 0
    }
    
    filled = fill_missing_features(mock_input)
    processed_df = preprocess_single(filled)
    
    assert isinstance(processed_df, pd.DataFrame)
    assert processed_df.shape[0] == 1
    # Check that processed columns match feature order list from OHE setup
    assert list(processed_df.columns) == feature_order
    assert processed_df.shape[1] == 85  # Expected dimensions
