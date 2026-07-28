import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to python path to import configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config

# Load model and preprocessors
if not os.path.exists(config.BEST_MODEL_PATH):
    raise FileNotFoundError(f"Champion model not found at {config.BEST_MODEL_PATH}. Train models first.")
if not os.path.exists(config.SCALER_PATH) or not os.path.exists(config.ENCODERS_PATH):
    raise FileNotFoundError("Preprocessor pickles not found. Run preprocessing first.")

model = joblib.load(config.BEST_MODEL_PATH)
scaler = joblib.load(config.SCALER_PATH)
encoder_payload = joblib.load(config.ENCODERS_PATH)

encoder = encoder_payload["encoder"]
feature_order = encoder_payload["feature_order"]
numerical_cols = encoder_payload["numerical_cols"]
categorical_cols = encoder_payload["categorical_cols"]
binary_cols = encoder_payload["binary_cols"]

def fill_missing_features(tx: dict) -> dict:
    """
    Fills in optional transaction/customer features using smart heuristics to match
    the 30-feature schema required by the ML model.
    """
    tx_copy = tx.copy()
    
    # 1. Date/Time extraction if missing
    now = datetime.now()
    tx_hour = tx_copy.get("hour_of_day")
    tx_day = tx_copy.get("day_of_week")
    tx_weekend = tx_copy.get("is_weekend")
    
    if tx_hour is None:
        tx_copy["hour_of_day"] = now.hour
    if tx_day is None:
        tx_copy["day_of_week"] = now.weekday()
    if tx_weekend is None:
        tx_copy["is_weekend"] = 1 if now.weekday() >= 5 else 0

    # 2. Category / Country / Device Defaults
    if not tx_copy.get("customer_country"):
        tx_copy["customer_country"] = tx_copy.get("merchant_country", "US")
    if not tx_copy.get("merchant_name"):
        tx_copy["merchant_name"] = f"Merchant_{tx_copy.get('merchant_category', 'Retail')}"
    if not tx_copy.get("customer_id"):
        tx_copy["customer_id"] = "cust_unknown"
        
    dev_type = tx_copy.get("device_type", "desktop")
    if not tx_copy.get("operating_system"):
        if dev_type == "desktop":
            tx_copy["operating_system"] = "Windows"
        else:
            tx_copy["operating_system"] = "Android"
            
    if not tx_copy.get("browser"):
        tx_copy["browser"] = "Chrome"
        
    if tx_copy.get("failed_transactions_last_24h") is None:
        tx_copy["failed_transactions_last_24h"] = 0
        
    if tx_copy.get("customer_gender") is None:
        tx_copy["customer_gender"] = "Other"

    # 3. Smart Risk Score Population
    # If not provided, we calculate risk scores based on suspicious variables
    tx_amount = float(tx_copy.get("transaction_amount", 0.0))
    avg_amount = float(tx_copy.get("average_transaction_amount", 50.0))
    transactions_24h = int(tx_copy.get("transactions_last_24h", 1))
    prev_fraud = int(tx_copy.get("previous_fraud_history", 0))
    failed_24h = int(tx_copy.get("failed_transactions_last_24h", 0))
    merch_cat = tx_copy.get("merchant_category", "")
    merch_country = tx_copy.get("merchant_country", "")

    if tx_copy.get("ip_risk_score") is None:
        # High IP risk if high transaction count or previous fraud
        if transactions_24h > 15 or failed_24h > 2 or prev_fraud == 1:
            tx_copy["ip_risk_score"] = round(np.random.uniform(0.75, 0.99), 2)
        else:
            tx_copy["ip_risk_score"] = round(np.random.uniform(0.01, 0.25), 2)

    if tx_copy.get("device_trust_score") is None:
        # Low trust score if high frequency/failed transactions or previous fraud
        if transactions_24h > 15 or failed_24h > 2 or prev_fraud == 1:
            tx_copy["device_trust_score"] = round(np.random.uniform(0.01, 0.30), 2)
        else:
            tx_copy["device_trust_score"] = round(np.random.uniform(0.75, 0.99), 2)

    if tx_copy.get("merchant_risk_score") is None:
        # High risk if high risk category or high risk country
        if merch_cat in config.HIGH_RISK_CATEGORIES or merch_country in config.HIGH_RISK_COUNTRIES:
            tx_copy["merchant_risk_score"] = round(np.random.uniform(0.70, 0.99), 2)
        else:
            tx_copy["merchant_risk_score"] = round(np.random.uniform(0.01, 0.30), 2)

    # 4. Binary Flags Verification
    tx_copy["international_transaction"] = 1 if tx_copy["customer_country"] != tx_copy["merchant_country"] else 0

    return tx_copy

def preprocess_single(tx_filled: dict) -> pd.DataFrame:
    """
    Transforms filled transaction features using standard scaling and one-hot encoding.
    """
    df = pd.DataFrame([tx_filled])
    
    # Scale numerical features
    scaled_numerical = scaler.transform(df[numerical_cols])
    df_scaled_num = pd.DataFrame(scaled_numerical, columns=numerical_cols)
    
    # Encode categorical features
    encoded_categorical = encoder.transform(df[categorical_cols])
    encoded_cat_names = encoder.get_feature_names_out(categorical_cols)
    df_encoded_cat = pd.DataFrame(encoded_categorical, columns=encoded_cat_names)
    
    # Keep binary features
    df_binary = df[binary_cols].reset_index(drop=True)
    
    # Concatenate features
    df_processed = pd.concat([df_scaled_num, df_encoded_cat, df_binary], axis=1)
    
    # Reorder columns
    df_processed = df_processed[feature_order]
    
    return df_processed

def predict_transaction(transaction: dict) -> dict:
    """
    Ingests a raw transaction, fills missing values, normalizes features, 
    makes fraud predictions, and applies action policies.
    """
    try:
        # Fill optional fields
        filled_tx = fill_missing_features(transaction)
        
        # Preprocess features
        processed_df = preprocess_single(filled_tx)
        
        # Inference
        prediction = int(model.predict(processed_df)[0])
        probability = float(model.predict_proba(processed_df)[0][1])
        
        # Risk categorization
        if probability < config.RISK_THRESHOLD_LOW:
            risk_level = "LOW"
            action = config.ACTION_APPROVE
        elif probability < config.RISK_THRESHOLD_MEDIUM:
            risk_level = "MEDIUM"
            # Require Manual Review if high amount, otherwise require OTP
            if filled_tx.get("transaction_amount", 0.0) > 1000.0:
                action = config.ACTION_REVIEW
            else:
                action = config.ACTION_OTP
        else:
            risk_level = "HIGH"
            action = config.ACTION_BLOCK
            
        return {
            "prediction": prediction,
            "fraud_probability": probability,
            "risk_level": risk_level,
            "recommended_action": action,
            "filled_transaction": filled_tx
        }
    except Exception as e:
        print(f"Error during model prediction: {str(e)}")
        raise e