import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

# Add src to python path to import configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config

def generate_synthetic_data(
    num_transactions: int = config.TOTAL_TRANSACTIONS, 
    seed: int = config.RANDOM_SEED,
    output_path: str = config.RAW_DATA_PATH
):
    """
    Generates a high-quality synthetic dataset of transactions with realistic fraud patterns.
    """
    print(f"Initializing synthetic data generation: target {num_transactions} rows...")
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker()
    fake.seed_instance(seed)
    
    # Define proportions
    fraud_ratio = 0.012  # ~1.2% fraud transactions
    num_fraud = int(num_transactions * fraud_ratio)
    num_normal = num_transactions - num_fraud
    
    # Generate customer pools
    num_customers = 5000
    customer_ids = [f"cust_{1000 + i}" for i in range(num_customers)]
    customer_genders = {cid: random.choice(["M", "F", "Other"]) for cid in customer_ids}
    customer_ages = {cid: random.randint(18, 85) for cid in customer_ids}
    customer_account_ages = {cid: random.randint(30, 3650) for cid in customer_ids}
    customer_base_country = {cid: random.choice(config.COUNTRIES) for cid in customer_ids}
    
    # Average transaction amount for each customer (log-normal distribution)
    customer_avg_amounts = {
        cid: round(np.random.lognormal(mean=3.5, sigma=0.8), 2) for cid in customer_ids
    }
    
    # Generate transactions lists
    records = []
    
    start_date = datetime.now() - timedelta(days=90)
    
    # Helper to generate device details
    def get_device_details():
        dev = random.choice(config.DEVICE_TYPES)
        if dev == "desktop":
            os_name = random.choice(["Windows", "macOS", "Linux"])
            browser = random.choice(["Chrome", "Firefox", "Edge", "Opera"])
        elif dev == "mobile" or dev == "tablet":
            os_name = random.choice(["Android", "iOS"])
            browser = random.choice(["Chrome", "Safari", "Firefox", "Opera"])
        else:  # wearable
            os_name = random.choice(["Android", "iOS"])
            browser = "Safari" if os_name == "iOS" else "Chrome"
        return dev, os_name, browser

    # 1. Generate Normal Transactions
    print(f"Generating {num_normal} normal transactions...")
    for i in range(num_normal):
        tx_id = f"tx_{1000000 + i}"
        cust_id = random.choice(customer_ids)
        
        # Normal amount centered around customer average
        base_amt = customer_avg_amounts[cust_id]
        tx_amount = round(np.random.gamma(shape=3.0, scale=base_amt/3.0), 2)
        tx_amount = max(0.5, tx_amount)  # Minimum 50c
        
        # Transaction time spread across 24 hours, favoring daytime
        hours = list(range(24))
        hour_weights = [1 if h in config.HIGH_RISK_HOURS else 4 for h in hours]
        tx_hour = random.choices(hours, weights=hour_weights, k=1)[0]
        tx_time = start_date + timedelta(
            days=random.randint(0, 89), 
            hours=tx_hour, 
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        category = random.choice(config.MERCHANT_CATEGORIES)
        # Avoid high risk categories mostly
        if category in config.HIGH_RISK_CATEGORIES and random.random() > 0.15:
            category = random.choice([c for c in config.MERCHANT_CATEGORIES if c not in config.HIGH_RISK_CATEGORIES])
            
        merchant_name = f"{fake.company()} {random.choice(['Shop', 'Inc', 'Ltd', 'Group', 'Store'])}"
        
        cust_country = customer_base_country[cust_id]
        # Most transactions are domestic
        if random.random() < 0.95:
            merch_country = cust_country
            distance = round(random.uniform(0.1, 45.0), 2)
            card_present = random.choices([1, 0], weights=[0.75, 0.25], k=1)[0]
        else:
            merch_country = random.choice([c for c in config.COUNTRIES if c != cust_country])
            distance = round(random.uniform(100.0, 5000.0), 2)
            card_present = 0
            
        pay_method = random.choice(config.PAYMENT_METHODS)
        tx_type = random.choices(["purchase", "transfer", "withdrawal"], weights=[0.85, 0.10, 0.05], k=1)[0]
        dev_type, os_name, browser = get_device_details()
        
        # Normal customer indicators
        transactions_24h = random.randint(1, 6)
        failed_24h = random.choices([0, 1, 2], weights=[0.95, 0.04, 0.01], k=1)[0]
        prev_fraud = 1 if random.random() < 0.005 else 0
        
        # Risk scores (Normal are mostly low, with occasional random spikes)
        ip_risk = round(np.random.beta(a=1, b=5), 2)
        dev_trust = round(np.random.beta(a=5, b=1.5), 2)
        merch_risk = round(np.random.beta(a=1.5, b=5), 2)
        
        # Extra calculated features
        is_intl = 1 if merch_country != cust_country else 0
        is_wknd = 1 if tx_time.weekday() >= 5 else 0
        
        records.append({
            "transaction_id": tx_id,
            "customer_id": cust_id,
            "transaction_amount": tx_amount,
            "transaction_time": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "merchant_category": category,
            "merchant_name": merchant_name,
            "merchant_country": merch_country,
            "customer_country": cust_country,
            "payment_method": pay_method,
            "transaction_type": tx_type,
            "device_type": dev_type,
            "operating_system": os_name,
            "browser": browser,
            "account_age_days": customer_account_ages[cust_id],
            "customer_age": customer_ages[cust_id],
            "customer_gender": customer_genders[cust_id],
            "average_transaction_amount": customer_avg_amounts[cust_id],
            "transactions_last_24h": transactions_24h,
            "failed_transactions_last_24h": failed_24h,
            "previous_fraud_history": prev_fraud,
            "card_present": card_present,
            "international_transaction": is_intl,
            "distance_from_home": distance,
            "ip_risk_score": ip_risk,
            "device_trust_score": dev_trust,
            "merchant_risk_score": merch_risk,
            "hour_of_day": tx_hour,
            "day_of_week": tx_time.weekday(),
            "is_weekend": is_wknd,
            "fraud": 0
        })

    # 2. Generate Fraud Transactions
    print(f"Generating {num_fraud} fraudulent transactions using realistic profiles...")
    for i in range(num_fraud):
        tx_id = f"tx_fraud_{1000000 + i}"
        cust_id = random.choice(customer_ids)
        cust_country = customer_base_country[cust_id]
        
        # Select a fraud profile
        profile = random.choice(["A", "B", "C", "D"])
        
        # Base templates
        tx_hour = random.choice(config.HIGH_RISK_HOURS)
        tx_time = start_date + timedelta(
            days=random.randint(0, 89), 
            hours=tx_hour, 
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        category = random.choice(config.MERCHANT_CATEGORIES)
        merch_country = cust_country
        distance = round(random.uniform(5.0, 50.0), 2)
        card_present = random.choice([1, 0])
        tx_amount = round(customer_avg_amounts[cust_id] * random.uniform(1.1, 1.8), 2)
        transactions_24h = random.randint(2, 5)
        failed_24h = random.choice([0, 1])
        prev_fraud = 1 if random.random() < 0.1 else 0
        
        ip_risk = round(random.uniform(0.3, 0.7), 2)
        dev_trust = round(random.uniform(0.4, 0.8), 2)
        merch_risk = round(random.uniform(0.3, 0.6), 2)
        
        pay_method = random.choice(config.PAYMENT_METHODS)
        tx_type = "purchase"
        dev_type, os_name, browser = get_device_details()
        
        if profile == "A":
            # Profile A: High Spender / Account Takeover
            tx_amount = round(customer_avg_amounts[cust_id] * random.uniform(6.0, 15.0), 2)
            tx_amount = max(tx_amount, random.uniform(800.0, 5000.0))  # Force a high value
            dev_trust = round(random.uniform(0.0, 0.25), 2)  # New/untrusted device
            ip_risk = round(random.uniform(0.75, 1.0), 2)  # Risky IP
            category = random.choice(["electronics", "luxury_goods", "travel"])
            tx_hour = random.choice(config.HIGH_RISK_HOURS)  # Midnight
            
        elif profile == "B":
            # Profile B: Card Cloning / Impossible Travel
            merch_country = random.choice([c for c in config.COUNTRIES if c != cust_country])
            distance = round(random.uniform(1500.0, 12000.0), 2)
            card_present = 1  # Card present in another country = impossible travel
            is_intl = 1
            ip_risk = round(random.uniform(0.6, 0.95), 2)
            dev_trust = round(random.uniform(0.1, 0.4), 2)
            category = random.choice(["groceries", "gas_station", "luxury_goods"])
            
        elif profile == "C":
            # Profile C: Velocity Attack / Rapid Fire Transactions
            transactions_24h = random.randint(15, 45)
            failed_24h = random.randint(3, 8)
            tx_amount = round(random.uniform(10.0, 150.0), 2)
            dev_trust = round(random.uniform(0.0, 0.3), 2)
            ip_risk = round(random.uniform(0.8, 1.0), 2)
            category = "online_shopping"
            pay_method = "credit_card"
            
        elif profile == "D":
            # Profile D: Suspicious Merchant Cashout / High Risk Target
            category = random.choice(config.HIGH_RISK_CATEGORIES)
            merch_country = random.choice(config.HIGH_RISK_COUNTRIES)
            merch_risk = round(random.uniform(0.8, 1.0), 2)
            ip_risk = round(random.uniform(0.7, 1.0), 2)
            prev_fraud = 1  # Often happens to customers with previous vulnerabilities
            tx_amount = round(customer_avg_amounts[cust_id] * random.uniform(3.0, 7.0), 2)
            card_present = 0

        # Adjust time for specific profile triggers
        tx_time = start_date + timedelta(
            days=random.randint(0, 89), 
            hours=tx_hour, 
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        is_intl = 1 if merch_country != cust_country else 0
        is_wknd = 1 if tx_time.weekday() >= 5 else 0
        merchant_name = f"Suspicious_{fake.company()}" if merch_risk > 0.85 else f"{fake.company()} {random.choice(['Shop', 'Inc', 'Ltd'])}"

        records.append({
            "transaction_id": tx_id,
            "customer_id": cust_id,
            "transaction_amount": tx_amount,
            "transaction_time": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "merchant_category": category,
            "merchant_name": merchant_name,
            "merchant_country": merch_country,
            "customer_country": cust_country,
            "payment_method": pay_method,
            "transaction_type": tx_type,
            "device_type": dev_type,
            "operating_system": os_name,
            "browser": browser,
            "account_age_days": customer_account_ages[cust_id],
            "customer_age": customer_ages[cust_id],
            "customer_gender": customer_genders[cust_id],
            "average_transaction_amount": customer_avg_amounts[cust_id],
            "transactions_last_24h": transactions_24h,
            "failed_transactions_last_24h": failed_24h,
            "previous_fraud_history": prev_fraud,
            "card_present": card_present,
            "international_transaction": is_intl,
            "distance_from_home": distance,
            "ip_risk_score": ip_risk,
            "device_trust_score": dev_trust,
            "merchant_risk_score": merch_risk,
            "hour_of_day": tx_hour,
            "day_of_week": tx_time.weekday(),
            "is_weekend": is_wknd,
            "fraud": 1
        })

    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Shuffle the dataset so fraud is mixed in randomly
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Ensure raw directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Data generation complete! Saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution:\n{df['fraud'].value_counts(normalize=True) * 100}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()
