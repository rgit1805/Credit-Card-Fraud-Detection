import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime

# Add src to python path to import configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config

class DatabaseManager:
    """
    Manages the SQLite database for logging transaction details, model inferences,
    risk profiles, and mitigation actions.
    """
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        """
        Establishes a connection to the SQLite database.
        """
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """
        Initializes the predictions table if it does not exist.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            customer_id TEXT,
            transaction_amount REAL,
            merchant_category TEXT,
            merchant_name TEXT,
            merchant_country TEXT,
            customer_country TEXT,
            payment_method TEXT,
            transaction_type TEXT,
            device_type TEXT,
            operating_system TEXT,
            browser TEXT,
            customer_age INTEGER,
            account_age_days INTEGER,
            previous_fraud_history INTEGER,
            transactions_last_24h INTEGER,
            failed_transactions_last_24h INTEGER,
            average_transaction_amount REAL,
            distance_from_home REAL,
            card_present INTEGER,
            international_transaction INTEGER,
            ip_risk_score REAL,
            device_trust_score REAL,
            merchant_risk_score REAL,
            fraud_probability REAL,
            risk_level TEXT,
            recommended_action TEXT,
            prediction INTEGER
        );
        """
        with self.get_connection() as conn:
            conn.execute(query)
            conn.commit()
        print(f"Database initialized at: {self.db_path}")

    def log_prediction(self, transaction: dict, pred_result: dict) -> int:
        """
        Logs a single transaction and its model inference result.
        """
        query = """
        INSERT INTO predictions (
            timestamp, customer_id, transaction_amount, merchant_category, merchant_name,
            merchant_country, customer_country, payment_method, transaction_type,
            device_type, operating_system, browser, customer_age, account_age_days,
            previous_fraud_history, transactions_last_24h, failed_transactions_last_24h,
            average_transaction_amount, distance_from_home, card_present,
            international_transaction, ip_risk_score, device_trust_score, merchant_risk_score,
            fraud_probability, risk_level, recommended_action, prediction
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """
        
        # Ensure timestamp is formatted
        timestamp = transaction.get("transaction_time")
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        params = (
            timestamp,
            transaction.get("customer_id", "cust_unknown"),
            float(transaction.get("transaction_amount", 0.0)),
            transaction.get("merchant_category", "unknown"),
            transaction.get("merchant_name", "unknown_merchant"),
            transaction.get("merchant_country", "US"),
            transaction.get("customer_country", "US"),
            transaction.get("payment_method", "credit_card"),
            transaction.get("transaction_type", "purchase"),
            transaction.get("device_type", "desktop"),
            transaction.get("operating_system", "Windows"),
            transaction.get("browser", "Chrome"),
            int(transaction.get("customer_age", 30)),
            int(transaction.get("account_age_days", 365)),
            int(transaction.get("previous_fraud_history", 0)),
            int(transaction.get("transactions_last_24h", 1)),
            int(transaction.get("failed_transactions_last_24h", 0)),
            float(transaction.get("average_transaction_amount", 50.0)),
            float(transaction.get("distance_from_home", 0.0)),
            int(transaction.get("card_present", 0)),
            int(transaction.get("international_transaction", 0)),
            float(transaction.get("ip_risk_score", 0.1)),
            float(transaction.get("device_trust_score", 0.9)),
            float(transaction.get("merchant_risk_score", 0.1)),
            float(pred_result.get("fraud_probability", 0.0)),
            pred_result.get("risk_level", "LOW"),
            pred_result.get("recommended_action", config.ACTION_APPROVE),
            int(pred_result.get("prediction", 0))
        )

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def log_prediction_batch(self, transactions: list, pred_results: list):
        """
        Logs a list of transactions and their model inference results efficiently in a transaction batch.
        """
        query = """
        INSERT INTO predictions (
            timestamp, customer_id, transaction_amount, merchant_category, merchant_name,
            merchant_country, customer_country, payment_method, transaction_type,
            device_type, operating_system, browser, customer_age, account_age_days,
            previous_fraud_history, transactions_last_24h, failed_transactions_last_24h,
            average_transaction_amount, distance_from_home, card_present,
            international_transaction, ip_risk_score, device_trust_score, merchant_risk_score,
            fraud_probability, risk_level, recommended_action, prediction
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """
        
        batch_params = []
        for tx, pred in zip(transactions, pred_results):
            timestamp = tx.get("transaction_time")
            if not timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            params = (
                timestamp,
                tx.get("customer_id", "cust_unknown"),
                float(tx.get("transaction_amount", 0.0)),
                tx.get("merchant_category", "unknown"),
                tx.get("merchant_name", "unknown_merchant"),
                tx.get("merchant_country", "US"),
                tx.get("customer_country", "US"),
                tx.get("payment_method", "credit_card"),
                tx.get("transaction_type", "purchase"),
                tx.get("device_type", "desktop"),
                tx.get("operating_system", "Windows"),
                tx.get("browser", "Chrome"),
                int(tx.get("customer_age", 30)),
                int(tx.get("account_age_days", 365)),
                int(tx.get("previous_fraud_history", 0)),
                int(tx.get("transactions_last_24h", 1)),
                int(tx.get("failed_transactions_last_24h", 0)),
                float(tx.get("average_transaction_amount", 50.0)),
                float(tx.get("distance_from_home", 0.0)),
                int(tx.get("card_present", 0)),
                int(tx.get("international_transaction", 0)),
                float(tx.get("ip_risk_score", 0.1)),
                float(tx.get("device_trust_score", 0.9)),
                float(tx.get("merchant_risk_score", 0.1)),
                float(pred.get("fraud_probability", 0.0)),
                pred.get("risk_level", "LOW"),
                pred.get("recommended_action", config.ACTION_APPROVE),
                int(pred.get("prediction", 0))
            )
            batch_params.append(params)
            
        with self.get_connection() as conn:
            conn.executemany(query, batch_params)
            conn.commit()

    def get_history_df(self, limit: int = 1000) -> pd.DataFrame:
        """
        Retrieves recent predictions as a Pandas DataFrame for dashboard consumption.
        """
        query = f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        return df

    def clear_history(self):
        """
        Clears all historical prediction logs.
        """
        query = "DELETE FROM predictions"
        with self.get_connection() as conn:
            conn.execute(query)
            conn.commit()
        print("Database transaction logs cleared.")

if __name__ == "__main__":
    # Test initialization
    db = DatabaseManager()
    print("Database manager ready.")
