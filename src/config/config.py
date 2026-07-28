import os

# ---------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SRC_MODELS_DIR = os.path.join(SRC_DIR, "models")

DB_DIR = os.path.join(DATA_DIR, "database")
DB_PATH = os.path.join(DATA_DIR, "predictions.db")

# Ensure required directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, SRC_MODELS_DIR, DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# File Paths
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "synthetic_transactions.csv")
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
SHAP_SUMMARY_PATH = os.path.join(MODELS_DIR, "shap_feature_importance.csv")

# ---------------------------------------------------------
# Synthetic Data Generation Configurations
# ---------------------------------------------------------
RANDOM_SEED = 42
TOTAL_TRANSACTIONS = 100000

MERCHANT_CATEGORIES = [
    "groceries", "gas_station", "online_shopping", "electronics", 
    "travel", "entertainment", "dining", "health", "utilities", 
    "cash_advance", "gambling", "luxury_goods"
]

PAYMENT_METHODS = ["credit_card", "debit_card", "prepaid_card", "apple_pay", "google_pay", "bank_transfer"]
DEVICE_TYPES = ["desktop", "mobile", "tablet", "wearable"]
OPERATING_SYSTEMS = ["Windows", "macOS", "Linux", "Android", "iOS"]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "Opera"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "JP", "AU", "IN", "CN", "BR", "MX", "RU", "ZA", "SG", "AE"]

# Fraud-Specific Risk Weights
HIGH_RISK_COUNTRIES = ["RU", "CN", "MX", "BR", "AE"]
HIGH_RISK_CATEGORIES = ["gambling", "luxury_goods", "cash_advance", "travel"]
HIGH_RISK_HOURS = [0, 1, 2, 3, 4]  # Midnight / Early morning hours

# ---------------------------------------------------------
# Risk Scoring and Decision Thresholds
# ---------------------------------------------------------
RISK_THRESHOLD_LOW = 0.20
RISK_THRESHOLD_MEDIUM = 0.60

ACTION_APPROVE = "Approve"
ACTION_OTP = "Require OTP"
ACTION_REVIEW = "Manual Review"
ACTION_BLOCK = "Block Transaction"

# ---------------------------------------------------------
# Model Training Settings
# ---------------------------------------------------------
SMOTE_SAMPLING_STRATEGY = 0.25  # Ratio of minority class to majority class
TEST_SIZE = 0.2
CV_FOLDS = 5
