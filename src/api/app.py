import os
import sys
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

# Add project root to sys.path for modular imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import config
from src.database.db_manager import DatabaseManager
from src.api.schema import TransactionInput, BatchTransactionInput
from src.api.predict import predict_transaction, model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "api.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("fraud_detection_api")

# Initialize database manager
db = DatabaseManager()

app = FastAPI(
    title="Banking Fraud Detection API",
    description="Production-style API for detecting credit card transaction fraud using Machine Learning.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """
    Root endpoint verifying API server status.
    """
    logger.info("Root endpoint pinged.")
    return {
        "service": "Banking Fraud Detection Platform API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
def health_check():
    """
    Performs critical service checks including database integrity and ML model load status.
    """
    health_status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": False
    }
    
    try:
        # Check SQLite connection
        conn = db.get_connection()
        conn.execute("SELECT 1")
        conn.close()
        health_status["database_connected"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        health_status["status"] = "unhealthy"
        health_status["database_error"] = str(e)

    if health_status["status"] == "unhealthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            content=health_status
        )
        
    return health_status

@app.post("/predict")
def predict(transaction: TransactionInput):
    """
    Predicts if a single transaction is fraudulent.
    Logs transaction detail and inference outputs to SQLite.
    """
    logger.info(f"Received single transaction prediction request for Customer ID: {transaction.customer_id}")
    try:
        tx_dict = transaction.model_dump()
        pred_res = predict_transaction(tx_dict)
        
        # Log to database
        db.log_prediction(pred_res["filled_transaction"], pred_res)
        
        return {
            "prediction": pred_res["prediction"],
            "fraud_probability": pred_res["fraud_probability"],
            "risk_level": pred_res["risk_level"],
            "recommended_action": pred_res["recommended_action"]
        }
    except Exception as e:
        logger.error(f"Error predicting single transaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict_batch")
def predict_batch(batch: BatchTransactionInput):
    """
    Predicts if a batch of transactions is fraudulent.
    Bulk-logs all transaction details and inference outputs to SQLite.
    """
    logger.info(f"Received batch transaction prediction request. Size: {len(batch.transactions)}")
    try:
        results = []
        filled_txs = []
        
        for tx in batch.transactions:
            tx_dict = tx.model_dump()
            pred_res = predict_transaction(tx_dict)
            
            results.append({
                "prediction": pred_res["prediction"],
                "fraud_probability": pred_res["fraud_probability"],
                "risk_level": pred_res["risk_level"],
                "recommended_action": pred_res["recommended_action"]
            })
            filled_txs.append(pred_res["filled_transaction"])
            
        # Bulk log to database
        db.log_prediction_batch(filled_txs, results)
        
        return {
            "batch_size": len(batch.transactions),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error predicting batch transactions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )